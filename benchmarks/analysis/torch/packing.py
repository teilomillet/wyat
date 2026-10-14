"""Experimental exact FP32 gate/up packing; no attention or precision change."""

from __future__ import annotations

import copy
import hashlib
import json
from argparse import ArgumentParser, Namespace
from collections.abc import Callable, Sequence
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import cast

import torch
from torch import nn
from transformers import Qwen3Model
from transformers.models.qwen3 import modeling_qwen3
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

from benchmarks.analysis.evidence import freeze_sources, process_diagnostic
from benchmarks.analysis.reference import read_reference
from wyat import Model
from wyat.backends.wyat.torch import WyatScorer
from wyat.benchmark import (
    BenchmarkCycle,
    BenchmarkEngine,
    BenchmarkProtocol,
    BenchmarkReport,
    QualityGate,
    _run,
    compare_cycles,
)
from wyat.evaluation import _input_digest, case_from_dict, percentile
from wyat.wire import decode_json

from .process import launch
from .threads import DATASET_SHA256


class PackedMLP(nn.Module):
    """Own packed storage; preserve the maintained activation and down projection."""

    width: int
    gate_up: nn.Linear
    down_proj: nn.Linear
    act_fn: Callable[[torch.Tensor], torch.Tensor]

    def __init__(self, original: Qwen3MLP) -> None:
        super().__init__()
        if original._forward_hooks or original._forward_pre_hooks:
            raise ValueError("Experimental replacement requires an unhooked MLP")
        gate = original.gate_proj
        up = original.up_proj
        if (
            gate.bias is not None
            or up.bias is not None
            or gate.weight.dtype != torch.float32
            or up.weight.dtype != torch.float32
            or gate.weight.device.type != "cpu"
            or up.weight.device != gate.weight.device
            or gate.weight.shape != up.weight.shape
        ):
            raise ValueError("Packing requires matching biasless CPU FP32 projections")
        self.width = gate.out_features
        self.gate_up = nn.Linear(
            gate.in_features, 2 * self.width, bias=False, device="meta"
        )
        weight = torch.cat((gate.weight.detach(), up.weight.detach()), dim=0)
        self.gate_up.weight = nn.Parameter(weight, requires_grad=False)
        if not (
            torch.equal(weight[: self.width], gate.weight)
            and torch.equal(weight[self.width :], up.weight)
        ):
            raise ValueError("Packing changed a source weight")
        self.act_fn = cast(Callable[[torch.Tensor], torch.Tensor], original.act_fn)
        self.down_proj = cast(nn.Linear, original.down_proj)
        self.training = original.training
        self.gate_up.training = original.training

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(value).split(self.width, dim=-1)
        return cast(torch.Tensor, self.down_proj(self.act_fn(gate) * up))


def packed_backbone(original: Qwen3Model) -> Qwen3Model:
    """Copy module ownership, share unchanged weights, replace only MLPs."""
    candidate = copy.copy(original)
    candidate._modules = original._modules.copy()
    layers: list[nn.Module] = []
    for block in cast(Sequence[nn.Module], original.layers):
        copied = copy.copy(block)
        copied._modules = block._modules.copy()
        copied.mlp = PackedMLP(cast(Qwen3MLP, block.mlp))
        layers.append(copied)
    candidate.layers = nn.ModuleList(layers)
    candidate.layers.training = original.layers.training
    # Do not call eval()/train() here: those traverse shared unchanged modules.
    return candidate


def variant(scorer: WyatScorer) -> WyatScorer:
    return WyatScorer(
        packed_backbone(scorer.backbone),
        scorer.head,
        scorer.tokenizer,
        device=scorer.device,
        max_length=scorer.max_length,
        max_options=scorer.max_options,
        token_cache_size=0,
        model_id=scorer.model_id,
    )


def study(model_path: Path, output: Path, *, batch_size: int = 8) -> bool:
    if type(batch_size) is not int or batch_size not in (1, 8):
        raise ValueError("Packing confirmation requires batch size 1 or 8")
    root = Path(__file__).resolve().parents[3]
    dataset = root / "benchmarks/real/banking77-train-dev-menu6.jsonl"
    reference_path = root / "benchmarks/real/torch-fp32-reference77.json"
    sources = freeze_sources(
        {
            "runner": Path(__file__),
            "launcher": Path(__file__).with_name("process.py"),
            "reference": root / "benchmarks/analysis/reference.py",
            "benchmark": root / "wyat/benchmark.py",
            "backend": root / "wyat/backends/wyat/torch.py",
            "encoding": root / "wyat/backends/wyat/encoding.py",
            "artifact": root / "wyat/backends/wyat/artifact.py",
            "qwen3": Path(modeling_qwen3.__file__),
        },
        output,
    )
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != DATASET_SHA256:
        raise ValueError("Declared dataset changed")
    cases = tuple(
        case_from_dict(decode_json(line)) for line in dataset.read_text().splitlines()
    )
    if len(cases) != 77:
        raise ValueError("Packing requires every declared case")
    reference_hash = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    gate = QualityGate()
    plan = BenchmarkProtocol(repeats=5, warmup=1, quality_gate=gate)
    payload: dict[str, object] = {
        "scope": "all77 training-source development cases; exposure unknown",
        "hypothesis": "Exact FP32 gate/up packing improves throughput",
        "completed": False,
        "quality_passed": False,
        "performance_passed": False,
        "accepted": False,
        "phase": "loading",
        "dataset_sha256": DATASET_SHA256,
        "reference_sha256": reference_hash,
        "source_sha256": sources.hashes(),
        "source_archive": sources.archive(),
        "threads": 8,
        "batch_size": batch_size,
        "max_length": 512,
        "token_cache_size": 0,
        "versions": {name: version(name) for name in ("torch", "transformers")},
        "quality_gate": asdict(gate),
        "performance_gate": {
            "speedup_p50_minimum": 1.02,
            "speedup_p05_minimum_exclusive": 1.0,
            "throughput_ratio_minimum_exclusive": 1.0,
            "candidate_p95_maximum_ratio": 1.0,
        },
        "paired_order": "five alternating baseline/candidate pairs, warmup one per arm",
    }

    def write() -> None:
        output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")

    write()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(8)
    try:
        scorer = WyatScorer.from_directory(
            model_path, max_length=512, token_cache_size=0
        )
        reference = read_reference(
            reference_path, cases, scorer, dataset_sha256=DATASET_SHA256
        )
        baseline = BenchmarkEngine(
            "cpu-fp32-separate-gate-up", Model(scorer, scorer.model_id), batch_size
        )
        candidate = BenchmarkEngine(
            "cpu-fp32-packed-gate-up",
            Model(variant(scorer), scorer.model_id),
            batch_size,
        )
        before = process_diagnostic()
        payload.update(
            model_id=scorer.model_id, interop_threads=torch.get_num_interop_threads()
        )
        measured: tuple[list[BenchmarkCycle], list[BenchmarkCycle]] = ([], [])
        with output.with_suffix(".cycles.jsonl").open("w") as journal:

            def run(engine: BenchmarkEngine, stage: str, repeat: int) -> BenchmarkCycle:
                cycle = _run(engine, cases)
                journal.write(
                    json.dumps(
                        {
                            "stage": stage,
                            "repeat": repeat,
                            "arm": engine.name,
                            "cycle": cycle.to_dict(),
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
                journal.flush()
                return cycle

            screens = (run(baseline, "screen", 0), run(candidate, "screen", 0))
            quality = compare_cycles((reference, reference), screens, gate)
            payload.update(
                screen=[cycle.to_dict() for cycle in screens],
                screen_quality=asdict(quality),
            )
            write()
            if not quality.passed:
                payload.update(completed=True, phase="rejected screen")
                write()
                return False
            warm = (run(baseline, "warmup", 0), run(candidate, "warmup", 0))
            for repeat in range(plan.repeats):
                order = (0, 1) if repeat % 2 == 0 else (1, 0)
                for index in order:
                    measured[index].append(
                        run((baseline, candidate)[index], "measured", repeat)
                    )
            cycles = screens + warm + tuple(measured[0]) + tuple(measured[1])
            frozen_gate = compare_cycles((reference,) * len(cycles), cycles, gate)
            report = BenchmarkReport(
                _input_digest(cases),
                cases,
                plan,
                baseline,
                candidate,
                tuple(measured[0]),
                tuple(measured[1]),
                compare_cycles(measured[0], measured[1], gate),
                (warm[0],),
                (warm[1],),
            )
            left = [cycle.latency_ms for cycle in measured[0]]
            right = [cycle.latency_ms for cycle in measured[1]]
            speedups = [a / b for a, b in zip(left, right, strict=True)]
            performance = (
                percentile(speedups, 0.5) >= 1.02
                and percentile(speedups, 0.05) > 1
                and sum(left) / sum(right) > 1
                and percentile(right, 0.95) <= percentile(left, 0.95)
            )
            quality_passed = report.passed and frozen_gate.passed
            payload.update(
                report=report.to_dict(),
                reference_quality=asdict(frozen_gate),
                quality_passed=quality_passed,
                performance_passed=performance,
                accepted=quality_passed and performance,
                completed=True,
                phase="complete",
            )
        payload["diagnostics"] = {
            "before": asdict(before),
            "after": asdict(process_diagnostic()),
        }
        write()
        return payload["quality_passed"] is True
    except Exception as error:
        payload.update(
            error=f"{type(error).__name__}: {error}",
            phase="failed",
            accepted=False,
            completed=False,
            quality_passed=False,
            performance_passed=False,
        )
        write()
        raise
    finally:
        torch.set_num_threads(previous_threads)


class Arguments(Namespace):
    model: Path
    output: Path
    worker: bool
    batch_size: int


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--batch-size", type=int, choices=(1, 8), default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/analysis/results/torch-packing77.json"),
    )
    parser.add_argument("--worker", action="store_true", help="internal bounded worker")
    args = parser.parse_args(namespace=Arguments())
    code = (
        int(not study(args.model, args.output, batch_size=args.batch_size))
        if args.worker
        else launch(
            __spec__.name if __spec__ else "benchmarks.analysis.torch.packing",
            args.model,
            args.output,
            timeout=180,
            worker_args=("--batch-size", str(args.batch_size)),
        )
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
