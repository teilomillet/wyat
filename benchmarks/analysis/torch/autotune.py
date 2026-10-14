"""One real FP32 gate projection: frozen max-autotune feasibility, not an engine win.

The supervisor fixes a private cache, freezing, output-code logging and a 120s
total wall cap. The first eight canonical real requests supply one layer-zero
post-attention normalized activation. No other projection is compiled.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from argparse import ArgumentParser, Namespace
from collections.abc import Callable
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import cast

import torch
from safetensors.torch import save_file
from torch import nn
from transformers.models.qwen3 import modeling_qwen3

from benchmarks.analysis.evidence import freeze_sources
from benchmarks.analysis.reference import read_reference
from wyat import Model
from wyat.backends.wyat.torch import WyatScorer
from wyat.benchmark import (
    BenchmarkCycle,
    BenchmarkEngine,
    QualityGate,
    _run,
    compare_cycles,
)
from wyat.evaluation import case_from_dict, percentile
from wyat.wire import decode_json

from .process import launch
from .threads import DATASET_SHA256


def output_gate(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, object]:
    """Additive allclose gate for a projection, distinct from model max tolerances."""
    if (
        reference.shape != actual.shape
        or actual.dtype != torch.float32
        or reference.dtype != torch.float32
        or actual.device.type != "cpu"
        or reference.device != actual.device
    ):
        raise ValueError("Projection shape or dtype changed")
    finite = bool(torch.isfinite(actual).all().item())
    error = (reference - actual).abs()
    return {
        "passed": finite
        and bool(torch.allclose(reference, actual, atol=1e-3, rtol=1e-3)),
        "finite": finite,
        "max_absolute_error": float(error.max().item()) if finite else None,
        "atol": 1e-3,
        "rtol": 1e-3,
    }


def retain_tensors(tensors: dict[str, torch.Tensor], path: Path) -> str:
    """Safetensors forbids aliases; archive independent exact named values."""
    save_file(
        {name: value.detach().contiguous().clone() for name, value in tensors.items()},
        path,
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probe(model_path: Path, output: Path) -> bool:
    if os.environ.get("TORCHINDUCTOR_FREEZING") != "1":
        raise ValueError("Use the supervisor to set freezing before Torch import")
    root = Path(__file__).resolve().parents[3]
    torch_root = Path(torch.__file__).parent
    sources = freeze_sources(
        {
            "runner": Path(__file__),
            "launcher": Path(__file__).with_name("process.py"),
            "reference": root / "benchmarks/analysis/reference.py",
            "benchmark": root / "wyat/benchmark.py",
            "backend": root / "wyat/backends/wyat/torch.py",
            "qwen3": Path(modeling_qwen3.__file__),
            "inductor_conditions": torch_root / "_inductor/utils.py",
            "cpp_gemm": torch_root / "_inductor/codegen/cpp_gemm_template.py",
            "cpp_micro_gemm": torch_root / "_inductor/codegen/cpp_micro_gemm.py",
            "cpu_isa": torch_root / "_inductor/cpu_vec_isa.py",
        },
        output,
    )
    dataset = root / "benchmarks/real/banking77-train-dev-menu6.jsonl"
    reference_path = root / "benchmarks/real/torch-fp32-reference77.json"
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != DATASET_SHA256:
        raise ValueError("Declared dataset changed")
    cases = tuple(
        case_from_dict(decode_json(line)) for line in dataset.read_text().splitlines()
    )
    if len(cases) != 77:
        raise ValueError("Expected all 77 reference cases before taking first eight")
    receipt: dict[str, object] = {
        "scope": "one layer-zero gate Linear; no complete-engine speed claim",
        "passed": False,
        "completed": False,
        "phase": "loading",
        "dataset_sha256": DATASET_SHA256,
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "source_sha256": sources.hashes(),
        "source_archive": sources.archive(),
        "versions": {name: version(name) for name in ("torch", "transformers")},
        "protocol": {
            "dtype": "float32",
            "device": "cpu",
            "threads": 8,
            "batch_size": 8,
            "max_length": 512,
            "token_cache_size": 0,
            "mode": "max-autotune",
            "frozen_weights": True,
            "repeats": 5,
            "warmup": "first compiled call; untimed eager reference",
            "projection_gate": {
                "atol": 1e-3,
                "rtol": 1e-3,
                "rule": "abs(error) <= atol + rtol * abs(reference); "
                "distinct from full-model symmetric max gate",
            },
            "full_model_quality_gate": asdict(QualityGate()),
        },
        "environment": {
            name: os.environ.get(name)
            for name in (
                "TORCHINDUCTOR_FREEZING",
                "TORCHINDUCTOR_CACHE_DIR",
                "TORCH_LOGS",
                "TORCHINDUCTOR_COMPILE_THREADS",
            )
        },
        "conditions": "FP32/FP32 NEON CPP micro-GEMM, constant frozen weight, "
        "stride-one input K; generated-code log establishes actual dispatch",
    }

    def write() -> None:
        output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")

    write()
    tensors: dict[str, torch.Tensor] = {}
    raw = output.with_suffix(".tensors.safetensors")

    def retain() -> None:
        receipt.update(raw_tensors=str(raw), raw_sha256=retain_tensors(tensors, raw))

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(8)
    try:
        scorer = WyatScorer.from_directory(
            model_path, max_length=512, token_cache_size=0
        )
        saved = read_reference(
            reference_path, cases, scorer, dataset_sha256=DATASET_SHA256
        )
        block = scorer.backbone.layers[0]
        captured: list[torch.Tensor] = []

        def capture(module: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            del module
            captured.append(args[0].detach().clone())

        handle = cast(nn.Module, block.mlp).register_forward_pre_hook(capture)
        try:
            cycle = _run(
                BenchmarkEngine(
                    "unchanged-cpu-fp32", Model(scorer, scorer.model_id), 8
                ),
                cases[:8],
            )
        finally:
            handle.remove()
        frozen = BenchmarkCycle(0, saved.rows[:8], 0, 0)
        gate = compare_cycles((frozen,), (cycle,), QualityGate())
        receipt.update(
            model_id=scorer.model_id,
            capture=cycle.to_dict(),
            capture_quality=asdict(gate),
        )
        if not gate.passed or len(captured) != 1:
            receipt["phase"] = "rejected capture"
            receipt["completed"] = True
            write()
            return False
        value = captured[0].reshape(-1, captured[0].shape[-1]).contiguous()
        mlp = cast(modeling_qwen3.Qwen3MLP, block.mlp)
        original = mlp.gate_proj
        linear = copy.deepcopy(original)
        linear.requires_grad_(False)
        linear.eval()
        if (
            value.dtype != torch.float32
            or value.device.type != "cpu"
            or not torch.isfinite(value).all().item()
        ):
            raise ValueError("Capture must be finite CPU FP32")
        if (
            linear.weight.dtype != torch.float32
            or linear.bias is not None
            or not torch.equal(linear.weight, original.weight)
        ):
            raise ValueError("Projection must preserve exact biasless FP32 weight")
        receipt.update(
            input_shape=list(value.shape),
            weight_shape=list(linear.weight.shape),
            weight_preserved=True,
            phase="cold compilation and first call",
        )
        write()
        tensors = {
            "input": value,
            "weight": linear.weight.detach(),
        }
        with torch.no_grad():
            reference = cast(torch.Tensor, linear(value))
            started = time.perf_counter()
            compiled = cast(
                Callable[[torch.Tensor], torch.Tensor],
                torch.compile(linear, mode="max-autotune"),
            )
            actual = compiled(value)
            cold_seconds = time.perf_counter() - started
            quality = output_gate(reference, actual)
            tensors.update(reference=reference, compiled_cold=actual)
            retain()
            receipt.update(
                cold_seconds=cold_seconds, cold_quality=quality, phase="warm pairs"
            )
            write()
            timings: dict[str, list[float]] = {"eager": [], "compiled": []}
            qualities: list[dict[str, object]] = []
            orders: list[tuple[str, str]] = []
            if quality["passed"] is True:
                functions: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
                    "eager": linear,
                    "compiled": compiled,
                }
                for repeat in range(5):
                    order = (
                        ("eager", "compiled")
                        if repeat % 2 == 0
                        else ("compiled", "eager")
                    )
                    orders.append(order)
                    for name in order:
                        start = time.perf_counter()
                        result = functions[name](value)
                        timings[name].append((time.perf_counter() - start) * 1000)
                        observed = output_gate(reference, result)
                        qualities.append({"repeat": repeat, "arm": name, **observed})
                        # Retain distinct raw outputs; hashes expose repeat variation.
                        digest = hashlib.sha256(result.numpy().tobytes()).hexdigest()
                        tensors.setdefault("output_" + digest, result)
                        observed["sha256"] = digest
                        qualities[-1]["sha256"] = digest
                    receipt.update(
                        timings_ms=timings, qualities=qualities, orders=orders
                    )
                    write()
            retain()
            if timings["compiled"]:
                ratios = [
                    left / right
                    for left, right in zip(
                        timings["eager"], timings["compiled"], strict=True
                    )
                ]
                receipt.update(
                    paired_speedups=ratios, paired_speedup_p50=percentile(ratios, 0.5)
                )
            passed = quality["passed"] is True and all(
                row["passed"] is True for row in qualities
            )
            receipt.update(passed=passed, completed=True, phase="complete")
            write()
            return passed
    except Exception as error:
        receipt.update(
            error=f"{type(error).__name__}: {error}",
            phase="failed",
            completed=False,
            passed=False,
        )
        if tensors:
            try:
                retain()
            except Exception as archive_error:
                receipt["archive_error"] = (
                    f"{type(archive_error).__name__}: {archive_error}"
                )
        write()
        raise
    finally:
        torch.set_num_threads(previous_threads)


class Arguments(Namespace):
    model: Path
    output: Path
    worker: bool


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/analysis/results/torch-autotune-projection.json"),
    )
    parser.add_argument("--worker", action="store_true", help="internal bounded worker")
    args = parser.parse_args(namespace=Arguments())
    code = (
        int(not probe(args.model, args.output))
        if args.worker
        else launch(
            "benchmarks.analysis.torch.autotune", args.model, args.output, timeout=120
        )
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
