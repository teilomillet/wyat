"""First-eight quality and actual dispatch; no ordinary latency verdict."""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import cast

import torch
from torch import nn
from transformers import masking_utils
from transformers.integrations import sdpa_attention
from transformers.models.qwen3 import modeling_qwen3

from benchmarks.analysis.cuda.manifest import (
    DATA,
    DATA_SHA,
    INPUT_SHA,
    MODEL_ID,
    REFERENCE,
    ROOT,
    asset_paths,
    digest,
)
from benchmarks.analysis.cuda.scorer import FP32Scorer, ieee_precision, require_fp32
from benchmarks.analysis.cuda.study import (
    admit,
    admit_control,
    control_cycle,
    occupancy,
)
from benchmarks.analysis.evidence import freeze_sources
from benchmarks.analysis.reference import read_reference
from wyat import Model
from wyat.backends.wyat.artifact import load_artifact
from wyat.backends.wyat.torch import WyatScorer
from wyat.benchmark import (
    BenchmarkCycle,
    BenchmarkEngine,
    QualityGate,
    _run,
    compare_cycles,
)
from wyat.core import Scorer
from wyat.evaluation import _input_digest, case_from_dict
from wyat.wire import decode_json, read_object

from .scorer import efficient_scorer
from .trace import efficient_dispatch_gate, profile_batch

SOURCES = (
    Path("benchmarks/analysis/cuda/attention/__init__.py"),
    Path("benchmarks/analysis/cuda/attention/scorer.py"),
    Path("benchmarks/analysis/cuda/attention/study.py"),
    Path("benchmarks/analysis/cuda/attention/trace.py"),
    Path("benchmarks/analysis/cuda/attention/run.py"),
    Path("tests/analysis/test_cuda_attention.py"),
    Path("docs/analysis/attention.md"),
)


def admit_sources(path: Path, sha256: str) -> dict[str, str]:
    if digest(path) != sha256:
        raise ValueError("New source declaration differs from reviewed identity")
    saved = read_object(path)
    if set(saved) != {str(item) for item in SOURCES}:
        raise ValueError("New source declaration must contain every exact source")
    hashes = cast(dict[str, str], saved)
    if any(digest(ROOT / name) != value for name, value in hashes.items()):
        raise ValueError("New experiment source bytes changed")
    return hashes


def check_cycle(
    cycle: BenchmarkCycle,
    reference: BenchmarkCycle,
    control: BenchmarkCycle,
) -> tuple[bool, dict[str, object]]:
    if tuple(row.id for row in cycle.rows) != tuple(row.id for row in reference.rows):
        raise ValueError("Returned cycle is incomplete or reorders canonical cases")
    cpu_drift = compare_cycles((reference,), (cycle,), QualityGate())
    cuda_drift = compare_cycles((control,), (cycle,), QualityGate())
    return cpu_drift.passed and cuda_drift.passed, {
        "cpu_reference": asdict(cpu_drift),
        "cuda_control": asdict(cuda_drift),
    }


def study(
    model: Path,
    output: Path,
    *,
    manifest: Path,
    manifest_sha256: str,
    sources: Path,
    sources_sha256: str,
    control: Path,
    control_sha256: str,
    batch_size: int,
) -> bool:
    if type(batch_size) is not int or batch_size not in (1, 8):
        raise ValueError("First-eight diagnostic requires batch1/8")
    payload: dict[str, object] = {
        "batch_size": batch_size,
        "completed": False,
        "quality_passed": False,
        "dispatch_passed": False,
        "accepted": False,
        "scope": "first8 pinned BANKING77 training-source; model exposure unknown",
        "gate": asdict(QualityGate()),
        "attention": "FP32 tensor efficient candidate; CUTLASS FastF32, not IEEE",
        "gemm_head": "IEEE global/CUDA matmul; FP32 weights/head, no autocast",
        "limitations": [
            "No speed/promotion claim",
            "Instantaneous occupancy does not reserve hardware",
            "Missing/unstarted outcomes unknown",
        ],
    }

    def write() -> None:
        output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")

    write()
    try:
        if digest(manifest) != manifest_sha256:
            raise ValueError("Original manifest changed")
        admission = admit(manifest)
        source_hashes = admit_sources(sources, sources_sha256)
        pinned = admit_control(
            control, control_sha256, admission, batch_size=batch_size
        )
        hf_policy = {
            "mps_available": sdpa_attention._is_torch_mps_available,
            "xpu_available": sdpa_attention._is_torch_xpu_available,
            "npu_available": sdpa_attention._is_torch_npu_available,
        }
        payload["hf_availability"] = hf_policy
        if any(hf_policy.values()):
            raise ValueError("Screen requires the frozen Linux CUDA HF policy")
        payload.update(
            admission=admission,
            control=pinned,
            source_declaration_sha256=sources_sha256,
        )
        paths = {
            str(path): ROOT / path
            for path in (*asset_paths(), *SOURCES)
            if path.suffix in (".py", ".md")
        }
        paths.update(
            external_qwen3=Path(modeling_qwen3.__file__),
            external_sdpa=Path(sdpa_attention.__file__),
            external_masking=Path(masking_utils.__file__),
        )
        frozen = freeze_sources(paths, output)
        payload.update(source_sha256=frozen.hashes(), source_archive=frozen.archive())
        if load_artifact(model, 512).model_id != MODEL_ID:
            raise ValueError("Model identity differs from frozen control")
        full = tuple(
            case_from_dict(decode_json(line))
            for line in (ROOT / DATA).read_text().splitlines()
        )
        if len(full) != 77 or _input_digest(full) != INPUT_SHA:
            raise ValueError("Canonical full cohort changed")
        cases = full[:8]
        profiled = cases[:batch_size]
        payload.update(
            case_ids=[case.id for case in cases],
            profiled_case_ids=[case.id for case in profiled],
            expected_ordinary_rows=16,
            expected_profile_rows=2 * len(profiled),
        )
        payload["unknown_case_ids"] = {
            f"{name}_{phase}": [case.id for case in selected]
            for name in ("math", "efficient")
            for phase, selected in (("ordinary", cases), ("profile", profiled))
        }
        payload["occupancy_before"] = occupancy()
        torch.set_num_threads(8)
        base = WyatScorer.from_directory(
            model, device="cpu", max_length=512, token_cache_size=0
        )
        reference = read_reference(
            ROOT / REFERENCE, full, base, dataset_sha256=DATA_SHA
        )
        cuda = control_cycle(control, full)
        if not compare_cycles((reference,), (cuda,), QualityGate()).passed:
            raise ValueError("Reviewed CUDA control fails unchanged CPU gates")
        reference, cuda = (
            replace(reference, rows=reference.rows[:8]),
            replace(cuda, rows=cuda.rows[:8]),
        )
        payload["occupancy_before_allocation"] = occupancy()
        with ieee_precision():
            base.device = torch.device("cuda:0")
            cast(nn.Module, base.backbone).to(base.device)
            base.head.to(base.device)
            require_fp32(base)
            payload["weight_dtypes"] = {
                "backbone": dict(
                    Counter(str(p.dtype) for p in base.backbone.parameters())
                ),
                "head": dict(Counter(str(p.dtype) for p in base.head.parameters())),
            }
            engines: tuple[tuple[str, Scorer], ...] = (
                ("math", FP32Scorer(base)),
                ("efficient", efficient_scorer(base)),
            )
            for name, scorer in engines:
                chunks: list[BenchmarkCycle] = []
                payload["phase"] = f"ordinary_{name}"
                for offset in range(0, len(cases), batch_size):
                    occupancy(own_pid=os.getpid())
                    chunk = _run(
                        BenchmarkEngine(
                            name,
                            Model(scorer, model_id=MODEL_ID),
                            batch_size=batch_size,
                        ),
                        cases[offset : offset + batch_size],
                    )
                    chunks.append(chunk)
                    payload[f"{name}_completed_batches"] = [
                        item.to_dict() for item in chunks
                    ]
                    cast(dict[str, list[str]], payload["unknown_case_ids"])[
                        f"{name}_ordinary"
                    ] = [
                        case.id
                        for case in cases
                        if case.id
                        not in {row.id for item in chunks for row in item.rows}
                    ]
                    write()
                    if chunk.failed:
                        raise ValueError(
                            f"{name} ordinary backend/output failure; no profiles"
                        )
                cycle = BenchmarkCycle(
                    sum(item.latency_ms for item in chunks),
                    tuple(row for item in chunks for row in item.rows),
                    chunks[0].peak_rss_before_bytes,
                    chunks[-1].peak_rss_after_bytes,
                )
                payload[f"{name}_cycle"] = cycle.to_dict()
                passed, drift = check_cycle(cycle, reference, cuda)
                payload[f"{name}_drift"] = drift
                write()
                if not passed:
                    raise ValueError(f"{name} numerical/rank drift; profiles unstarted")
            for name, scorer in engines:
                occupancy(own_pid=os.getpid())
                payload["phase"] = f"profile_{name}"
                write()
                cycle, trace = profile_batch(scorer, profiled, name=name)
                payload[f"{name}_profile"] = trace
                payload[f"{name}_profile_cycle"] = cycle.to_dict()
                cast(dict[str, list[str]], payload["unknown_case_ids"])[
                    f"{name}_profile"
                ] = [
                    case.id
                    for case in profiled
                    if case.id not in {row.id for row in cycle.rows}
                ]
                expected = replace(reference, rows=reference.rows[: len(profiled)])
                cuda_expected = replace(cuda, rows=cuda.rows[: len(profiled)])
                passed, drift = check_cycle(cycle, expected, cuda_expected)
                payload[f"{name}_profile_drift"] = drift
                write()
                if not passed:
                    raise ValueError(f"{name} profiled numerical/rank drift")
            length = max(
                len(base.encode_request(case.request).ids) for case in profiled
            )
            candidate_trace = cast(dict[str, object], payload["efficient_profile"])
            dispatch_passed = efficient_dispatch_gate(
                candidate_trace,
                batch_size=len(profiled),
                length=length,
            )
            payload["dispatch_passed"] = dispatch_passed
            payload["cuda_memory"] = {
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
            payload["occupancy_after"] = occupancy(own_pid=os.getpid())
        if (
            digest(manifest) != manifest_sha256
            or admit(manifest) != admission
            or admit_sources(sources, sources_sha256) != source_hashes
            or digest(control) != pinned["sha256"]
            or digest(Path(cast(str, pinned["wall_path"]))) != pinned["wall_sha256"]
            or any(digest(Path(item.path)) != item.sha256 for item in frozen.files)
        ):
            raise ValueError("Consumed identity/source changed during diagnostics")
        payload.update(
            completed=True,
            quality_passed=True,
            phase="completed",
            torch_build=torch.__config__.show(),
        )
        write()
        return dispatch_passed
    except Exception as error:
        payload.update(
            completed=False,
            quality_passed=False,
            dispatch_passed=False,
            accepted=False,
            phase="failed",
            error=f"{type(error).__name__}: {error}",
        )
        write()
        return False
