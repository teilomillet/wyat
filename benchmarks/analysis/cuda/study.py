"""One frozen full77 quality cohort; no CUDA speed or production verdict."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from collections import Counter
from dataclasses import asdict
from functools import partial
from importlib.metadata import version
from pathlib import Path
from typing import Literal, cast

import torch
from torch import nn
from transformers.integrations import sdpa_attention
from transformers.models.qwen3 import modeling_qwen3

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
from wyat.core import Scorer, decision_from_logits
from wyat.evaluation import EvaluationCase, EvaluationRow, _input_digest, case_from_dict
from wyat.wire import decode_json, read_object

from .manifest import (
    DATA,
    DATA_SHA,
    INPUT_SHA,
    MODEL_ID,
    PACKAGES,
    REFERENCE,
    REFERENCE_SHA,
    ROOT,
    RUNTIME,
    asset_paths,
    digest,
)
from .scorer import FP32Scorer, ieee_precision, require_fp32

type Arm = Literal["fp32", "released"]


def dtype_gate(
    events: list[dict[str, object]],
    arm: Arm,
    calls: int,
    weights: dict[str, dict[str, int]],
) -> bool:
    expected_roles = ["q_proj", "gate_proj", "head"] * calls
    if [event["role"] for event in events] != expected_roles:
        return False
    if set(weights["head"]) != {"torch.float32"}:
        return False
    if arm == "fp32" and set(weights["backbone"]) != {"torch.float32"}:
        return False
    return all(
        event["output_dtype"]
        == (
            "torch.float32"
            if arm == "fp32" or event["role"] == "head"
            else "torch.bfloat16"
        )
        and event["autocast_enabled"] is (arm == "released")
        for event in events
    )


def occupancy(*, own_pid: int | None = None) -> dict[str, object]:
    """Instantaneous observation, not a reservation; reject other compute owners."""
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    ).stdout.strip()
    apps = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    ).stdout.strip()
    rows = [row.strip() for row in gpu.splitlines()]
    if len(rows) != 1:
        raise ValueError("This experiment requires exactly one visible GPU")
    name, uuid, total, used, utilization = rows[0].split(", ")
    foreign = [row for row in apps.splitlines() if int(row.split(",")[0]) != own_pid]
    if foreign or (own_pid is None and (int(utilization) != 0 or int(used) > 256)):
        raise ValueError(f"GPU busy; no authorized interference: {gpu}; {apps}")
    return {
        "gpu": gpu,
        "compute_apps": apps,
        "uuid": uuid,
        "name": name,
        "memory_total_MiB": int(total),
        "memory_used_MiB": int(used),
        "utilization_percent": int(utilization),
        "exclusive": False,
    }


def admit(manifest: Path, root: Path = ROOT) -> dict[str, object]:
    declared = read_object(manifest)
    expected = {
        "model_id": MODEL_ID,
        "dataset_sha256": DATA_SHA,
        "reference_sha256": REFERENCE_SHA,
        "input_sha256": INPUT_SHA,
    }
    if any(declared.get(key) != value for key, value in expected.items()):
        raise ValueError("Manifest canonical identities changed")
    files = cast(list[dict[str, object]], declared["files"])
    if [item["path"] for item in files] != [str(p) for p in asset_paths(root)]:
        raise ValueError("Manifest must contain every declared asset exactly once")
    for item in files:
        path = root / cast(str, item["path"])
        if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            raise ValueError(f"Transferred asset changed: {path}")
    actual = {name: version(name) for name in PACKAGES}
    runtime = {
        "python": platform.python_version(),
        "torch_runtime": str(torch.__version__),
        "cuda_build": torch.version.cuda,
    }
    if (
        actual != PACKAGES
        or runtime != RUNTIME
        or actual != declared["versions"]
        or runtime != declared["runtime"]
    ):
        raise ValueError(f"Runtime differs from prospective declaration: {actual}")
    if digest(root / DATA) != DATA_SHA or digest(root / REFERENCE) != REFERENCE_SHA:
        raise ValueError("Canonical data/reference bytes changed")
    return {
        **expected,
        "manifest_sha256": digest(manifest),
        "asset_count": len(files),
        "versions": actual,
        "runtime": runtime,
    }


def admit_control(
    path: Path | None,
    sha256: str | None,
    admission: dict[str, object],
    *,
    batch_size: int,
) -> dict[str, object]:
    if path is None or sha256 is None or digest(path) != sha256:
        raise ValueError("A reviewed immutable FP32 B1 receipt is required")
    control = read_object(path)
    wall_path = path.with_suffix(".wall.json")
    wall = read_object(wall_path)
    if (
        wall.get("exit_code") != 0
        or wall.get("completed") is not True
        or wall.get("timed_out") is not False
        or wall.get("timeout_seconds") != 180
    ):
        raise ValueError("FP32 control parent did not complete safe terminal cleanup")
    if (
        control.get("completed") is not True
        or control.get("quality_passed") is not True
        or control.get("arm") != "fp32"
        or control.get("batch_size") != 1
        or control.get("admission") != admission
    ):
        raise ValueError("FP32 B1 control incomplete or canonical identity differs")
    if batch_size == 8:
        memory = cast(dict[str, int], control["cuda_memory"])
        total = cast(dict[str, int], control["occupancy_before"])[
            "memory_total_MiB"
        ] * (1 << 20)
        if memory["peak_reserved_bytes"] >= total * 0.6:
            raise ValueError("B1 memory leaves insufficient prospective B8 margin")
    return {
        "path": str(path),
        "sha256": sha256,
        "wall_path": str(wall_path),
        "wall_sha256": digest(wall_path),
    }


def control_cycle(path: Path, cases: tuple[EvaluationCase, ...]) -> BenchmarkCycle:
    """Reconstruct all rows from scores and reject incomplete or inconsistent claims."""
    saved = cast(dict[str, object], read_object(path)["cycle"])
    rows = cast(list[dict[str, object]], saved["rows"])
    if len(rows) != len(cases):
        raise ValueError("CUDA control must contain every canonical row")
    reconstructed: list[EvaluationRow] = []
    for row, case in zip(rows, cases, strict=True):
        prediction = cast(dict[str, object], row["prediction"])
        if row["id"] != case.id or row["status"] != "ok" or row["error"] is not None:
            raise ValueError("CUDA control contains missing/failed or reordered rows")
        decision = decision_from_logits(
            case.request, cast(list[float], prediction["scores"]), MODEL_ID
        )
        correct = decision.selected in case.expected
        if (
            prediction != decision.to_dict()
            or row["correct"] != correct
            or row["expected"] != list(case.expected)
        ):
            raise ValueError("CUDA control interpretation differs from recorded scores")
        reconstructed.append(
            EvaluationRow(case.id, case.expected, "ok", decision, correct, 0, None)
        )
    return BenchmarkCycle(0, tuple(reconstructed), 0, 0)


def study(
    model: Path,
    output: Path,
    *,
    manifest: Path,
    manifest_sha256: str,
    arm: Arm,
    batch_size: int,
    control: Path | None = None,
    control_sha256: str | None = None,
) -> bool:
    if (
        arm not in ("fp32", "released")
        or type(batch_size) is not int
        or batch_size not in (1, 8)
    ):
        raise ValueError("Use fp32/released and batch1/8")
    payload: dict[str, object] = {
        "arm": arm,
        "batch_size": batch_size,
        "completed": False,
        "quality_passed": False,
        "accepted": False,
        "scope": "full77 BANKING77 training-source quality; exposure unknown",
        "gate": asdict(QualityGate()),
        "precision": "modern IEEE global/CUDA matmul",
        "attention": "math only" if arm == "fp32" else "maintained SDPA selection",
        "limitations": [
            "No warm/repeated timing claim; no promotion",
            "Occupancy snapshots do not reserve shared hardware",
            "Raw completed batches retained; interrupted/unstarted rows unknown",
        ],
    }

    def write() -> None:
        output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")

    write()
    handles: list[torch.utils.hooks.RemovableHandle] = []
    try:
        if digest(manifest) != manifest_sha256:
            raise ValueError("Manifest differs from root-reviewed immutable identity")
        admission = admit(manifest)
        payload["admission"] = admission
        if arm == "released" or batch_size == 8:
            payload["control"] = admit_control(
                control, control_sha256, admission, batch_size=batch_size
            )
        sources = {
            str(path): ROOT / path for path in asset_paths() if path.suffix == ".py"
        }
        sources["external_qwen3"] = Path(modeling_qwen3.__file__)
        sources["external_sdpa"] = Path(sdpa_attention.__file__)
        frozen = freeze_sources(sources, output)
        payload.update(source_sha256=frozen.hashes(), source_archive=frozen.archive())
        artifact = load_artifact(model, 512)
        if artifact.model_id != MODEL_ID:
            raise ValueError("Loaded artifact differs from frozen model")
        cases = tuple(
            case_from_dict(decode_json(line))
            for line in (ROOT / DATA).read_text().splitlines()
        )
        if len(cases) != 77 or _input_digest(cases) != INPUT_SHA:
            raise ValueError("Full canonical cohort changed")
        payload["occupancy_before"] = occupancy()
        torch.set_num_threads(8)
        base = WyatScorer.from_directory(
            model, device="cpu", max_length=512, token_cache_size=0
        )
        reference = read_reference(
            ROOT / REFERENCE, cases, base, dataset_sha256=DATA_SHA
        )
        cuda_reference = None if control is None else control_cycle(control, cases)
        if (
            cuda_reference is not None
            and not compare_cycles(
                (reference,), (cuda_reference,), QualityGate()
            ).passed
        ):
            raise ValueError("Admitted CUDA control no longer passes strict CPU gates")
        payload["occupancy_before_allocation"] = occupancy()
        with ieee_precision():
            scorer: Scorer
            if arm == "fp32":
                require_fp32(base)
                base.device = torch.device("cuda:0")
                cast(nn.Module, base.backbone).to(base.device)
                base.head.to(base.device)
                require_fp32(base)
                scorer = FP32Scorer(base)
            else:
                del base
                base = WyatScorer.from_directory(
                    model, device="cuda:0", max_length=512, token_cache_size=0
                )
                scorer = base
            dtype_events: list[dict[str, object]] = []

            def record(
                role: str, module: nn.Module, inputs: tuple[object, ...], result: object
            ) -> None:
                if isinstance(result, torch.Tensor):
                    dtype_events.append(
                        {
                            "module": type(module).__name__,
                            "role": role,
                            "input_dtypes": [
                                str(value.dtype)
                                for value in inputs
                                if isinstance(value, torch.Tensor)
                            ],
                            "output_dtype": str(result.dtype),
                            "autocast_enabled": torch.is_autocast_enabled("cuda"),
                        }
                    )

            modules = dict(cast(nn.Module, base.backbone).named_modules())
            for role, module in (
                ("q_proj", modules["layers.0.self_attn.q_proj"]),
                ("gate_proj", modules["layers.0.mlp.gate_proj"]),
                ("head", base.head),
            ):
                handles.append(module.register_forward_hook(partial(record, role)))
            payload["weight_dtypes"] = {
                "backbone": dict(
                    Counter(str(p.dtype) for p in base.backbone.parameters())
                ),
                "head": dict(Counter(str(p.dtype) for p in base.head.parameters())),
            }
            torch.cuda.reset_peak_memory_stats()
            payload["phase"] = "quality_cohort"
            payload["expected_rows"] = 77
            payload["dtype_events"] = dtype_events
            write()
            engine = BenchmarkEngine(
                arm, Model(scorer, model_id=MODEL_ID), batch_size=batch_size
            )
            chunks: list[BenchmarkCycle] = []
            for offset in range(0, 77, batch_size):
                occupancy(own_pid=os.getpid())
                chunks.append(_run(engine, cases[offset : offset + batch_size]))
                payload["completed_batches"] = [chunk.to_dict() for chunk in chunks]
                payload["completed_rows"] = sum(len(chunk.rows) for chunk in chunks)
                payload["unknown_case_ids"] = [
                    case.id for case in cases[offset + batch_size :]
                ]
                write()
            cycle = BenchmarkCycle(
                sum(chunk.latency_ms for chunk in chunks),
                tuple(row for chunk in chunks for row in chunk.rows),
                chunks[0].peak_rss_before_bytes,
                chunks[-1].peak_rss_after_bytes,
            )
            payload["cycle"] = cycle.to_dict()
            payload["dtype_events"] = dtype_events
            payload["dtype_passed"] = dtype_gate(
                dtype_events,
                arm,
                (77 + batch_size - 1) // batch_size,
                cast(dict[str, dict[str, int]], payload["weight_dtypes"]),
            )
            payload["cuda_memory"] = {
                "allocated_bytes": torch.cuda.memory_allocated(),
                "reserved_bytes": torch.cuda.memory_reserved(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
            drift = compare_cycles((reference,), (cycle,), QualityGate())
            payload["drift"] = asdict(drift)
            cuda_drift = (
                None
                if cuda_reference is None
                else compare_cycles((cuda_reference,), (cycle,), QualityGate())
            )
            payload["cuda_control_drift"] = (
                None if cuda_drift is None else asdict(cuda_drift)
            )
            payload["occupancy_after"] = occupancy(own_pid=os.getpid())
            if (
                digest(manifest) != manifest_sha256
                or admit(manifest) != admission
                or any(
                    digest(Path(source.path)) != source.sha256
                    for source in frozen.files
                )
            ):
                raise ValueError("Consumed source/assets changed during cohort")
            if control is not None:
                pinned = cast(dict[str, str], payload["control"])
                if (
                    digest(control) != pinned["sha256"]
                    or digest(Path(pinned["wall_path"])) != pinned["wall_sha256"]
                ):
                    raise ValueError("Consumed CUDA control/wall changed during cohort")
            while handles:
                handles.pop().remove()
        # Successful state is published only after hook and IEEE-state cleanup.
        payload.update(
            completed=True,
            quality_passed=drift.passed
            and len(cycle.rows) == 77
            and payload["dtype_passed"] is True
            and (cuda_drift is None or cuda_drift.passed),
            phase="completed",
            torch_build=torch.__config__.show(),
        )
        write()
        return cast(bool, payload["quality_passed"])
    except Exception as error:
        payload.update(
            completed=False,
            quality_passed=False,
            accepted=False,
            error=f"{type(error).__name__}: {error}",
            phase="failed",
        )
        write()
        return False
    finally:
        for handle in handles:
            try:
                handle.remove()
            except Exception as error:
                payload.update(
                    completed=False,
                    quality_passed=False,
                    accepted=False,
                    cleanup_error=f"{type(error).__name__}: {error}",
                )
                write()
