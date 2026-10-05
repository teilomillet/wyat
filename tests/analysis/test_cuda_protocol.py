"""Offline challenges to CUDA admission and the explicit FP32 control."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from torch import nn

from benchmarks.analysis.cuda import scorer as control
from benchmarks.analysis.cuda import study as runner
from benchmarks.analysis.cuda.manifest import MODEL_ID, digest
from benchmarks.fixtures.kai import create_kai_bundle
from wyat import Candidate, Request
from wyat.backends.kai.torch import KaiScorer
from wyat.benchmark import BenchmarkCycle
from wyat.core import decision_from_logits
from wyat.evaluation import EvaluationCase, EvaluationRow


@pytest.fixture
def base(tmp_path: Path) -> KaiScorer:
    return KaiScorer.from_directory(create_kai_bundle(tmp_path / "kai"), max_length=512)


@settings(
    max_examples=6,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(count=st.integers(2, 5), repeats=st.integers(1, 7))
def test_control_matches_independent_original_and_disables_outer_autocast(
    base: KaiScorer,
    count: int,
    repeats: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = (
        Request("read?", (Candidate("a", "read"), Candidate("b", "write"))),
        Request(
            "write? " * repeats,
            tuple(Candidate(str(i), "read " * (i + 1)) for i in range(count)),
        ),
    )
    expected = KaiScorer.score_many(base, requests)

    def forbidden(requests: object) -> object:
        raise AssertionError("Never call production CUDA-autocast score_many")

    monkeypatch.setattr(base, "score_many", forbidden)
    actual = control.FP32Scorer(base)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        observed = actual.score_many(requests)
    for left, right in zip(observed, expected, strict=True):
        torch.testing.assert_close(
            torch.tensor(left), torch.tensor(right), atol=1e-6, rtol=1e-5
        )
    assert actual.score_many(()) == []


def test_control_rejects_non_fp32_parameters_and_buffers(base: KaiScorer) -> None:
    base.head.key.weight.data = base.head.key.weight.half()
    with pytest.raises(ValueError, match="non-FP32"):
        control.FP32Scorer(base)
    base.head.float()
    base.head.register_buffer("probe", torch.ones(1, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="non-FP32"):
        control.require_fp32(base)


def test_ieee_uses_only_modern_controls_and_restores_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matmul = SimpleNamespace(fp32_precision="tf32")
    backend = SimpleNamespace(
        fp32_precision="none", cuda=SimpleNamespace(matmul=matmul)
    )
    monkeypatch.setattr(torch, "backends", backend)
    with pytest.raises(RuntimeError, match="probe"):
        with control.ieee_precision():
            assert backend.fp32_precision == matmul.fp32_precision == "ieee"
            raise RuntimeError("probe")
    assert backend.fp32_precision == "none" and matmul.fp32_precision == "tf32"
    assert not hasattr(matmul, "allow_tf32")


@pytest.mark.parametrize(
    "gpu,apps",
    [
        ("GPU, UUID, 12282, 128, 1", ""),
        ("GPU, UUID, 12282, 512, 0", ""),
        ("GPU, UUID, 12282, 128, 0", "999, python, 1024"),
    ],
)
def test_occupancy_rejects_busy_without_allocating(
    gpu: str,
    apps: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = iter((gpu, apps))

    def run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 0, stdout=next(outputs), stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(ValueError, match="busy"):
        runner.occupancy()


def test_occupancy_allows_only_current_compute_pid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = iter(("GPU, UUID, 12282, 4096, 90", "42, python, 4000"))

    def run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 0, stdout=next(outputs), stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    assert runner.occupancy(own_pid=42)["exclusive"] is False


def test_control_raw_rows_are_reconstructed_not_trusted(tmp_path: Path) -> None:
    request = Request("read?", (Candidate("a", "read"), Candidate("b", "write")))
    case = EvaluationCase("tiny", request, ("a",))
    decision = decision_from_logits(request, (1.0, 0.0), MODEL_ID)
    cycle = BenchmarkCycle(
        0, (EvaluationRow("tiny", ("a",), "ok", decision, True, 0, None),), 0, 0
    )
    path = tmp_path / "control.json"
    payload = {"cycle": cycle.to_dict()}
    path.write_text(json.dumps(payload))
    path.with_suffix(".wall.json").write_text(
        json.dumps(
            {
                "completed": True,
                "exit_code": 0,
                "timed_out": False,
                "timeout_seconds": 180,
            }
        )
    )
    assert runner.control_cycle(path, (case,)).rows[0].prediction == decision
    rows = cast(list[dict[str, object]], payload["cycle"]["rows"])
    rows[0]["correct"] = False
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="interpretation"):
        runner.control_cycle(path, (case,))
    rows.clear()
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="every"):
        runner.control_cycle(path, (case,))


@pytest.mark.parametrize(
    "field,value",
    [
        ("completed", False),
        ("quality_passed", False),
        ("arm", "released"),
        ("batch_size", 8),
    ],
)
def test_released_admission_requires_reviewed_complete_control(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    path = tmp_path / "control.json"
    payload = {
        "completed": True,
        "quality_passed": True,
        "arm": "fp32",
        "batch_size": 1,
        "admission": {"fixture": True},
    }
    payload[field] = value
    path.write_text(json.dumps(payload))
    path.with_suffix(".wall.json").write_text(
        json.dumps(
            {
                "completed": True,
                "exit_code": 0,
                "timed_out": False,
                "timeout_seconds": 180,
            }
        )
    )
    with pytest.raises(ValueError, match="incomplete"):
        runner.admit_control(path, digest(path), {"fixture": True}, batch_size=1)


@pytest.mark.parametrize(
    "field,value",
    [
        ("completed", False),
        ("exit_code", 1),
        ("timed_out", True),
        ("timeout_seconds", 181),
    ],
)
def test_positive_control_rejected_when_parent_cleanup_unknown(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    path = tmp_path / "positive.json"
    path.write_text(
        json.dumps(
            {
                "completed": True,
                "quality_passed": True,
                "arm": "fp32",
                "batch_size": 1,
                "admission": {},
            }
        )
    )
    wall: dict[str, object] = {
        "completed": True,
        "exit_code": 0,
        "timed_out": False,
        "timeout_seconds": 180,
    }
    wall[field] = value
    path.with_suffix(".wall.json").write_text(json.dumps(wall))
    with pytest.raises(ValueError, match="parent"):
        runner.admit_control(path, digest(path), {}, batch_size=1)


def test_failure_before_gpu_retains_unknown_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def rejected(path: Path) -> dict[str, object]:
        raise ValueError("manifest mismatch")

    monkeypatch.setattr(runner, "admit", rejected)
    manifest = tmp_path / "manifest"
    manifest.write_text("fixture")
    path = tmp_path / "failed.json"
    assert not runner.study(
        tmp_path / "model",
        path,
        manifest=manifest,
        manifest_sha256=digest(manifest),
        arm="fp32",
        batch_size=1,
    )
    saved = json.loads(path.read_text())
    assert saved["completed"] is saved["quality_passed"] is saved["accepted"] is False
    assert "manifest mismatch" in saved["error"] and "cycle" not in saved


def test_parent_import_does_not_import_frameworks() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import benchmarks.analysis.cuda.run; "
            "assert 'torch' not in sys.modules; "
            "assert 'transformers' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


def dtype_events(arm: str, calls: int) -> list[dict[str, object]]:
    return [
        {
            "role": role,
            "output_dtype": "torch.bfloat16"
            if arm == "released" and role != "head"
            else "torch.float32",
            "autocast_enabled": arm == "released",
        }
        for _ in range(calls)
        for role in ("q_proj", "gate_proj", "head")
    ]


@given(calls=st.integers(1, 5), arm=st.sampled_from(("fp32", "released")))
def test_dtype_gate_rejects_missing_wrong_dtype_or_autocast(
    calls: int, arm: runner.Arm
) -> None:
    events = dtype_events(arm, calls)
    weights = {"head": {"torch.float32": 13}, "backbone": {"torch.float32": 13}}
    assert runner.dtype_gate(events, arm, calls, weights)
    assert not runner.dtype_gate(events[:-1], arm, calls, weights)
    events[0]["autocast_enabled"] = not (arm == "released")
    assert not runner.dtype_gate(events, arm, calls, weights)
    events[0]["autocast_enabled"] = arm == "released"
    events[-1]["output_dtype"] = "torch.bfloat16"
    assert not runner.dtype_gate(events, arm, calls, weights)


@pytest.mark.parametrize("failure", ["save", "restore", "source", "partial"])
def test_late_failure_resets_flags_and_preserves_observed_rows(
    base: KaiScorer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    # Fake accelerator transport/counters; only the tiny CPU modules execute.
    from collections.abc import Iterator
    from contextlib import contextmanager, nullcontext

    from benchmarks.analysis.cuda.manifest import DATA, ROOT
    from wyat.evaluation import case_from_dict
    from wyat.wire import decode_json

    cases = tuple(
        case_from_dict(decode_json(line))
        for line in (ROOT / DATA).read_text().splitlines()
    )
    rows = tuple(
        EvaluationRow(
            case.id,
            case.expected,
            "ok",
            decision_from_logits(
                case.request,
                tuple(float(i) for i in range(len(case.request.candidates))),
                MODEL_ID,
            ),
            False,
            0,
            None,
        )
        for case in cases
    )
    cycle = BenchmarkCycle(1, rows, 0, 0)
    admissions = 0

    def admit(path: Path) -> dict[str, object]:
        nonlocal admissions
        admissions += 1
        if failure == "source" and admissions > 1:
            raise ValueError("changed source probe")
        return {}

    monkeypatch.setattr(runner, "admit", admit)
    monkeypatch.setattr(
        runner, "load_artifact", lambda *args: SimpleNamespace(model_id=MODEL_ID)
    )
    monkeypatch.setattr(runner, "occupancy", lambda **kwargs: {})
    monkeypatch.setattr(runner, "read_reference", lambda *args, **kwargs: cycle)
    monkeypatch.setattr(runner, "ieee_precision", nullcontext)
    if failure == "restore":

        @contextmanager
        def restore_failure() -> Iterator[None]:
            yield
            raise RuntimeError("restore probe")

        monkeypatch.setattr(runner, "ieee_precision", restore_failure)
    monkeypatch.setattr(KaiScorer, "from_directory", lambda *args, **kwargs: base)
    monkeypatch.setattr(nn.Module, "to", lambda self, *args, **kwargs: self)
    for name in (
        "reset_peak_memory_stats",
        "memory_allocated",
        "memory_reserved",
        "max_memory_allocated",
        "max_memory_reserved",
    ):
        monkeypatch.setattr(torch.cuda, name, lambda: 0)

    calls = 0

    def fake_run(engine: object, batch: tuple[EvaluationCase, ...]) -> BenchmarkCycle:
        nonlocal calls
        calls += 1
        if failure == "partial" and calls == 3:
            raise RuntimeError("partial probe")
        modules = dict(cast(nn.Module, base.backbone).named_modules())
        for _ in batch:
            modules["layers.0.self_attn.q_proj"](torch.zeros(1, 16))
            modules["layers.0.mlp.gate_proj"](torch.zeros(1, 16))
            base.head(torch.zeros(1, 1, 16), torch.zeros(1, 16))
        ids = {case.id for case in batch}
        return BenchmarkCycle(
            1, tuple(row for row in cycle.rows if row.id in ids), 0, 0
        )

    monkeypatch.setattr(runner, "_run", fake_run)
    original_write = Path.write_text
    failed = False

    def write(path: Path, data: str, *args: object, **kwargs: object) -> int:
        nonlocal failed
        if failure == "save" and not failed and '"completed": true' in data:
            failed = True
            raise OSError("final-save probe")
        return original_write(path, data)

    monkeypatch.setattr(Path, "write_text", write)
    path = tmp_path / "late.json"
    manifest = tmp_path / "manifest"
    manifest.write_text("fixture")
    assert not runner.study(
        tmp_path / "model",
        path,
        manifest=manifest,
        manifest_sha256=digest(manifest),
        arm="fp32",
        batch_size=1,
    )
    payload = json.loads(path.read_text())
    if failure == "partial":
        assert payload["completed_rows"] == 2 and len(payload["unknown_case_ids"]) == 75
        assert "cycle" not in payload
    else:
        assert payload["cycle"]["rows"] == cycle.to_dict()["rows"]
    assert payload["completed"] is payload["quality_passed"] is False
    assert "probe" in payload["error"]
    assert not base.head._forward_hooks
