"""Prospective confirmation contracts; fake engines never exercise real weights."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st
from transformers.integrations import sdpa_attention

from benchmarks.analysis.torch.prefix.production import study as assay
from wyat.backends.kai.encoding import EncodedRequest
from wyat.backends.kai.torch import KaiScorer
from wyat.benchmark import BenchmarkCycle, BenchmarkEngine, _run
from wyat.core import Request, decision_from_logits
from wyat.evaluation import EvaluationCase, EvaluationRow
from wyat.wire import read_object


def cycle(milliseconds: float) -> BenchmarkCycle:
    return BenchmarkCycle(milliseconds, (), 0, 0)


@given(speedup=st.floats(min_value=1.03, max_value=10, allow_nan=False))
def test_canonical_cohort_metric_and_original_speed_gates(speedup: float) -> None:
    observed = assay.performance([cycle(100)] * 5, [cycle(100 / speedup)] * 5)
    assert observed["paired_median"] == pytest.approx(speedup)
    assert observed["aggregate_throughput_ratio"] == pytest.approx(speedup)
    assert assay.speed_passed(observed)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_nonfinite_or_nonpositive_timing_is_unknown(value: float) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        assay.performance([cycle(1)] * 5, [cycle(value)] * 5)


@pytest.mark.parametrize(
    "name,value",
    [
        ("paired_median", 1.019),
        ("paired_p05", 1.0),
        ("aggregate_throughput_ratio", 1.0),
        ("candidate_cohort_p95_ratio", 1.001),
    ],
)
def test_each_original_speed_gate_is_required(name: str, value: float) -> None:
    observed = assay.performance([cycle(2)] * 5, [cycle(1)] * 5)
    observed[name] = value
    assert not assay.speed_passed(observed)


def test_history_uses_archived_bytes_after_legitimate_live_source_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(assay, "ROOT", tmp_path)
    assert len(assay.admit_history()) == 25
    assert list(tmp_path.iterdir()) == []


def test_current_source_inventory_covers_final_runtime_and_benchmark() -> None:
    paths = assay.source_paths()
    assert all(path.is_file() for path in paths.values())
    assert {
        "wyat/backends/kai/cpu/prefix.py",
        "wyat/backends/kai/torch.py",
        "wyat/server.py",
        "benchmarks/analysis/torch/prefix/production/study.py",
        "benchmarks/analysis/torch/prefix/production/run.py",
        "tests/analysis/test_cpu_prefix_production.py",
    } <= paths.keys()


@pytest.mark.parametrize("name", tuple(assay.HISTORY))
def test_edited_history_cannot_be_admitted(tmp_path: Path, name: str) -> None:
    for historical in assay.HISTORY:
        shutil.copyfile(assay.RESULTS / historical, tmp_path / historical)
    changed = tmp_path / name
    changed.write_bytes(changed.read_bytes() + b" ")
    with pytest.raises(ValueError, match="Pinned"):
        assay.admit_history(tmp_path)


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    prefix = assay.admit_history()
    source = tmp_path / "owned.py"
    source.write_text("# frozen fake-engine owner\n")

    class FakeScorer:
        model_id = assay.MODEL_ID

        def encode_request(self, request: Request) -> EncodedRequest:
            width = len(request.candidates)
            return EncodedRequest(
                prefix + tuple(range(width + 2)),
                tuple(range(25, 25 + width)),
                26 + width,
            )

        def score_many(self, requests: Sequence[Request]) -> list[list[float]]:
            return [[-float(i) for i in range(len(row.candidates))] for row in requests]

    class FakePrefix(FakeScorer):
        prefix_state_bytes = 0

        def __init__(self, base: FakeScorer, tokens: tuple[int, ...]) -> None:
            assert tokens == prefix

        def _admits(self, row: EncodedRequest) -> bool:
            return row.ids[:25] == prefix

    def load(*args: object, **kwargs: object) -> FakeScorer:
        return FakeScorer()

    def reference(
        path: Path,
        cases: tuple[EvaluationCase, ...],
        encoder: object,
        *,
        dataset_sha256: str,
    ) -> BenchmarkCycle:
        rows = []
        for case in cases:
            decision = decision_from_logits(
                case.request,
                [-float(i) for i in range(len(case.request.candidates))],
                assay.MODEL_ID,
            )
            rows.append(
                EvaluationRow(
                    case.id,
                    case.expected,
                    "ok",
                    decision,
                    decision.selected in case.expected,
                    0,
                    None,
                )
            )
        return BenchmarkCycle(0, tuple(rows), 0, 0)

    def run(
        engine: BenchmarkEngine, cases: tuple[EvaluationCase, ...]
    ) -> BenchmarkCycle:
        actual = _run(engine, cases)
        milliseconds = 2.0 if engine.name == "cpu-full-prefill" else 1.5
        return BenchmarkCycle(milliseconds, actual.rows, 0, 0)

    monkeypatch.setattr(KaiScorer, "from_directory", load)
    monkeypatch.setattr(assay, "PrefixScorer", FakePrefix)
    monkeypatch.setattr(assay, "read_reference", reference)
    monkeypatch.setattr(assay, "_run", run)
    monkeypatch.setattr(assay, "source_paths", lambda: {"owner": source})
    monkeypatch.setattr(assay, "MODEL_FILES", {})
    monkeypatch.setattr(sdpa_attention, "_is_torch_mps_available", True)
    monkeypatch.setattr(sdpa_attention, "_is_torch_greater_or_equal_than_2_13", True)
    return tmp_path / "fake-model"


@pytest.mark.parametrize("batch_size", [1, 8])
def test_all14_cohorts_gates_journal_and_thread_restoration(
    tmp_path: Path, fake: Path, batch_size: int
) -> None:
    previous = torch.get_num_threads()
    output = tmp_path / "confirmation.json"
    # The owned launcher creates these siblings before the worker starts.
    output.with_suffix(".wall.json").write_text("{}")
    output.with_suffix(".log").write_text("")
    assert assay.study(fake, output, batch_size)
    receipt = read_object(output)
    rows = cast(list[dict[str, object]], receipt["cycles"])
    assert len(rows) == 14
    assert [row["arm"] for row in rows[4:]] == [
        "cpu-full-prefill",
        "cpu-exact-prefix25",
        "cpu-exact-prefix25",
        "cpu-full-prefill",
        "cpu-full-prefill",
        "cpu-exact-prefix25",
        "cpu-exact-prefix25",
        "cpu-full-prefill",
        "cpu-full-prefill",
        "cpu-exact-prefix25",
    ]
    assert all(
        len(cast(list[object], cast(dict[str, object], row["cycle"])["rows"])) == 77
        for row in rows
    )
    assert rows == [
        json.loads(line)
        for line in output.with_suffix(".cycles.jsonl").read_text().splitlines()
    ]
    assert receipt["accepted"] is receipt["quality_passed"] is True
    assert receipt["screen_passed"] is True
    assert receipt["dispatch_observed"] is False
    assert torch.get_num_threads() == previous


def test_first8_six_cycle_screen_has_no_speed_or_promotion_claim(
    tmp_path: Path, fake: Path
) -> None:
    output = tmp_path / "screen.json"
    assert assay.study(fake, output, 8, screen=True)
    receipt = read_object(output)
    rows = cast(list[dict[str, object]], receipt["cycles"])
    assert len(rows) == 6
    assert all(
        len(cast(list[object], cast(dict[str, object], row["cycle"])["rows"])) == 8
        for row in rows
    )
    assert receipt["screen_passed"] is receipt["quality_passed"] is True
    assert (
        receipt["accepted"]
        is receipt["performance_passed"]
        is receipt["dispatch_observed"]
        is False
    )


def test_source_drift_retains_last_raw_cohort_and_blocks_next(
    tmp_path: Path, fake: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = cast(
        Callable[[BenchmarkEngine, tuple[EvaluationCase, ...]], BenchmarkCycle],
        assay._run,
    )

    def mutate(
        engine: BenchmarkEngine, cases: tuple[EvaluationCase, ...]
    ) -> BenchmarkCycle:
        actual = original(engine, cases)
        (tmp_path / "owned.py").write_text("# unexpected drift\n")
        return actual

    monkeypatch.setattr(assay, "_run", mutate)
    output = tmp_path / "drift.json"
    previous = torch.get_num_threads()
    with pytest.raises(ValueError, match="Pinned"):
        assay.study(fake, output, 8)
    receipt = read_object(output)
    assert receipt["completed"] is receipt["accepted"] is False
    assert len(cast(list[object], receipt["cycles"])) == 1
    assert len(output.with_suffix(".cycles.jsonl").read_text().splitlines()) == 1
    assert torch.get_num_threads() == previous


def test_rejected_speed_is_completed_scientific_negative(
    tmp_path: Path, fake: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = cast(
        Callable[[BenchmarkEngine, tuple[EvaluationCase, ...]], BenchmarkCycle],
        assay._run,
    )

    def slow(
        engine: BenchmarkEngine, cases: tuple[EvaluationCase, ...]
    ) -> BenchmarkCycle:
        actual = original(engine, cases)
        return BenchmarkCycle(
            3 if engine.name == "cpu-exact-prefix25" else 2, actual.rows, 0, 0
        )

    monkeypatch.setattr(assay, "_run", slow)
    output = tmp_path / "negative.json"
    assert assay.study(fake, output, 8)
    receipt = read_object(output)
    assert receipt["completed"] is receipt["quality_passed"] is True
    assert receipt["accepted"] is receipt["performance_passed"] is False
    assert len(cast(list[object], receipt["cycles"])) == 14


@pytest.mark.parametrize("failed_call,expected_cycles", [(1, 2), (4, 14)])
def test_screen_or_warmup_failed_row_cannot_be_hidden(
    tmp_path: Path,
    fake: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_call: int,
    expected_cycles: int,
) -> None:
    original = cast(
        Callable[[BenchmarkEngine, tuple[EvaluationCase, ...]], BenchmarkCycle],
        assay._run,
    )
    calls = 0

    def fail(
        engine: BenchmarkEngine, cases: tuple[EvaluationCase, ...]
    ) -> BenchmarkCycle:
        nonlocal calls
        calls += 1
        actual = original(engine, cases)
        if calls != failed_call:
            return actual
        first = actual.rows[0]
        failed = EvaluationRow(
            first.id, first.expected, "error", None, False, 0, "deliberate failure"
        )
        return BenchmarkCycle(actual.latency_ms, (failed, *actual.rows[1:]), 0, 0)

    monkeypatch.setattr(assay, "_run", fail)
    output = tmp_path / "quality.json"
    assert not assay.study(fake, output, 8)
    receipt = read_object(output)
    rows = cast(list[dict[str, object]], receipt["cycles"])
    assert receipt["completed"] is True
    assert receipt["accepted"] is receipt["quality_passed"] is False
    assert len(rows) == expected_cycles
    assert rows[failed_call - 1]["unknown_case_ids"] == [
        cast(list[str], receipt["case_ids"])[0]
    ]


def test_malformed_cycle_is_retained_before_quality_validation_fails(
    tmp_path: Path, fake: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = cast(
        Callable[[BenchmarkEngine, tuple[EvaluationCase, ...]], BenchmarkCycle],
        assay._run,
    )

    def missing(
        engine: BenchmarkEngine, cases: tuple[EvaluationCase, ...]
    ) -> BenchmarkCycle:
        actual = original(engine, cases)
        return BenchmarkCycle(actual.latency_ms, actual.rows[1:], 0, 0)

    monkeypatch.setattr(assay, "_run", missing)
    output = tmp_path / "missing.json"
    with pytest.raises(ValueError, match="case counts"):
        assay.study(fake, output, 8)
    receipt = read_object(output)
    rows = cast(list[dict[str, object]], receipt["cycles"])
    assert receipt["completed"] is receipt["accepted"] is False
    assert len(rows) == 1 and len(cast(list[object], rows[0]["unknown_case_ids"])) == 1
    assert len(output.with_suffix(".cycles.jsonl").read_text().splitlines()) == 1


@pytest.mark.parametrize("location", ["cohort", "row"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_invalid_raw_timing_is_json_safe_retained_and_rejected(
    tmp_path: Path,
    fake: Path,
    monkeypatch: pytest.MonkeyPatch,
    location: str,
    value: float,
) -> None:
    original = cast(
        Callable[[BenchmarkEngine, tuple[EvaluationCase, ...]], BenchmarkCycle],
        assay._run,
    )

    def invalid(
        engine: BenchmarkEngine, cases: tuple[EvaluationCase, ...]
    ) -> BenchmarkCycle:
        actual = original(engine, cases)
        if location == "cohort":
            return BenchmarkCycle(value, actual.rows, 0, 0)
        first = actual.rows[0]
        row = EvaluationRow(
            first.id,
            first.expected,
            first.status,
            first.prediction,
            first.correct,
            value,
            first.error,
        )
        return BenchmarkCycle(actual.latency_ms, (row, *actual.rows[1:]), 0, 0)

    monkeypatch.setattr(assay, "_run", invalid)
    output = tmp_path / "timing.json"
    with pytest.raises(ValueError, match="timings required"):
        assay.study(fake, output, 8)
    receipt = read_object(output)
    rows = cast(list[dict[str, object]], receipt["cycles"])
    saved = cast(dict[str, object], rows[0]["cycle"])
    assert len(cast(list[object], saved["rows"])) == 77
    if location == "row":
        saved = cast(list[dict[str, object]], saved["rows"])[0]
    assert saved["latency_ms"] is None and saved["invalid_timing"] == repr(value)
    assert receipt["completed"] is receipt["accepted"] is False
    assert receipt["threads_restored"] is True
    assert len(output.with_suffix(".cycles.jsonl").read_text().splitlines()) == 1


def test_final_write_failure_resets_success_and_keeps_all_raw_cycles(
    tmp_path: Path, fake: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = Path.write_text

    def fail(path: Path, text: str) -> int:
        if path.name == "writer.json" and json.loads(text).get("completed") is True:
            raise OSError("forced final save failure")
        return original(path, text)

    monkeypatch.setattr(Path, "write_text", fail)
    output = tmp_path / "writer.json"
    with pytest.raises(OSError, match="final save"):
        assay.study(fake, output, 8)
    receipt = read_object(output)
    assert receipt["completed"] is receipt["accepted"] is False
    assert receipt["threads_restored"] is True
    assert len(cast(list[object], receipt["cycles"])) == 14
    assert len(output.with_suffix(".cycles.jsonl").read_text().splitlines()) == 14


def test_thread_restore_failure_cannot_leave_positive_receipt(
    tmp_path: Path, fake: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = torch.set_num_threads
    previous = torch.get_num_threads()
    calls = 0

    def fail(count: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("forced thread restoration failure")
        original(count)

    monkeypatch.setattr(torch, "set_num_threads", fail)
    output = tmp_path / "restore.json"
    try:
        with pytest.raises(RuntimeError, match="restoration failure"):
            assay.study(fake, output, 8)
        receipt = read_object(output)
        assert receipt["completed"] is receipt["accepted"] is False
        assert receipt["threads_restored"] is False
        assert "restoration failure" in cast(str, receipt["thread_restoration_error"])
        assert len(cast(list[object], receipt["cycles"])) == 14
        assert calls == 2
    finally:
        original(previous)


@pytest.mark.parametrize("suffix", [".json", ".cycles.jsonl", ".sources-123.json"])
def test_worker_preserves_existing_receipt_family(tmp_path: Path, suffix: str) -> None:
    output = tmp_path / "owned.json"
    existing = tmp_path / ("owned" + suffix)
    existing.write_text("preserve")
    with pytest.raises(FileExistsError):
        assay.study(tmp_path / "model", output, 8)
    assert existing.read_text() == "preserve"
