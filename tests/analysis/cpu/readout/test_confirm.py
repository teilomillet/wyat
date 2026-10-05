"""Prospective scalar confirmation using the existing fake-engine screen fixture."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import torch
from test_screen import fake as fake
from test_screen import historical_checkout as historical_checkout

from benchmarks.analysis.torch.readout import confirm as assay
from benchmarks.analysis.torch.readout import study as screen
from wyat.benchmark import BenchmarkCycle, BenchmarkEngine
from wyat.evaluation import EvaluationCase
from wyat.wire import read_object


@pytest.fixture
def configured(fake: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # Reuse sound screen-fake contracts; no second model/fixture implementation.
    for name in (
        "PrefixScorer",
        "SelectedScorer",
        "_run",
        "read_reference",
        "source_paths",
    ):
        monkeypatch.setattr(assay, name, vars(screen)[name])
    return fake


@pytest.mark.parametrize("selected_ms,accepted", [(1.0, True), (2.1, False)])
def test_all1078_outcomes_pair_order_and_scientific_performance_negative(
    tmp_path: Path,
    configured: Path,
    monkeypatch: pytest.MonkeyPatch,
    selected_ms: float,
    accepted: bool,
) -> None:
    run = assay._run

    def timed(
        engine: BenchmarkEngine, cases: tuple[EvaluationCase, ...]
    ) -> BenchmarkCycle:
        actual = run(engine, cases)
        assert engine.batch_size == 1
        return BenchmarkCycle(
            2.0 if engine.name == "retained" else selected_ms, actual.rows, 0, 0
        )

    monkeypatch.setattr(assay, "_run", timed)
    output = tmp_path / "confirm.json"
    previous = torch.get_num_threads()
    assert assay.study(configured, output)
    saved = read_object(output)
    rows = cast(list[dict[str, object]], saved["cycles"])
    assert [(row["stage"], row["arm"]) for row in rows] == list(assay.SCHEDULE)
    assert [row["arm"] for row in rows[4:]] == [
        "retained",
        "selected",
        "selected",
        "retained",
        "retained",
        "selected",
        "selected",
        "retained",
        "retained",
        "selected",
    ]
    assert (
        sum(
            len(cast(list[object], cast(dict[str, object], row["cycle"])["rows"]))
            for row in rows
        )
        == 1078
    )
    assert saved["completed"] is saved["quality_passed"] is True
    assert saved["performance_passed"] is saved["accepted"] is accepted
    assert saved["dispatch_observed"] is False
    assert saved["pending_cohorts"] == []
    assert saved["threads_restored"] is True and torch.get_num_threads() == previous
    raw = [
        json.loads(line)
        for line in output.with_suffix(".cycles.jsonl").read_text().splitlines()
    ]
    assert [
        {
            key: value
            for key, value in row.items()
            if key not in ("reference_quality", "anchor_quality")
        }
        for row in rows
    ] == raw


def test_first8_admission_uses_archived_inventory_not_new_source_glob(
    monkeypatch: pytest.MonkeyPatch,
    historical_checkout: Path,
) -> None:
    def forbidden() -> dict[str, Path]:
        raise AssertionError("Live expanded inventory is not historical identity")

    monkeypatch.setattr(screen, "source_paths", forbidden)
    assert len(assay.admit_screen()) == 25


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_raw_outcomes_precede_invalid_timing_rejection(
    tmp_path: Path,
    configured: Path,
    monkeypatch: pytest.MonkeyPatch,
    value: float,
) -> None:
    run = assay._run

    def invalid(
        engine: BenchmarkEngine, cases: tuple[EvaluationCase, ...]
    ) -> BenchmarkCycle:
        actual = run(engine, cases)
        return BenchmarkCycle(value, actual.rows, 0, 0)

    monkeypatch.setattr(assay, "_run", invalid)
    output = tmp_path / "clock.json"
    with pytest.raises(ValueError, match="timings"):
        assay.study(configured, output)
    raw = json.loads(output.with_suffix(".cycles.jsonl").read_text())
    assert len(raw["cycle"]["rows"]) == 77
    assert (
        raw["cycle"]["latency_ms"] is None
        if value != -1
        else raw["cycle"]["latency_ms"] == -1
    )
    assert read_object(output)["pending_cohorts"] == list(range(1, 14))


def test_first_quality_failure_stops_and_retains_partial_cohort(
    tmp_path: Path,
    configured: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = assay._run
    count = 0

    def fail(
        engine: BenchmarkEngine, cases: tuple[EvaluationCase, ...]
    ) -> BenchmarkCycle:
        nonlocal count
        count += 1
        actual = run(engine, cases)
        rows = (
            replace(
                actual.rows[0],
                status="failed",
                prediction=None,
                correct=False,
                error="native failed",
            ),
            *actual.rows[1:],
        )
        return BenchmarkCycle(1, rows, 0, 0)

    monkeypatch.setattr(assay, "_run", fail)
    output = tmp_path / "failure.json"
    with pytest.raises(ValueError, match="gate failed"):
        assay.study(configured, output)
    assert count == 1
    assert len(output.with_suffix(".cycles.jsonl").read_text().splitlines()) == 1
    assert read_object(output)["completed"] is False


def test_restore_failure_prevents_success_after_all_rows(
    tmp_path: Path,
    configured: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setter = torch.set_num_threads
    previous = torch.get_num_threads()
    calls = 0

    def fail(count: int) -> None:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("restore failed")
        setter(count)

    monkeypatch.setattr(torch, "set_num_threads", fail)
    output = tmp_path / "restore.json"
    try:
        with pytest.raises(RuntimeError, match="restore failed"):
            assay.study(configured, output)
    finally:
        setter(previous)
    saved = read_object(output)
    assert len(cast(list[object], saved["cycles"])) == 14
    assert (
        saved["completed"]
        is saved["quality_passed"]
        is saved["threads_restored"]
        is False
    )


def test_comparison_can_only_run_after_raw_journal(
    tmp_path: Path,
    configured: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "compare.json"

    def fail(*args: object, **kwargs: object) -> object:
        assert output.with_suffix(".cycles.jsonl").read_text()
        raise RuntimeError("compare failed")

    monkeypatch.setattr(assay, "compare_cycles", fail)
    with pytest.raises(RuntimeError, match="compare failed"):
        assay.study(configured, output)
    assert read_object(output)["completed"] is False


def test_final_save_failure_retains_complete_raw_history_without_success(
    tmp_path: Path,
    configured: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "save.json"
    writer = Path.write_text
    rejected = False

    def write(path: Path, data: str, *args: object, **kwargs: object) -> int:
        nonlocal rejected
        if path == output and json.loads(data)["completed"] and not rejected:
            rejected = True
            raise OSError("final save failed")
        return writer(path, data)

    monkeypatch.setattr(Path, "write_text", write)
    with pytest.raises(OSError, match="final save failed"):
        assay.study(configured, output)
    saved = read_object(output)
    assert saved["completed"] is saved["quality_passed"] is saved["accepted"] is False
    assert saved["threads_restored"] is True
    assert len(output.with_suffix(".cycles.jsonl").read_text().splitlines()) == 14


def test_changed_source_stops_after_retaining_attempted_cohort(
    tmp_path: Path,
    configured: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = assay._run
    count = 0

    def changed(
        engine: BenchmarkEngine, cases: tuple[EvaluationCase, ...]
    ) -> BenchmarkCycle:
        nonlocal count
        count += 1
        actual = run(engine, cases)
        if count == 2:
            (tmp_path / "owner.py").write_text("# changed\n")
        return actual

    monkeypatch.setattr(assay, "_run", changed)
    output = tmp_path / "source.json"
    with pytest.raises(ValueError, match="Pinned"):
        assay.study(configured, output)
    assert count == 2
    assert len(output.with_suffix(".cycles.jsonl").read_text().splitlines()) == 2
    assert read_object(output)["completed"] is False


@pytest.mark.parametrize("suffix", [".json", ".cycles.jsonl", ".sources-old.json"])
def test_confirmation_output_family_is_exclusive(tmp_path: Path, suffix: str) -> None:
    output = tmp_path / "confirm.json"
    existing = output.with_suffix(suffix)
    existing.write_text("original")
    with pytest.raises(FileExistsError):
        assay.study(tmp_path, output)
    assert existing.read_text() == "original"


def test_cli_owns_only_scalar_180_second_worker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import sys

    captured: dict[str, object] = {}

    def launch(module: str, model: Path, output: Path, *, timeout: float) -> int:
        captured.update(module=module, model=model, output=output, timeout=timeout)
        return 0

    monkeypatch.setattr(assay, "launch", launch)
    monkeypatch.setattr(
        sys,
        "argv",
        ["confirm", str(tmp_path), "--output", str(tmp_path / "result.json")],
    )
    with pytest.raises(SystemExit) as exited:
        assay.main()
    assert exited.value.code == 0
    assert captured["module"] == "benchmarks.analysis.torch.readout.confirm"
    assert captured["timeout"] == 180


def test_current_confirmation_source_inventory_is_complete() -> None:
    paths = assay.source_paths()
    assert all(path.is_file() for path in paths.values())
    assert {
        "benchmarks/analysis/torch/readout/confirm.py",
        "tests/analysis/cpu/readout/test_confirm.py",
        "docs/analysis/cpu/readout/confirm.md",
    } <= paths.keys()
