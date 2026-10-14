"""Screen ownership and negative evidence; fake cohorts do not establish quality."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from transformers.integrations import sdpa_attention
from transformers.modeling_outputs import BaseModelOutputWithPast

from benchmarks.analysis.torch.prefix.production import study as original
from benchmarks.analysis.torch.readout import confirm, scorer
from benchmarks.analysis.torch.readout import study as assay
from benchmarks.fixtures.wyat import create_wyat_bundle
from wyat import Candidate, Request
from wyat.backends.wyat.cpu.prefix import PrefixScorer
from wyat.backends.wyat.encoding import EncodedRequest
from wyat.backends.wyat.torch import WyatScorer
from wyat.benchmark import BenchmarkCycle, BenchmarkEngine, _run
from wyat.core import decision_from_logits
from wyat.evaluation import EvaluationCase, EvaluationRow
from wyat.wire import read_object


@pytest.fixture
def base(tmp_path: Path) -> WyatScorer:
    return WyatScorer.from_directory(create_wyat_bundle(tmp_path / "wyat"), max_length=512)


@settings(
    max_examples=8,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(batch=st.integers(1, 3), length=st.integers(2, 9), count=st.integers(1, 5))
def test_stock_early_blocks_offset_and_ordered_duplicate_readouts(
    base: WyatScorer,
    batch: int,
    length: int,
    count: int,
) -> None:
    resource = PrefixScorer(base, (1, 2, 3))
    ids = (
        torch.arange(batch * length).reshape(batch, length)
        % base.backbone.config.vocab_size
    )
    positions = torch.tensor([[0] * count + [length - 1]] * batch)
    with torch.inference_mode():
        full = cast(
            BaseModelOutputWithPast,
            base.backbone(
                input_ids=ids, past_key_values=resource._fork(batch), use_cache=True
            ),
        )
        actual = scorer.selected_hidden(base, ids, positions, resource._fork(batch), 3)
    assert full.last_hidden_state is not None
    expected = full.last_hidden_state[torch.arange(batch)[:, None], positions]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_public_ragged_readouts_and_independent_snapshot_owners(
    base: WyatScorer,
) -> None:
    requests = tuple(
        Request(
            "read? " * n,
            tuple(Candidate(str(i), "read " * (i + 1)) for i in range(n)),
            context="Bonjour",
        )
        for n in (2, 4, 3)
    )
    prefix = base.encode_request(requests[0]).ids[:2]
    retained = PrefixScorer(base, prefix)
    selected = scorer.SelectedScorer(base, prefix)
    for left, right in zip(retained._states, selected._states, strict=True):
        for a, b in zip(left, right, strict=True):
            assert a.data_ptr() != b.data_ptr() and torch.equal(a, b)
    expected = retained.score_many(requests)
    actual = selected.score_many(requests)
    for actual_scores, expected_scores in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            torch.tensor(actual_scores),
            torch.tensor(expected_scores),
            atol=1e-6,
            rtol=1e-5,
        )
    assert selected.score_many(()) == []


def test_full_batch_fallback_encodes_once_and_keeps_original_rows(
    base: WyatScorer, monkeypatch: pytest.MonkeyPatch
) -> None:
    resource = scorer.SelectedScorer(base, (1, 2))
    encoded = (
        EncodedRequest((1, 2, 3, 4), (2,), 3),
        EncodedRequest((1, 2, 5, 6), (0,), 3),
    )
    calls: list[EncodedRequest] = []

    def encode(request: Request) -> EncodedRequest:
        result = encoded[int(request.query)]
        calls.append(result)
        return result

    def fallback(rows: Sequence[EncodedRequest]) -> list[list[float]]:
        assert tuple(rows) == encoded
        return [[1.0], [2.0]]

    monkeypatch.setattr(base, "encode_request", encode)
    monkeypatch.setattr(base, "_score_encoded", fallback)
    assert resource.score_many(
        tuple(Request(str(i), (Candidate("a", "a"),)) for i in range(2))
    ) == [[1.0], [2.0]]
    assert tuple(calls) == encoded


def events(selected: bool) -> list[dict[str, object]]:
    full = [
        [8, 16, 167, 128],
        [8, 8, 192, 128],
        [8, 8, 192, 128],
        [],
        [],
        [8, 1, 167, 192],
        [],
    ]
    short = [
        [8, 16, 7, 128],
        [8, 8, 192, 128],
        [8, 8, 192, 128],
        [],
        [],
        [8, 1, 7, 192],
        [],
    ]
    return [
        {
            "operator": "aten::_scaled_dot_product_flash_attention_for_cpu",
            "count": count,
            "input_shapes": shape,
            "self_cpu_time_us": 1.0,
        }
        for shape, count in ([(full, 27), (short, 1)] if selected else [(full, 28)])
    ]


@pytest.mark.parametrize("selected", [False, True])
def test_dispatch_requires_observed_exact_query_kv_mask_and_counts(
    selected: bool,
) -> None:
    observed = events(selected)
    assert assay.dispatch(observed, selected=selected)
    observed[-1]["count"] = 0
    assert not assay.dispatch(observed, selected=selected)
    assert not assay.dispatch([], selected=selected)
    bad = events(selected) + [
        {"operator": "aten::_scaled_dot_product_attention_math", "count": 1}
    ]
    assert not assay.dispatch(bad, selected=selected)


@pytest.fixture
def historical_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Exercise the live-source guard with archived identity, without importing it."""
    archive_name = next(name for name in confirm.SCREEN_HISTORY if ".sources-" in name)
    original.verify_files(
        {
            original.RESULTS / assay.CORE: assay.CORE_SHA,
            original.RESULTS / archive_name: confirm.SCREEN_HISTORY[archive_name],
        }
    )
    core = read_object(original.RESULTS / assay.CORE)
    expected = cast(dict[str, str], core["source_sha256"]) | cast(
        dict[str, str], core["production_source_sha256"]
    )
    assert len(expected) == 25
    archive = read_object(original.RESULTS / archive_name)
    entries: dict[str, dict[str, object]] = {}
    for item in cast(list[dict[str, object]], archive["files"]):
        name = cast(str, item["name"])
        assert name not in entries
        assert (
            hashlib.sha256(cast(str, item["text"]).encode()).hexdigest()
            == item["sha256"]
        )
        entries[name] = item
    checkout = tmp_path / "historical-checkout"
    for name, checksum in expected.items():
        assert entries[name]["sha256"] == checksum
        relative = Path(name)
        assert not relative.is_absolute() and ".." not in relative.parts
    for name in expected:
        path = checkout / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(cast(str, entries[name]["text"]), encoding="utf-8")
    monkeypatch.setattr(assay, "ROOT", checkout)
    return checkout


def test_historical_fixture_still_rejects_changed_runtime_bytes(
    historical_checkout: Path,
) -> None:
    assert len(assay.admit_history()) == 25
    source = historical_checkout / "wyat/core.py"
    source.write_bytes(source.read_bytes() + b"\n# changed identity\n")
    with pytest.raises(ValueError, match="Pinned input, source or historical evidence"):
        assay.admit_history()


@pytest.fixture
def fake(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    historical_checkout: Path,
) -> Path:
    prefix = assay.admit_history()
    owner = tmp_path / "owner.py"
    owner.write_text("# frozen fake engine\n")

    class FakeScorer:
        model_id = original.MODEL_ID

        def encode_request(self, request: Request) -> EncodedRequest:
            return EncodedRequest(
                prefix + tuple(range(167)),
                tuple(range(25, 25 + len(request.candidates))),
                191,
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
            prediction = decision_from_logits(
                case.request,
                [-float(i) for i in range(len(case.request.candidates))],
                original.MODEL_ID,
            )
            rows.append(
                EvaluationRow(
                    case.id,
                    case.expected,
                    "ok",
                    prediction,
                    prediction.selected in case.expected,
                    0,
                    None,
                )
            )
        return BenchmarkCycle(0, tuple(rows), 0, 0)

    def run(
        engine: BenchmarkEngine, cases: tuple[EvaluationCase, ...]
    ) -> BenchmarkCycle:
        actual = _run(engine, cases)
        return BenchmarkCycle(1.0, actual.rows, 0, 0)

    def profile(
        engine: BenchmarkEngine,
        cases: tuple[EvaluationCase, ...],
        retain: Callable[[BenchmarkCycle], None],
    ) -> list[dict[str, object]]:
        retain(assay._run(engine, cases))
        return events(engine.name == "selected")

    monkeypatch.setattr(WyatScorer, "from_directory", load)
    monkeypatch.setattr(assay, "PrefixScorer", FakePrefix)
    monkeypatch.setattr(assay, "SelectedScorer", FakePrefix)
    monkeypatch.setattr(assay, "read_reference", reference)
    monkeypatch.setattr(assay, "_run", run)
    monkeypatch.setattr(assay, "profile_cycle", profile)
    monkeypatch.setattr(assay, "source_paths", lambda: {"owner": owner})
    monkeypatch.setattr(original, "MODEL_FILES", {})
    monkeypatch.setattr(sdpa_attention, "_is_torch_mps_available", True)
    monkeypatch.setattr(sdpa_attention, "_is_torch_greater_or_equal_than_2_13", True)
    return tmp_path / "fake-model"


def test_exact88_schedule_raw_journal_and_no_promotion(
    tmp_path: Path, fake: Path
) -> None:
    output = tmp_path / "screen.json"
    previous = torch.get_num_threads()
    assert assay.study(fake, output)
    saved = read_object(output)
    rows = cast(list[dict[str, object]], saved["cycles"])
    assert [(row["stage"], row["arm"]) for row in rows] == list(assay.SCHEDULE)
    assert (
        sum(
            len(cast(list[object], cast(dict[str, object], row["cycle"])["rows"]))
            for row in rows
        )
        == 88
    )
    raw_rows = [
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
    ] == raw_rows
    assert saved["pending_cohorts"] == []
    assert saved["quality_passed"] is saved["dispatch_passed"] is True
    assert saved["accepted"] is saved["performance_passed"] is False
    assert saved["threads_restored"] is True and torch.get_num_threads() == previous


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_invalid_cohort_timing_retains_outputs_before_rejection(
    tmp_path: Path, fake: Path, monkeypatch: pytest.MonkeyPatch, value: float
) -> None:
    run = assay._run

    def invalid(
        engine: BenchmarkEngine, cases: tuple[EvaluationCase, ...]
    ) -> BenchmarkCycle:
        actual = run(engine, cases)
        return BenchmarkCycle(value, actual.rows, 0, 0)

    monkeypatch.setattr(assay, "_run", invalid)
    output = tmp_path / "invalid.json"
    with pytest.raises(ValueError, match="timings"):
        assay.study(fake, output)
    saved = read_object(output)
    rows = cast(list[dict[str, object]], saved["cycles"])
    raw = cast(dict[str, object], rows[0]["cycle"])
    assert len(cast(list[object], raw["rows"])) == 8
    assert (
        raw["latency_ms"] is None
        if not torch.isfinite(torch.tensor(value))
        else raw["latency_ms"] == value
    )
    assert saved["completed"] is saved["quality_passed"] is False
    assert output.with_suffix(".cycles.jsonl").read_text()


def test_profiler_exit_failure_retains_produced_predictions(
    tmp_path: Path, fake: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(
        engine: BenchmarkEngine,
        cases: tuple[EvaluationCase, ...],
        retain: Callable[[BenchmarkCycle], None],
    ) -> list[dict[str, object]]:
        retain(assay._run(engine, cases))
        raise RuntimeError("profiler cleanup")

    monkeypatch.setattr(assay, "profile_cycle", fail)
    output = tmp_path / "profile.json"
    with pytest.raises(RuntimeError, match="cleanup"):
        assay.study(fake, output)
    saved = read_object(output)
    rows = cast(list[dict[str, object]], saved["cycles"])
    assert (
        len(rows) == 10
        and rows[-1]["capture_error"] == "RuntimeError: profiler cleanup"
    )
    assert saved["pending_cohorts"] == [10] and saved["completed"] is False


def test_first_failed_outcome_stops_calls_after_raw_journal(
    tmp_path: Path, fake: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = assay._run
    count = 0

    def failed(
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
                error="native failure",
            ),
            *actual.rows[1:],
        )
        return BenchmarkCycle(1.0, rows, 0, 0)

    monkeypatch.setattr(assay, "_run", failed)
    output = tmp_path / "failed.json"
    with pytest.raises(ValueError, match="gate failed"):
        assay.study(fake, output)
    saved = read_object(output)
    rows = cast(list[dict[str, object]], saved["cycles"])
    assert count == len(rows) == 1
    assert rows[0]["unknown_case_ids"] == ["banking77-train-05698"]
    assert saved["pending_cohorts"] == list(range(1, 11))
    assert len(output.with_suffix(".cycles.jsonl").read_text().splitlines()) == 1


def test_comparison_failure_cannot_prevent_raw_persistence(
    tmp_path: Path, fake: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "comparison.json"

    def fail(*args: object, **kwargs: object) -> object:
        assert output.with_suffix(".cycles.jsonl").read_text()
        raise RuntimeError("comparison failed")

    monkeypatch.setattr(assay, "compare_cycles", fail)
    with pytest.raises(RuntimeError, match="comparison failed"):
        assay.study(fake, output)
    saved = read_object(output)
    assert len(cast(list[object], saved["cycles"])) == 1
    assert saved["completed"] is False


def test_thread_restore_failure_never_saves_success(
    tmp_path: Path, fake: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setter = torch.set_num_threads
    previous = torch.get_num_threads()
    calls = 0

    def fail_restore(count: int) -> None:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("restore failed")
        setter(count)

    monkeypatch.setattr(torch, "set_num_threads", fail_restore)
    output = tmp_path / "restore.json"
    try:
        with pytest.raises(RuntimeError, match="restore failed"):
            assay.study(fake, output)
    finally:
        setter(previous)
    saved = read_object(output)
    assert len(cast(list[object], saved["cycles"])) == 11
    assert (
        saved["completed"]
        is saved["quality_passed"]
        is saved["threads_restored"]
        is False
    )


def test_final_save_failure_retains_journal_and_resets_success(
    tmp_path: Path,
    fake: Path,
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
        assay.study(fake, output)
    saved = read_object(output)
    assert saved["completed"] is saved["quality_passed"] is False
    assert saved["threads_restored"] is True
    assert len(output.with_suffix(".cycles.jsonl").read_text().splitlines()) == 11


def test_source_change_stops_after_retaining_current_cohort(
    tmp_path: Path,
    fake: Path,
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
            (tmp_path / "owner.py").write_text("# changed during inference\n")
        return actual

    monkeypatch.setattr(assay, "_run", changed)
    output = tmp_path / "source.json"
    with pytest.raises(ValueError, match="Pinned"):
        assay.study(fake, output)
    assert count == 2
    assert len(output.with_suffix(".cycles.jsonl").read_text().splitlines()) == 2
    assert read_object(output)["completed"] is False


def test_invalid_profile_clock_retains_raw_events_before_rejection(
    tmp_path: Path,
    fake: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = assay.profile_cycle

    def invalid(
        engine: BenchmarkEngine,
        cases: tuple[EvaluationCase, ...],
        retain: Callable[[BenchmarkCycle], None],
    ) -> list[dict[str, object]]:
        captured = profile(engine, cases, retain)
        captured[0]["self_cpu_time_us"] = float("nan")
        return captured

    monkeypatch.setattr(assay, "profile_cycle", invalid)
    output = tmp_path / "clock.json"
    with pytest.raises(ValueError, match="clock evidence"):
        assay.study(fake, output)
    saved = read_object(output)
    rows = cast(list[dict[str, object]], saved["cycles"])
    captured = cast(list[dict[str, object]], rows[-1]["operator_self_times"])
    assert len(rows) == 10
    assert captured[0]["self_cpu_time_us"] is None
    assert captured[0]["invalid_timing"] == "nan"
    assert len(output.with_suffix(".cycles.jsonl").read_text().splitlines()) == 10


@pytest.mark.parametrize("suffix", [".json", ".cycles.jsonl", ".sources-old.json"])
def test_existing_output_family_is_preserved(tmp_path: Path, suffix: str) -> None:
    output = tmp_path / "screen.json"
    existing = output.with_suffix(suffix)
    existing.write_text("original bytes")
    with pytest.raises(FileExistsError):
        assay.study(tmp_path, output)
    assert existing.read_text() == "original bytes"


def test_source_inventory_covers_current_runtime_and_experiment() -> None:
    paths = assay.source_paths()
    assert all(path.is_file() for path in paths.values())
    assert {
        "benchmarks/analysis/torch/readout/block.py",
        "benchmarks/analysis/torch/readout/scorer.py",
        "benchmarks/analysis/torch/readout/study.py",
        "wyat/server.py",
        "wyat/backends/wyat/cpu/prefix.py",
        "tests/analysis/cpu/readout/test_screen.py",
    } <= paths.keys()
