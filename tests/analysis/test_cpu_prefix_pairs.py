"""Confirmation history/input admission and late-failure lifecycle; no timing."""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest
import torch
from transformers.integrations import sdpa_attention

from benchmarks.analysis.torch.prefix import pairs
from benchmarks.analysis.torch.prefix.study import ROOT
from wyat.backends.wyat.encoding import EncodedRequest
from wyat.backends.wyat.torch import WyatScorer
from wyat.benchmark import BenchmarkCycle, BenchmarkEngine, _run
from wyat.core import Request, decision_from_logits
from wyat.evaluation import EvaluationCase, EvaluationRow
from wyat.wire import read_object


def copy_history(root: Path) -> None:
    target = root / "benchmarks/analysis/results"
    target.mkdir(parents=True)
    for suffix in pairs.HISTORY:
        name = "torch-prefix8." + suffix
        shutil.copyfile(ROOT / "benchmarks/analysis/results" / name, target / name)
    target = root / "benchmarks/real"
    target.mkdir(parents=True)
    for name in ("banking77-train-dev-menu6.jsonl", "torch-fp32-reference77.json"):
        shutil.copyfile(ROOT / "benchmarks/real" / name, target / name)


def test_exact_independently_audited_history_admitted() -> None:
    history = pairs.admit_history()
    assert history["raw_outputs"] == 64
    assert history["receipt_sha256"] == pairs.HISTORY["json"]
    assert len(cast(list[int], history["prefix_tokens"])) == 25


@pytest.mark.parametrize("suffix", tuple(pairs.HISTORY))
def test_edited_receipt_journal_wall_or_archive_rejected(
    tmp_path: Path, suffix: str
) -> None:
    copy_history(tmp_path)
    path = tmp_path / ("benchmarks/analysis/results/torch-prefix8." + suffix)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="Frozen first-eight history changed"):
        pairs.admit_history(tmp_path)


@pytest.mark.parametrize(
    "name", ["banking77-train-dev-menu6.jsonl", "torch-fp32-reference77.json"]
)
def test_current_data_or_reference_drift_rejected(tmp_path: Path, name: str) -> None:
    copy_history(tmp_path)
    path = tmp_path / "benchmarks/real" / name
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        pairs.admit_history(tmp_path)


@pytest.mark.parametrize("batch_size", [True, False, 0, 2, 16])
def test_invalid_batch_rejected_before_effects(tmp_path: Path, batch_size: int) -> None:
    with pytest.raises(ValueError, match="batch size 1 or 8"):
        pairs.study(
            tmp_path / "model", tmp_path / "receipt.json", batch_size=batch_size
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("batch_size", [1, 8])
def test_failed_model_load_retains_admission_and_restores_threads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, batch_size: int
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise ValueError("deliberate load failure")

    monkeypatch.setattr(WyatScorer, "from_directory", fail)
    monkeypatch.setattr(sdpa_attention, "_is_torch_mps_available", True)
    monkeypatch.setattr(sdpa_attention, "_is_torch_greater_or_equal_than_2_13", True)
    prior = torch.get_num_threads()
    output = tmp_path / "receipt.json"
    with pytest.raises(ValueError, match="deliberate load failure"):
        pairs.study(tmp_path / "model", output, batch_size=batch_size)
    receipt = read_object(output)
    assert (
        receipt["completed"]
        is receipt["accepted"]
        is receipt["quality_passed"]
        is False
    )
    assert receipt["phase"] == "failed" and receipt["first8_admission"] is not None
    assert torch.get_num_threads() == prior


def test_failed_history_blocks_model_load_and_retains_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def invalid_history() -> dict[str, object]:
        raise ValueError("invalid raw history")

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("model load before admission")

    monkeypatch.setattr(pairs, "admit_history", invalid_history)
    monkeypatch.setattr(WyatScorer, "from_directory", forbidden)
    output = tmp_path / "receipt.json"
    with pytest.raises(ValueError, match="invalid raw history"):
        pairs.study(tmp_path / "model", output, batch_size=8)
    receipt = read_object(output)
    assert receipt["completed"] is receipt["accepted"] is False


def test_setup_write_failure_preserves_nonacceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A setup failure must remain visible even when the model was never loaded.
    history = pairs.admit_history()
    monkeypatch.setattr(pairs, "admit_history", lambda: history)
    monkeypatch.setattr(sdpa_attention, "_is_torch_mps_available", True)
    monkeypatch.setattr(sdpa_attention, "_is_torch_greater_or_equal_than_2_13", True)

    def fail(*args: object, **kwargs: object) -> None:
        raise ValueError("artifact failure after setup")

    monkeypatch.setattr(WyatScorer, "from_directory", fail)
    original = Path.write_text

    def write(path: Path, text: str, *args: object, **kwargs: object) -> int:
        if path.name == "receipt.json" and json.loads(text).get("phase") == "loading":
            raise OSError("write failed before load")
        # Exact signature is narrow; these runner writes use only UTF-8 text.
        return original(path, text)

    monkeypatch.setattr(Path, "write_text", write)
    output = tmp_path / "receipt.json"
    with pytest.raises(OSError, match="write failed"):
        pairs.study(tmp_path / "model", output, batch_size=8)
    receipt = read_object(output)
    assert (
        receipt["completed"]
        is receipt["accepted"]
        is receipt["performance_passed"]
        is False
    )


@pytest.fixture
def fake_engines(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pure fake scores exercise orchestration; they make no model-quality claim."""
    history = pairs.admit_history()
    prefix = tuple(cast(list[int], history["prefix_tokens"]))
    identifier = cast(str, history["model_id"])

    class FakeScorer:
        model_id = identifier

        def encode_request(self, request: Request) -> EncodedRequest:
            width = len(request.candidates)
            return EncodedRequest(
                prefix + tuple(range(width + 2)),
                tuple(range(25, 25 + width)),
                26 + width,
            )

        def score_many(self, requests: Sequence[Request]) -> list[list[float]]:
            return [
                [float(-index) for index in range(len(request.candidates))]
                for request in requests
            ]

    class FakePrefix(FakeScorer):
        prefix_state_bytes = 0

        def __init__(self, base: FakeScorer, tokens: tuple[int, ...]) -> None:
            assert tokens == prefix

        def admits(self, row: EncodedRequest) -> bool:
            return row.ids[:25] == prefix

    def reference(
        path: Path,
        cases: tuple[EvaluationCase, ...],
        encoder: object,
        *,
        dataset_sha256: str,
    ) -> BenchmarkCycle:
        rows = tuple(
            EvaluationRow(
                case.id,
                case.expected,
                "ok",
                decision,
                decision.selected in case.expected,
                0,
                None,
            )
            for case in cases
            for decision in [
                decision_from_logits(
                    case.request,
                    tuple(
                        float(-index) for index in range(len(case.request.candidates))
                    ),
                    identifier,
                )
            ]
        )
        return BenchmarkCycle(0, rows, 0, 0)

    def load(*args: object, **kwargs: object) -> FakeScorer:
        return FakeScorer()

    monkeypatch.setattr(WyatScorer, "from_directory", load)
    monkeypatch.setattr(pairs, "PrefixScorer", FakePrefix)
    monkeypatch.setattr(pairs, "read_reference", reference)
    monkeypatch.setattr(sdpa_attention, "_is_torch_mps_available", True)
    monkeypatch.setattr(sdpa_attention, "_is_torch_greater_or_equal_than_2_13", True)


def test_completed_cycles_retain_reference_gates_and_alternating_schedule(
    tmp_path: Path, fake_engines: None
) -> None:
    output = tmp_path / "receipt.json"
    assert pairs.study(tmp_path / "model", output, batch_size=8)
    receipt = read_object(output)
    rows = cast(list[dict[str, object]], receipt["cycles"])
    journal = [
        json.loads(line)
        for line in output.with_suffix(".cycles.jsonl").read_text().splitlines()
    ]
    assert rows == journal and len(rows) == 14
    assert all(
        cast(dict[str, object], row["reference_quality"])["passed"] is True
        for row in rows
    )
    order = [row["arm"] for row in rows[4:]]
    assert order == [
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
    assert receipt["completed"] is receipt["quality_passed"] is True


def test_final_write_failure_resets_success_and_preserves_raw_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_engines: None
) -> None:
    original = Path.write_text

    def write(path: Path, text: str) -> int:
        if path.name == "receipt.json" and json.loads(text).get("completed") is True:
            raise OSError("deliberate final write failure")
        return original(path, text)

    monkeypatch.setattr(Path, "write_text", write)
    output = tmp_path / "receipt.json"
    with pytest.raises(OSError, match="final write failure"):
        pairs.study(tmp_path / "model", output, batch_size=1)
    receipt = read_object(output)
    assert (
        receipt["completed"]
        is receipt["accepted"]
        is receipt["quality_passed"]
        is receipt["performance_passed"]
        is False
    )
    assert len(cast(list[object], receipt["cycles"])) == 14
    assert len(output.with_suffix(".cycles.jsonl").read_text().splitlines()) == 14


def test_warmup_failure_is_not_hidden_by_passing_measured_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_engines: None
) -> None:
    original = _run
    calls = 0

    def run(
        engine: BenchmarkEngine, cases: tuple[EvaluationCase, ...]
    ) -> BenchmarkCycle:
        nonlocal calls
        calls += 1
        cycle = original(engine, cases)
        if calls == 4:  # Candidate warmup after both complete screens.
            first = cycle.rows[0]
            failed = EvaluationRow(
                first.id,
                first.expected,
                "error",
                None,
                False,
                0,
                "deliberate warmup failure",
            )
            return BenchmarkCycle(cycle.latency_ms, (failed, *cycle.rows[1:]), 0, 0)
        return cycle

    monkeypatch.setattr(pairs, "_run", run)
    output = tmp_path / "receipt.json"
    assert not pairs.study(tmp_path / "model", output, batch_size=8)
    receipt = read_object(output)
    assert receipt["completed"] is True and receipt["accepted"] is False
    assert receipt["quality_passed"] is False
    assert len(cast(list[object], receipt["cycles"])) == 14
