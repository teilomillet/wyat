# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Check the application policies around model decisions, without loading weights."""

import importlib
import json
import runpy
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

import wyat

PACKAGE = "examples.use_cases.benchmark_workflows"
FOLDER = Path(__file__).resolve().parents[2] / "examples/use_cases/benchmark_workflows"
WORKFLOW_NAMES = (
    "group_statement",
    "phishing_triage",
    "review_tags",
    "product_relevance",
    "code_output_review",
    "chess_move",
    "application_conditions",
    "service_handoff",
    "updated_request",
    "incident_priority",
)


class FixedScorer:
    def __init__(self, logits: tuple[float, ...]) -> None:
        self.logits = logits

    def score_many(
        self, requests: Sequence[wyat.Request]
    ) -> tuple[tuple[float, ...], ...]:
        return tuple(self.logits for _ in requests)


def test_rejected_handoff_does_not_establish_another_owner() -> None:
    module = importlib.import_module(f"{PACKAGE}.08_service_handoff")
    model = wyat.Model(FixedScorer((10.0, -10.0)), model_id="fixture")
    result = module.prepare_handoff(model, "A refund", "Ownership unknown", "Rules")
    assert result["probability_network_billing"] < 0.01
    assert result["destination"] == "routing_review"


def test_missing_evidence_requests_information() -> None:
    module = importlib.import_module(f"{PACKAGE}.07_application_conditions")
    model = wyat.Model(FixedScorer((-10.0, 10.0, -10.0)), model_id="fixture")
    result = module.check_application(model, "A prerequisite is unverified", "Rules")
    assert result["next_step"] == "request_missing_evidence"


def test_equal_proposal_scores_do_not_choose_by_input_order() -> None:
    module = importlib.import_module(f"{PACKAGE}.09_updated_request")
    model = wyat.Model(FixedScorer((0.0,) * 5), model_id="fixture")
    for proposals in [{"a": "A", "b": "B"}, {"b": "B", "a": "A"}]:
        result = module.rank_proposals(model, "History", proposals)
        assert result["proposal_id"] is None
        assert result["next_step"] == "review_tie"


def test_ranking_requires_a_comparison() -> None:
    module = importlib.import_module(f"{PACKAGE}.09_updated_request")
    model = wyat.Model(FixedScorer((0.0,) * 5), model_id="fixture")
    with pytest.raises(ValueError, match="at least two"):
        module.rank_proposals(model, "History", {})


def test_failed_calls_stay_in_diagnostic_denominator() -> None:
    module = importlib.import_module(f"{PACKAGE}.evaluate")

    class FailingScorer:
        def score_many(self, requests: Sequence[wyat.Request]) -> tuple[()]:
            raise wyat.ModelInputError("fixture refuses request", reason="input-length")

    report = module.evaluate(wyat.Model(FailingScorer(), model_id="fixture"))
    assert report["passed"] == 0
    assert report["expected_examples"] == 10
    assert len(report["outcomes"]) == 10
    assert all(row["error"] for row in report["outcomes"])


def test_chess_receives_all_legal_moves_without_checkmate_hints() -> None:
    chess = pytest.importorskip("chess")
    module = importlib.import_module(f"{PACKAGE}.06_chess_move")
    captured: list[wyat.Request] = []

    class CaptureScorer:
        def score_many(
            self, requests: Sequence[wyat.Request]
        ) -> tuple[tuple[float, ...], ...]:
            captured.extend(requests)
            return tuple(
                tuple(10.0 if c.id == "h5f7" else -10.0 for c in request.candidates)
                for request in requests
            )

    result = module.run(wyat.Model(CaptureScorer(), model_id="fixture"))
    legal = {move.uci() for move in chess.Board(module.FEN).legal_moves}
    assert {candidate.id for candidate in captured[0].candidates} == legal
    assert all(candidate.text == candidate.id for candidate in captured[0].candidates)
    assert result["checkmate"] is True


@pytest.mark.parametrize("model_id,exit_code", [("fixture", 1), ("demo-fixture", 0)])
def test_diagnostic_cli_keeps_json_and_exposes_failure_status(
    model_id: str,
    exit_code: int,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from examples.use_cases import common

    def load_fixture(description: str | None) -> wyat.Model:
        print("Loader progress")
        return wyat.Model(FixedScorer((0.0,)), model_id=model_id)

    monkeypatch.setattr(common, "load_model", load_fixture)
    monkeypatch.delitem(sys.modules, f"{PACKAGE}.evaluate", raising=False)
    with pytest.raises(SystemExit) as stopped:
        runpy.run_module(f"{PACKAGE}.evaluate", run_name="__main__")
    assert stopped.value.code == exit_code
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["passed"] == 0
    assert report["expected_examples"] == 10
    assert "Loader progress" in captured.err


@pytest.mark.parametrize("name", WORKFLOW_NAMES)
def test_sample_json_and_embedded_fixture_make_identical_requests(name: str) -> None:
    runner = importlib.import_module(f"{PACKAGE}.run")
    if name == "chess_move":
        pytest.importorskip("chess")
    samples = json.loads((FOLDER / "sample-inputs.json").read_text())

    class RecordingScorer:
        def __init__(self) -> None:
            self.requests: list[wyat.Request] = []

        def score_many(
            self, requests: Sequence[wyat.Request]
        ) -> tuple[tuple[float, ...], ...]:
            self.requests.extend(requests)
            return tuple(tuple(0.0 for _ in request.candidates) for request in requests)

    scorer = RecordingScorer()
    model = wyat.Model(scorer, model_id="fixture")
    module_name, _ = runner.WORKFLOWS[name]
    module = importlib.import_module(f"{PACKAGE}.{module_name}")
    embedded = module.run(model)
    original_requests = list(scorer.requests)
    scorer.requests.clear()
    external = runner.run_workflow(model, name, samples[name])
    assert external == embedded
    assert scorer.requests == original_requests


def test_bad_input_is_rejected_before_loading_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = importlib.import_module(f"{PACKAGE}.run")
    path = tmp_path / "bad.json"
    path.write_text('{"application": "case", "expected_answer": "1"}')

    def unexpected_load(*args: object, **kwargs: object) -> wyat.Model:
        raise AssertionError("malformed input must not load or download weights")

    monkeypatch.setattr(wyat, "load", unexpected_load)
    assert runner.main(["application_conditions", "--input", str(path)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "missing=['rules']" in captured.err
    assert "expected_answer" in captured.err


def test_file_runner_accepts_custom_data_and_emits_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runner = importlib.import_module(f"{PACKAGE}.run")
    path = tmp_path / "custom.json"
    path.write_text('{"application": "My own record", "rules": "My own rules"}')
    assert runner.main(["application_conditions", "--input", str(path), "--demo"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["mode"] == "demo_equal_scores"
    assert output["example"] == "application_conditions"
    assert output["result"]["decision"]["model_id"] == "demo"


def test_incorrect_input_types_are_rejected() -> None:
    runner = importlib.import_module(f"{PACKAGE}.run")
    with pytest.raises(ValueError, match="opinions must be"):
        runner.validate_input(
            "group_statement",
            {"opinions": "not a list", "drafts": {"a": "A", "b": "B"}},
        )
    with pytest.raises(ValueError, match="drafts must contain"):
        runner.validate_input(
            "group_statement", {"opinions": ["A view"], "drafts": {"a": "A", "b": 2}}
        )
