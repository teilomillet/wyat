from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import cast

import pytest

from wyat.core import Candidate, Model, Request
from wyat.evaluation import EvaluationCase, case_from_dict, evaluate


def request(query: str = "question") -> Request:
    return Request(query, (Candidate("a", "First"), Candidate("b", "Second")))


class Scorer:
    def __init__(self) -> None:
        self.calls: list[Request] = []

    def score_many(self, requests: Sequence[Request]) -> Sequence[Sequence[float]]:
        self.calls.extend(requests)
        if any(value.query == "failure" for value in requests):
            raise RuntimeError("fixture failure")
        return [(1.0, 2.0) for _ in requests]


def test_multiple_expected_ids_and_failed_predictions_stay_in_denominator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = iter([0, 10_000_000, 20_000_000, 40_000_000, 50_000_000, 80_000_000])
    monkeypatch.setattr("wyat.evaluation.time.perf_counter_ns", lambda: next(clock))
    scorer = Scorer()
    cases = [
        EvaluationCase("correct", request(), ("a", "b")),
        EvaluationCase("wrong", request(), ("a",)),
        EvaluationCase("failed", request("failure"), ("b",)),
    ]
    report = evaluate(Model(scorer, model_id="fixture"), cases).to_dict()
    assert report["total"] == 3
    assert report["correct"] == 1
    assert report["accuracy"] == pytest.approx(1 / 3)
    assert report["coverage"] == pytest.approx(2 / 3)
    assert report["latency_p50_ms"] == 20
    assert report["latency_p95_ms"] == pytest.approx(29)
    rows = cast(list[dict[str, object]], report["rows"])
    assert len(rows) == 3
    failure = rows[2]
    assert failure["status"] == "error"
    assert failure["prediction"] is None
    assert cast(float, failure["latency_ms"]) >= 0


def test_empty_evaluation_is_rejected_without_inference() -> None:
    scorer = Scorer()
    with pytest.raises(ValueError):
        evaluate(Model(scorer, model_id="fixture"), [])
    assert scorer.calls == []


def test_duplicate_case_ids_fail_before_any_inference() -> None:
    scorer = Scorer()
    cases = [
        EvaluationCase("same", request(), ("a",)),
        EvaluationCase("same", request(), ("b",)),
    ]
    with pytest.raises(ValueError):
        evaluate(Model(scorer, model_id="fixture"), cases)
    assert scorer.calls == []


@pytest.mark.parametrize("identifier", ["\ud800", "case-\udfff"])
def test_invalid_unicode_case_id_is_rejected_at_construction(identifier: str) -> None:
    with pytest.raises(ValueError, match="case ID must contain valid Unicode"):
        EvaluationCase(identifier, request(), ("b",))


def test_unicode_case_and_label_ids_remain_exact_with_failures_in_denominator() -> None:
    candidates = (Candidate("é", "First"), Candidate("e\u0301", "Second"))
    first_id = "é-日本-العربية-🧪"
    second_id = "e\u0301-日本-العربية-🧪"
    cases = [
        EvaluationCase(
            first_id, Request("Choisir / 選択 / اختر", candidates), ("e\u0301",)
        ),
        EvaluationCase(second_id, Request("failure", candidates), ("é",)),
    ]
    report = evaluate(Model(Scorer(), model_id="fixture"), cases).to_dict()
    rows = cast(list[dict[str, object]], report["rows"])
    assert [row["id"] for row in rows] == [first_id, second_id]
    assert rows[0]["expected"] == ["e\u0301"]
    prediction = cast(dict[str, object], rows[0]["prediction"])
    assert prediction["selected"] == "e\u0301"
    assert prediction["ranking"] == ["e\u0301", "é"]
    assert rows[1]["status"] == "error"
    assert report["total"] == 2
    assert report["accuracy"] == report["coverage"] == 0.5


@pytest.mark.parametrize("expected", [(), ("unknown",), ("a", "a")])
def test_invalid_expected_labels_fail_before_any_inference(
    expected: tuple[str, ...],
) -> None:
    scorer = Scorer()
    with pytest.raises((ValueError, TypeError)):
        evaluate(
            Model(scorer, model_id="fixture"),
            [EvaluationCase("case", request(), expected)],
        )
    assert scorer.calls == []


def test_case_parser_preserves_gold_labels_outside_public_request() -> None:
    case = case_from_dict(
        {
            "id": "case",
            "request": {
                "query": "question",
                "candidates": [
                    {"id": "a", "text": "First"},
                    {"id": "b", "text": "Second"},
                ],
            },
            "expected": ["b"],
        }
    )
    assert case == EvaluationCase("case", request(), ("b",))


@pytest.mark.parametrize(
    "value",
    [
        {
            "id": "case",
            "request": {
                "query": "question",
                "candidates": [{"id": "a", "text": "First"}],
            },
            "expected": ["a"],
            "extra": True,
        },
        {
            "id": "case",
            "request": {
                "query": "question",
                "candidates": [{"id": "a", "text": "First"}],
            },
            "expected": "a",
        },
    ],
)
def test_case_parser_rejects_unknown_fields_and_string_labels(value: object) -> None:
    with pytest.raises((ValueError, TypeError)):
        case_from_dict(value)


def test_evaluation_digest_is_stable_across_outputs_and_timings() -> None:
    class OtherScorer:
        def score_many(self, requests: Sequence[Request]) -> Sequence[Sequence[float]]:
            return [(2.0, 1.0) for _ in requests]

    case = EvaluationCase("case", request(), ("b",))
    first = evaluate(Model(Scorer(), model_id="fixture"), [case]).to_dict()
    second = evaluate(Model(OtherScorer(), model_id="fixture"), [case]).to_dict()
    assert first["correct"] != second["correct"]
    assert first["schema_version"] == 1
    assert len(cast(str, first["input_sha256"])) == 64
    assert first["input_sha256"] == second["input_sha256"]


@pytest.mark.parametrize(
    "change",
    ["id", "gold", "query", "context", "task", "candidate_text", "candidate_order"],
)
def test_evaluation_digest_binds_each_public_input_and_gold_field(change: str) -> None:
    original = EvaluationCase("case", request(), ("b",))
    if change == "id":
        changed = replace(original, id="different")
    elif change == "gold":
        changed = replace(original, expected=("a",))
    elif change == "query":
        changed = replace(
            original, request=replace(original.request, query="different")
        )
    elif change == "context":
        changed = replace(
            original, request=replace(original.request, context="different")
        )
    elif change == "task":
        changed = replace(original, request=replace(original.request, task="different"))
    elif change == "candidate_text":
        changed = replace(
            original,
            request=replace(
                original.request,
                candidates=(Candidate("a", "Changed"), Candidate("b", "Second")),
            ),
        )
    else:
        assert change == "candidate_order"
        changed = replace(
            original,
            request=replace(
                original.request,
                candidates=tuple(reversed(original.request.candidates)),
            ),
        )
    model = Model(Scorer(), model_id="fixture")
    assert (
        evaluate(model, [original]).input_sha256
        != evaluate(model, [changed]).input_sha256
    )


def test_gold_labels_never_enter_scorer_request() -> None:
    scorer = Scorer()
    public = request()
    evaluate(
        Model(scorer, model_id="fixture"), [EvaluationCase("case", public, ("b",))]
    )
    assert scorer.calls == [public]
    assert not hasattr(scorer.calls[0], "expected")


def test_report_binds_model_identity_separately_from_input_digest() -> None:
    case = EvaluationCase("case", request(), ("b",))
    first = evaluate(Model(Scorer(), model_id="model:first"), [case]).to_dict()
    second = evaluate(Model(Scorer(), model_id="model:second"), [case]).to_dict()
    assert first["model_id"] == "model:first"
    assert second["model_id"] == "model:second"
    assert first["input_sha256"] == second["input_sha256"]
