from __future__ import annotations

import math
from collections.abc import Sequence
from typing import cast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from wyat.core import Candidate, Model, Request, request_from_dict


class FixedScorer:
    def __init__(self, scores: Sequence[float]) -> None:
        self.scores = scores
        self.calls: list[tuple[Request, ...]] = []

    def score_many(self, requests: Sequence[Request]) -> Sequence[Sequence[float]]:
        self.calls.append(tuple(requests))
        return [self.scores for _ in requests]


def request(count: int = 3) -> Request:
    return Request(
        "Choose an action",
        tuple(Candidate(str(i), f"Action {i}") for i in range(count)),
    )


def test_preserves_logits_and_stable_input_order_for_ties() -> None:
    decision = Model(FixedScorer([2.0, -1.0, 2.0]), model_id="fixture").decide(
        request()
    )
    assert decision.scores == (2.0, -1.0, 2.0)
    assert decision.ranking == ("0", "2", "1")
    assert decision.selected == "0"
    assert decision.model_id == "fixture"
    assert math.isclose(sum(decision.probabilities), 1.0)
    assert decision.probabilities[0] == decision.probabilities[2]


@settings(max_examples=75, deadline=None)
@given(
    st.lists(
        st.floats(
            min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False
        ),
        min_size=1,
        max_size=15,
    )
)
def test_ranking_and_softmax_against_independent_reference(scores: list[float]) -> None:
    decision = Model(FixedScorer(scores), model_id="fixture").decide(
        request(len(scores))
    )
    ranking = tuple(
        str(i) for i in sorted(range(len(scores)), key=lambda i: -scores[i])
    )
    weights = [math.exp(value - max(scores)) for value in scores]
    expected = tuple(value / sum(weights) for value in weights)
    assert decision.scores == tuple(scores)
    assert decision.ranking == ranking
    assert decision.probabilities == pytest.approx(expected, abs=1e-14)
    assert all(
        math.isfinite(value) and 0 <= value <= 1 for value in decision.probabilities
    )


@pytest.mark.parametrize(
    "scores",
    [
        [0.0],
        [0.0, 1.0, 2.0, 3.0],
        [0.0, float("nan"), 1.0],
        [float("inf"), 0.0, 1.0],
        [-float("inf"), 0.0, 1.0],
    ],
)
def test_rejects_wrong_cardinality_or_nonfinite_logits(scores: list[float]) -> None:
    with pytest.raises((ValueError, TypeError)):
        Model(FixedScorer(scores), model_id="fixture").decide(request())


def test_batch_calls_scorer_once_and_matches_scalar_results() -> None:
    scorer = FixedScorer([1.0, 3.0, 2.0])
    model = Model(scorer, model_id="fixture")
    requests = (request(), request())
    decisions = model.decide_many(requests)
    assert scorer.calls == [requests]
    assert decisions == tuple(model.decide(value) for value in requests)


def test_batch_rejects_backend_row_count_mismatch() -> None:
    class BrokenScorer:
        def score_many(self, requests: Sequence[Request]) -> Sequence[Sequence[float]]:
            return []

    with pytest.raises(ValueError):
        Model(BrokenScorer(), model_id="fixture").decide(request())


def test_empty_batch_has_no_scorer_effect() -> None:
    scorer = FixedScorer([0.0])
    assert Model(scorer, model_id="fixture").decide_many(()) == ()
    assert scorer.calls == []


@pytest.mark.parametrize("score", [True, "1.0", None])
def test_logits_do_not_coerce_boolean_or_text_values(score: object) -> None:
    with pytest.raises(ValueError):
        # Deliberately violate the backend contract to exercise runtime validation.
        invalid_scores = cast(Sequence[float], [score])
        Model(FixedScorer(invalid_scores), model_id="fixture").decide(request(1))


@pytest.mark.parametrize(
    "candidates", [(), (Candidate("a", "First"), Candidate("a", "Second"))]
)
def test_request_rejects_empty_or_duplicate_candidate_ids(
    candidates: tuple[Candidate, ...],
) -> None:
    with pytest.raises(ValueError):
        Request("Question", candidates)


def test_request_from_dict_preserves_candidate_order_and_optional_fields() -> None:
    value = {
        "query": "Question",
        "context": "Context",
        "task": "choice",
        "candidates": [{"id": "b", "text": "Second"}, {"id": "a", "text": "First"}],
    }
    decoded = request_from_dict(value)
    assert decoded == Request(
        "Question",
        (Candidate("b", "Second"), Candidate("a", "First")),
        context="Context",
        task="choice",
    )


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {"query": "Question", "candidates": [{"id": "a", "text": "A"}], "answer": "a"},
        {"query": 12, "candidates": [{"id": "a", "text": "A"}]},
        {"query": "Question", "candidates": [{"id": "a", "text": "A", "score": 1}]},
        {"query": "Question", "candidates": "abc"},
    ],
)
def test_request_parser_rejects_unknown_fields_and_wrong_types(value: object) -> None:
    with pytest.raises((ValueError, TypeError)):
        request_from_dict(value)
