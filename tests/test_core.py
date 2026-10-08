from __future__ import annotations

import math
from collections import UserList
from collections.abc import Callable, Iterator, Sequence
from dataclasses import FrozenInstanceError, asdict, astuple, fields, replace
from inspect import signature
from typing import cast, get_type_hints

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from wyat import Candidate, Decision, Model, ModelOutputError, Request
from wyat.core import decision_from_logits, request_from_dict


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
    inputs = request(len(scores))
    decision = Model(FixedScorer(scores), model_id="fixture").decide(inputs)
    ranking = tuple(
        str(i) for i in sorted(range(len(scores)), key=lambda i: -scores[i])
    )
    weights = [math.exp(value - max(scores)) for value in scores]
    expected = tuple(value / sum(weights) for value in weights)
    assert decision.scores == tuple(scores)
    assert decision.ranking == ranking
    assert decision.probabilities == pytest.approx(expected, abs=1e-14)
    ids = tuple(candidate.id for candidate in inputs.candidates)
    assert decision.candidate_ids == ids
    assert decision.scores_by_id == dict(zip(ids, scores, strict=True))
    assert decision.probabilities_by_id == pytest.approx(
        dict(zip(ids, expected, strict=True)), abs=1e-14
    )
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
    with pytest.raises(ModelOutputError):
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

    with pytest.raises(ModelOutputError):
        Model(BrokenScorer(), model_id="fixture").decide(request())


@pytest.mark.parametrize("output", [None, [None], [0.0]])
def test_malformed_scorer_output_raises_public_error(output: object) -> None:
    class BrokenScorer:
        def score_many(self, requests: Sequence[Request]) -> Sequence[Sequence[float]]:
            return cast(Sequence[Sequence[float]], output)

    with pytest.raises(ModelOutputError) as caught:
        Model(BrokenScorer(), model_id="fixture").decide(request())
    assert isinstance(caught.value.__cause__, TypeError)


@pytest.mark.parametrize("as_row", [False, True])
def test_len_only_scorer_output_raises_public_error(as_row: bool) -> None:
    class LenOnly:
        def __len__(self) -> int:
            return 1

    class BrokenScorer:
        def score_many(self, requests: Sequence[Request]) -> Sequence[Sequence[float]]:
            output = [LenOnly()] if as_row else LenOnly()
            return cast(Sequence[Sequence[float]], output)

    with pytest.raises(ModelOutputError) as caught:
        Model(BrokenScorer(), model_id="fixture").decide(request(1))
    assert isinstance(caught.value.__cause__, TypeError)


@pytest.mark.parametrize("as_row", [False, True])
def test_output_iteration_execution_errors_propagate_unchanged(as_row: bool) -> None:
    error = TypeError("iteration failed")

    class FailingOutput:
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> float:
            raise error

    class BrokenScorer:
        def score_many(self, requests: Sequence[Request]) -> Sequence[Sequence[float]]:
            output = [FailingOutput()] if as_row else FailingOutput()
            return cast(Sequence[Sequence[float]], output)

    with pytest.raises(TypeError) as caught:
        Model(BrokenScorer(), model_id="fixture").decide(request(1))
    assert caught.value is error


def test_overflowing_logit_conversion_raises_public_error() -> None:
    with pytest.raises(ModelOutputError) as caught:
        Model(FixedScorer([10**1000]), model_id="fixture").decide(request(1))
    assert isinstance(caught.value.__cause__, OverflowError)


def test_accepts_indexable_outputs_without_sequence_inheritance() -> None:
    class Row:
        def __len__(self) -> int:
            return 3

        def __getitem__(self, index: int) -> float:
            return (1.0, 3.0, 2.0)[index]

    class Batch:
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> Row:
            return (Row(),)[index]

    class IndexableScorer:
        def score_many(self, requests: Sequence[Request]) -> Sequence[Sequence[float]]:
            return cast(Sequence[Sequence[float]], Batch())

    decision = Model(IndexableScorer(), model_id="fixture").decide(request())
    assert decision.scores == (1.0, 3.0, 2.0)
    assert decision.ranking == ("1", "2", "0")


@pytest.mark.parametrize(
    "error",
    [
        TypeError("backend failed"),
        ValueError("backend failed"),
        OverflowError("backend failed"),
        RuntimeError("backend failed"),
    ],
)
def test_scorer_execution_errors_propagate_unchanged(error: Exception) -> None:
    class FailingScorer:
        def score_many(self, requests: Sequence[Request]) -> Sequence[Sequence[float]]:
            raise error

    with pytest.raises(type(error)) as caught:
        Model(FailingScorer(), model_id="fixture").decide(request())
    assert caught.value is error


def test_empty_batch_has_no_scorer_effect() -> None:
    scorer = FixedScorer([0.0])
    assert Model(scorer, model_id="fixture").decide_many(()) == ()
    assert scorer.calls == []


@pytest.mark.parametrize("score", [True, "1.0", None])
def test_logits_do_not_coerce_boolean_or_text_values(score: object) -> None:
    with pytest.raises(ModelOutputError):
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


@pytest.mark.parametrize("container", [list, tuple, UserList])
def test_request_snapshots_ordered_sequences(
    container: Callable[[Sequence[Candidate]], Sequence[Candidate]],
) -> None:
    candidates = [Candidate("é", "Premier"), Candidate("查询", "Deuxième")]
    inputs = container(candidates)
    actual = Request("Question", inputs, "Context", "choice")
    expected = Request(
        query="Question", candidates=tuple(candidates), context="Context", task="choice"
    )
    stored: tuple[Candidate, ...] = actual.candidates
    assert isinstance(stored, tuple)
    assert stored == tuple(candidates)
    assert actual == expected
    assert hash(actual) == hash(expected)
    scorer = Model(FixedScorer([1.0, 1.0]), model_id="fixture")
    assert scorer.decide(actual) == scorer.decide(expected)
    if isinstance(inputs, (list, UserList)):
        inputs.reverse()
        inputs[0] = Candidate("replacement", "Different")
        inputs.clear()
    assert actual == expected
    assert hash(actual) == hash(expected)
    # Mypy's dataclass replace typing follows the stored tuple field type.
    assert replace(actual, candidates=[Candidate("new", "New")]).candidates == (  # type: ignore[arg-type]
        Candidate("new", "New"),
    )


def test_request_retains_dataclass_and_constructor_contracts() -> None:
    actual = Request("Question", [Candidate("a", "First")])
    assert actual.context == ""
    assert actual.task == "choice"
    assert tuple(field.name for field in fields(Request)) == (
        "query",
        "candidates",
        "context",
        "task",
    )
    assert Request.__match_args__ == ("query", "candidates", "context", "task")
    assert tuple(signature(Request).parameters) == Request.__match_args__
    assert get_type_hints(Request)["candidates"] == tuple[Candidate, ...]
    assert get_type_hints(Request.__init__)["candidates"] == Sequence[Candidate]
    assert asdict(actual) == {
        "query": "Question",
        "candidates": ({"id": "a", "text": "First"},),
        "context": "",
        "task": "choice",
    }
    assert not hasattr(actual, "__dict__")
    with pytest.raises(FrozenInstanceError):
        actual.query = "Changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "candidates",
    [
        set(),
        {Candidate("a", "First")},
        {},
        {"a": Candidate("a", "First")},
        "abc",
        b"abc",
        bytearray(b"abc"),
        iter([Candidate("a", "First")]),
        [],
        (),
        ["a"],
        [None],
        [Candidate("a", "First"), Candidate("a", "Second")],
    ],
)
def test_request_rejects_unordered_nonsequence_or_invalid_candidates(
    candidates: object,
) -> None:
    with pytest.raises(ValueError):
        Request("Question", cast(Sequence[Candidate], candidates))


@pytest.mark.parametrize("field", ["query", "context", "task"])
@pytest.mark.parametrize("value", [12, "\ud800"])
def test_sequence_request_retains_text_validation(field: str, value: object) -> None:
    kwargs = {"query": "Question", "context": "Context", "task": "choice"}
    kwargs[field] = cast(str, value)
    with pytest.raises(ValueError):
        Request(candidates=[Candidate("a", "First")], **kwargs)


@pytest.mark.parametrize("field", ["query", "task"])
def test_sequence_request_retains_nonempty_instruction_and_task(field: str) -> None:
    kwargs = {"query": "Question", "task": "choice"}
    kwargs[field] = " \n "
    with pytest.raises(ValueError, match="nonempty"):
        Request(candidates=[Candidate("a", "First")], **kwargs)


@pytest.mark.parametrize(
    "scores",
    [
        [1.0, 3.0, 2.0, -1.0],
        [1.0, 1.0, 1.0, 1.0],
        [2.0, -1.0, 2.0, 3.0],
        [0.0, -0.0, 1.0, -0.0],
        [-1000.0, 1000.0, -1001.0, 999.0],
    ],
)
def test_decision_identity_and_maps_follow_original_unicode_candidate_order(
    scores: list[float],
) -> None:
    inputs = Request(
        "Question",
        [
            Candidate(identifier, identifier)
            for identifier in ("é", "e\u0301", "查询", "z")
        ],
    )
    decision = Model(FixedScorer(scores), model_id="fixture").decide(inputs)
    ids = tuple(candidate.id for candidate in inputs.candidates)
    assert decision.candidate_ids == ids
    assert tuple(decision.scores_by_id) == ids
    assert tuple(decision.probabilities_by_id) == ids
    assert decision.scores_by_id == dict(zip(ids, scores, strict=True))
    assert decision.probabilities_by_id == dict(
        zip(ids, decision.probabilities, strict=True)
    )
    changed_scores = decision.scores_by_id
    changed_probabilities = decision.probabilities_by_id
    changed_scores[ids[0]] = 999.0
    changed_probabilities.clear()
    assert decision.scores_by_id[ids[0]] == scores[0]
    assert len(decision.probabilities_by_id) == len(ids)


def test_legacy_decision_and_wire_reconstruction_retain_dataclass_contracts() -> None:
    legacy = Decision("fixture", (1.0, 3.0, 2.0), (0.1, 0.6, 0.3), ("a", "z", "b"))
    payload = legacy.to_dict()
    assert payload == {
        "model_id": "fixture",
        "selected": "a",
        "ranking": ["a", "z", "b"],
        "scores": [1.0, 3.0, 2.0],
        "probabilities": [0.1, 0.6, 0.3],
    }
    reconstructed = Decision(
        cast(str, payload["model_id"]),
        tuple(cast(list[float], payload["scores"])),
        tuple(cast(list[float], payload["probabilities"])),
        tuple(cast(list[str], payload["ranking"])),
    )
    for decision in (legacy, reconstructed):
        assert decision.candidate_ids == ("b", "a", "z")
        assert decision.scores_by_id == {"b": 1.0, "a": 3.0, "z": 2.0}
        assert decision.probabilities_by_id == {"b": 0.1, "a": 0.6, "z": 0.3}
        assert decision.to_dict() == payload
        assert asdict(decision) == {
            "model_id": "fixture",
            "scores": (1.0, 3.0, 2.0),
            "probabilities": (0.1, 0.6, 0.3),
            "ranking": ("a", "z", "b"),
        }
        assert astuple(decision) == (
            "fixture",
            (1.0, 3.0, 2.0),
            (0.1, 0.6, 0.3),
            ("a", "z", "b"),
        )
        assert repr(decision) == (
            "Decision(model_id='fixture', scores=(1.0, 3.0, 2.0), "
            "probabilities=(0.1, 0.6, 0.3), ranking=('a', 'z', 'b'))"
        )
        assert not hasattr(decision, "__dict__")
    assert reconstructed == legacy
    assert hash(reconstructed) == hash(legacy)
    names = ("model_id", "scores", "probabilities", "ranking")
    assert tuple(field.name for field in fields(Decision)) == names
    assert Decision.__match_args__ == names
    assert tuple(signature(Decision).parameters) == names
    for name in ("candidate_ids", "scores_by_id", "probabilities_by_id"):
        with pytest.raises((AttributeError, TypeError)):
            setattr(legacy, name, ())


@pytest.mark.parametrize(
    "scores,ranking",
    [
        ((), ()),
        ((1.0,), ()),
        ((1.0,), ("a", "b")),
        ((1.0, 2.0), ("a", "a")),
        ((1.0,), ("",)),
        ((1.0,), ("\ud800",)),
        ((1.0,), (None,)),
        ((1.0,), (["a"],)),
        ((None,), ("a",)),
        ((True,), ("a",)),
        (("1",), ("a",)),
        ((math.nan,), ("a",)),
        ((math.inf,), ("a",)),
        ((10**1000,), ("a",)),
        (None, ("a",)),
        ("1", ("a",)),
        ((1.0,), "a"),
    ],
)
@pytest.mark.parametrize(
    "accessor", ["candidate_ids", "scores_by_id", "probabilities_by_id"]
)
def test_new_decision_accessors_reject_unusable_scores_or_ranking(
    scores: object,
    ranking: object,
    accessor: str,
) -> None:
    decision = Decision(
        "fixture",
        cast(tuple[float, ...], scores),
        (1.0,),
        cast(tuple[str, ...], ranking),
    )
    with pytest.raises(ValueError, match="decision"):
        getattr(decision, accessor)


@pytest.mark.parametrize(
    "probabilities",
    [
        (),
        (0.5, 0.5),
        (None,),
        (True,),
        ("1",),
        (math.nan,),
        (math.inf,),
        (10**1000,),
        None,
        "1",
    ],
)
def test_probability_accessor_validates_only_when_requested(
    probabilities: object,
) -> None:
    decision = Decision(
        "fixture", (1.0,), cast(tuple[float, ...], probabilities), ("a",)
    )
    assert decision.candidate_ids == ("a",)
    assert decision.scores_by_id == {"a": 1.0}
    with pytest.raises(ValueError, match="decision"):
        _ = decision.probabilities_by_id


def test_invalid_manual_decision_retains_existing_construction_and_wire_behavior() -> (
    None
):
    decision = Decision("fixture", (1.0, 2.0), (0.4,), ("a", "a"))
    assert decision.selected == "a"
    assert decision.to_dict() == {
        "model_id": "fixture",
        "selected": "a",
        "ranking": ["a", "a"],
        "scores": [1.0, 2.0],
        "probabilities": [0.4],
    }
    with pytest.raises(ValueError, match="unique"):
        _ = decision.scores_by_id


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


@pytest.mark.parametrize("count", [0, 1, 3])
def test_logits_actual_cardinality_must_match_reported_length(count: int) -> None:
    class Row:
        def __len__(self) -> int:
            return 2

        def __iter__(self) -> Iterator[float]:
            return iter([0.0] * count)

    with pytest.raises(ModelOutputError, match="candidate count"):
        decision_from_logits(request(2), cast(Sequence[float], Row()), "fixture")


@pytest.mark.parametrize("count", [0, 2])
def test_batch_actual_cardinality_must_match_reported_length(count: int) -> None:
    class Batch:
        def __len__(self) -> int:
            return 1

        def __iter__(self) -> Iterator[Sequence[float]]:
            return iter([(0.0,)] * count)

    class Backend:
        def score_many(self, requests: Sequence[Request]) -> Sequence[Sequence[float]]:
            return cast(Sequence[Sequence[float]], Batch())

    with pytest.raises(ModelOutputError, match="request count"):
        Model(Backend(), "fixture").decide(request(1))
