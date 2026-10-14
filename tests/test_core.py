# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import sys
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


def test_request_context_canonicalizes_json_objects_and_arrays() -> None:
    state = {"b": [1, 2.5, None], "a": {"é": True}}
    expected = '{"a":{"é":true},"b":[1,2.5,null]}'
    for request in (
        Request("Question", [Candidate("a", "First")], context=state),
        Request.choice("Question", {"a": "First"}, context=state),
        Request.noul("Question", context=state),
        Request.score("Question", ("low", "high"), context=state),
    ):
        assert request.context == expected
    assert Request.noul("Question", context=["x"]).context == '["x"]'
    assert Request.noul("Question", context='{"b":1, "a":2}').context == (
        '{"b":1, "a":2}'
    )
    for invalid in (None, True, 1.5, {"a": float("nan")}, {1: "key"}, ("t",)):
        with pytest.raises(ValueError):
            Request.noul("Question", context=invalid)  # type: ignore[arg-type]


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


def test_temperature_scaling_sharpens_and_softens_probabilities() -> None:
    req = request(2)
    scores = (2.0, 0.0)
    # T = 1.0
    dec_1 = decision_from_logits(req, scores, "fixture", temperature=1.0)
    assert dec_1.scores == scores
    assert dec_1.ranking == ("0", "1")
    p0_1, p1_1 = dec_1.probabilities
    assert p0_1 > p1_1

    # T = 2.0 (higher entropy / softer)
    dec_2 = decision_from_logits(req, scores, "fixture", temperature=2.0)
    assert dec_2.scores == scores
    assert dec_2.ranking == ("0", "1")
    p0_2, p1_2 = dec_2.probabilities
    assert p0_1 > p0_2 > 0.5

    # T = 0.5 (lower entropy / sharper)
    dec_half = decision_from_logits(req, scores, "fixture", temperature=0.5)
    assert dec_half.scores == scores
    assert dec_half.ranking == ("0", "1")
    p0_half, p1_half = dec_half.probabilities
    assert p0_half > p0_1


@pytest.mark.parametrize(
    "invalid_temp",
    [0.0, -1.0, -0.01, float("nan"), float("inf"), -float("inf"), "1.0", True, False],
)
def test_temperature_validation_rejects_invalid_values(invalid_temp: object) -> None:
    req = request(2)
    scores = (1.0, 2.0)
    error_msg = "temperature must be a positive finite number"
    with pytest.raises(ValueError, match=error_msg):
        decision_from_logits(
            req, scores, "fixture", temperature=cast(float, invalid_temp)
        )

    with pytest.raises(ValueError, match=error_msg):
        Model(FixedScorer(scores), "fixture", temperature=cast(float, invalid_temp))

    model = Model(FixedScorer(scores), "fixture")
    with pytest.raises(ValueError, match=error_msg):
        model.decide(req, temperature=cast(float, invalid_temp))

    with pytest.raises(ValueError, match=error_msg):
        model.decide_many((req,), temperature=cast(float, invalid_temp))


def test_temperature_none_rejected_in_constructor_and_logits() -> None:
    req = request(2)
    scores = (1.0, 2.0)
    error_msg = "temperature must be a positive finite number"
    with pytest.raises(ValueError, match=error_msg):
        decision_from_logits(req, scores, "fixture", temperature=cast(float, None))

    with pytest.raises(ValueError, match=error_msg):
        Model(FixedScorer(scores), "fixture", temperature=cast(float, None))

    model = Model(FixedScorer(scores), "fixture")
    # None is the default sentinel meaning "use model.temperature"
    dec = model.decide(req, temperature=None)
    assert dec.scores == scores
    batch = model.decide_many((req,), temperature=None)
    assert batch[0].scores == scores


def test_model_temperature_default_and_override() -> None:
    scorer = FixedScorer([2.0, 0.0])
    model = Model(scorer, "fixture", temperature=2.0)
    assert model.temperature == 2.0

    # Uses model.temperature = 2.0
    dec = model.decide(request(2))
    p0_t2 = math.exp(1.0) / (math.exp(1.0) + math.exp(0.0))
    assert dec.probabilities[0] == pytest.approx(p0_t2)

    # Per-call override temperature = 1.0
    dec_t1 = model.decide(request(2), temperature=1.0)
    p0_t1 = math.exp(2.0) / (math.exp(2.0) + math.exp(0.0))
    assert dec_t1.probabilities[0] == pytest.approx(p0_t1)

    # decide_many override
    batch_dec = model.decide_many((request(2),), temperature=0.5)
    p0_thalf = math.exp(4.0) / (math.exp(4.0) + math.exp(0.0))
    assert batch_dec[0].probabilities[0] == pytest.approx(p0_thalf)


def test_candidate_from_schema_renders_canonical_sorted_json() -> None:
    schema_a: dict[str, object] = {
        "name": "GetWeather",
        "description": "Fetch weather for Genève",
        "input_parameters": {
            "city": {"type": "str", "required": True},
            "days": {"type": "int", "default": 3, "scale": 1.5, "extra": None},
        },
        "tags": ["weather", "forecast"],
    }
    schema_b: dict[str, object] = {
        "tags": ("weather", "forecast"),
        "input_parameters": {
            "days": {"extra": None, "scale": 1.5, "default": 3, "type": "int"},
            "city": {"required": True, "type": "str"},
        },
        "description": "Fetch weather for Genève",
        "name": "GetWeather",
    }
    cand_a = Candidate.from_schema("GetWeather", schema_a)
    cand_b = Candidate.from_schema("GetWeather", schema_b)
    assert cand_a == cand_b
    assert cand_a.id == "GetWeather"
    assert cand_a.text == (
        '{"description":"Fetch weather for Genève",'
        '"input_parameters":{"city":{"required":true,"type":"str"},'
        '"days":{"default":3,"extra":null,"scale":1.5,"type":"int"}},'
        '"name":"GetWeather","tags":["weather","forecast"]}'
    )


@pytest.mark.parametrize(
    "bad_schema,match",
    [
        ({}, "nonempty mapping"),
        ([], "nonempty mapping"),
        ("not-a-dict", "nonempty mapping"),
        ({"  ": "blank root key"}, "nonempty text"),
        ({1: "non-string key"}, "candidate schema key"),
        ({"key": "\ud800"}, "valid Unicode"),
        ({"key": float("nan")}, "finite JSON values"),
        ({"key": float("inf")}, "finite JSON values"),
        ({"key": b"bytes"}, "finite JSON values"),
        ({"key": {1, 2}}, "finite JSON values"),
    ],
)
def test_candidate_from_schema_rejects_invalid_schemas(
    bad_schema: object, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        Candidate.from_schema("tool", cast(dict[str, object], bad_schema))


def test_candidate_from_schema_rejects_cyclic_references() -> None:
    cyclic_dict: dict[str, object] = {"name": "Loop"}
    cyclic_dict["self"] = cyclic_dict
    with pytest.raises(ValueError, match="cyclic"):
        Candidate.from_schema("tool", cyclic_dict)

    cyclic_list: list[object] = []
    cyclic_list.append(cyclic_list)
    with pytest.raises(ValueError, match="cyclic"):
        Candidate.from_schema("tool", {"items": cyclic_list})


def test_candidate_from_schema_rejects_nesting_beyond_the_stack() -> None:
    nested: list[object] = []
    for _ in range(sys.getrecursionlimit()):
        nested = [nested]
    with pytest.raises(ValueError, match="nested too deeply"):
        Candidate.from_schema("tool", {"items": nested})


def test_decision_abstention_and_uncertainty_properties() -> None:
    req = request(3)
    # Uniform scores -> zero gap/margin, maximum normalized entropy = 1.0
    uniform = decision_from_logits(req, (1.0, 1.0, 1.0), "fixture")
    assert uniform.score_gap == pytest.approx(0.0)
    assert uniform.probability_margin == pytest.approx(0.0)
    assert uniform.entropy == pytest.approx(math.log(3.0))
    assert uniform.normalized_entropy == pytest.approx(1.0)

    # Distinct scores under calibrated temperature
    peaked = decision_from_logits(req, (3.0, 1.0, 0.0), "fixture", temperature=1.084)
    assert peaked.score_gap == pytest.approx(2.0)
    p0, p1, p2 = peaked.probabilities
    assert peaked.probability_margin == pytest.approx(p0 - p1)
    expected_entropy = -(p0 * math.log(p0) + p1 * math.log(p1) + p2 * math.log(p2))
    assert peaked.entropy == pytest.approx(expected_entropy)
    assert peaked.normalized_entropy == pytest.approx(expected_entropy / math.log(3.0))

    # Single-candidate edge case
    single = decision_from_logits(request(1), (2.5,), "fixture")
    assert single.entropy == pytest.approx(0.0)
    for prop in ("score_gap", "probability_margin", "normalized_entropy"):
        with pytest.raises(ValueError, match="at least two candidates"):
            getattr(single, prop)

    # Invalid manual probability distributions rejected
    for bad_probs in ((0.2, 0.2), (-0.1, 1.1), (1.2, -0.2)):
        bad_dec = Decision("fixture", (1.0, 0.0), bad_probs, ("a", "b"))
        for prop in ("probability_margin", "entropy", "normalized_entropy"):
            with pytest.raises(ValueError, match="valid probability distribution"):
                getattr(bad_dec, prop)


def test_request_noul_score_and_choice_factories() -> None:
    choice_req = Request.choice(
        "Route request",
        {"weather": "Get weather", "calendar": "Check calendar"},
        context="User wants forecast",
    )
    assert choice_req.task == "choice"
    assert choice_req.context == "User wants forecast"
    assert choice_req.candidates == (
        Candidate("weather", "Get weather"),
        Candidate("calendar", "Check calendar"),
    )

    noul_default = Request.noul("Are all required parameters present?", context="City?")
    assert noul_default.task == "noul"
    assert noul_default.context == "City?"
    assert noul_default.candidates == (
        Candidate("false", "False."),
        Candidate("true", "True."),
    )
    assert Request("Are all required parameters present?", task="noul") == Request.noul(
        "Are all required parameters present?"
    )

    noul_custom = Request.noul(
        "Abstain?",
        false_text="Proceed with tool call.",
        true_text="Abstain and ask clarifying question.",
    )
    assert noul_custom.candidates == (
        Candidate("false", "Proceed with tool call."),
        Candidate("true", "Abstain and ask clarifying question."),
    )

    score_req = Request.score(
        "Rate tool safety",
        ("Unsafe", "Needs review", "Safe"),
        context="Read-only lookup",
    )
    assert score_req.task == "score"
    assert score_req.context == "Read-only lookup"
    assert score_req.candidates == (
        Candidate("0", "Unsafe"),
        Candidate("1", "Needs review"),
        Candidate("2", "Safe"),
    )

    with pytest.raises(ValueError, match="at least two"):
        Request.score("Rate", ("Only one",))


def test_request_from_dict_supports_noul_default_and_score_rubric() -> None:
    noul_req = request_from_dict(
        {
            "query": "Missing required argument?",
            "task": "noul",
            "context": "Book flight",
        }
    )
    assert noul_req == Request.noul("Missing required argument?", context="Book flight")

    score_req = request_from_dict(
        {
            "query": "Rate relevance",
            "task": "score",
            "candidates": ["Irrelevant", "Partial", "Exact"],
        }
    )
    assert score_req == Request.score(
        "Rate relevance", ("Irrelevant", "Partial", "Exact")
    )

    with pytest.raises(ValueError, match="request requires query and candidates"):
        request_from_dict({"query": "Pick one", "task": "choice"})


def test_decision_probability_true_and_expected_ordinal_value() -> None:
    noul_req = Request.noul("Abstain?")
    noul_dec = decision_from_logits(noul_req, (0.0, 2.0), "fixture")
    expected_p_true = math.exp(2.0) / (math.exp(0.0) + math.exp(2.0))
    assert noul_dec.probability_true == pytest.approx(expected_p_true)

    with pytest.raises(ValueError, match="binary 'false' and 'true'"):
        _ = decision_from_logits(request(2), (0.0, 1.0), "fixture").probability_true

    score_req = Request.score("Quality", ("Low", "Medium", "High"))
    score_dec = decision_from_logits(score_req, (0.0, 1.0, 2.0), "fixture")
    p0, p1, p2 = score_dec.probabilities
    assert score_dec.expected_ordinal_value == pytest.approx(0 * p0 + 1 * p1 + 2 * p2)

    with pytest.raises(ValueError, match="at least two candidates"):
        _ = decision_from_logits(
            Request("Single", (Candidate("0", "Only"),)), (1.0,), "fixture"
        ).expected_ordinal_value

    with pytest.raises(ValueError, match="sequential '0'..'K-1'"):
        _ = noul_dec.expected_ordinal_value


def test_model_predict_and_logits_across_generic_and_native_scorers() -> None:
    import wyat

    record = {
        "state": {"user": "Book a flight tomorrow"},
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Select tool",
                "criteria": {"search": "Search flights", "cancel": "Cancel booking"},
            },
            "doubt": {
                "type": "noul",
                "instructions": "Is destination missing?",
            },
            "confidence": {
                "type": "score",
                "instructions": "Rate request completeness",
                "criteria": ["Incomplete", "Partial", "Complete"],
            },
        },
    }

    class GenericMultiScorer:
        def score_many(self, requests: Sequence[Request]) -> Sequence[Sequence[float]]:
            out: list[tuple[float, ...]] = []
            for req in requests:
                if req.task == "choice":
                    out.append((2.5, 0.5))
                elif req.task == "noul":
                    out.append((-1.0, 1.5))
                else:
                    out.append((1.0, 0.5, -0.5))
            return out

    generic_model = Model(GenericMultiScorer(), "generic-fixture")
    raw_logits = generic_model.logits(record)
    assert len(raw_logits) == 3
    answers = generic_model.predict(record)
    assert set(answers) == {"route", "doubt", "confidence"}
    route_ans = answers["route"]
    assert isinstance(route_ans, wyat.ChoiceAnswer)
    assert route_ans.choice == "search"
    doubt_ans = answers["doubt"]
    assert isinstance(doubt_ans, wyat.NoulAnswer)
    assert doubt_ans.option_ids == ("false", "true")
    assert doubt_ans.probability_true > 0.5
    conf_ans = answers["confidence"]
    assert isinstance(conf_ans, wyat.ScoreAnswer)
    assert 0.0 <= conf_ans.expected_ordinal_value <= 2.0

    # Temperature override on Model.predict scales probabilities
    cooled = generic_model.predict(record, temperature=0.5)
    warmed = generic_model.predict(record, temperature=2.0)
    assert isinstance(cooled["doubt"], wyat.NoulAnswer)
    assert isinstance(warmed["doubt"], wyat.NoulAnswer)
    assert cooled["doubt"].probability_true > doubt_ans.probability_true
    assert warmed["doubt"].probability_true < doubt_ans.probability_true


def test_predict_many_generic_scorer_matches_predict_in_record_order() -> None:
    def record(user: str) -> dict[str, object]:
        return {
            "state": {"user": user},
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Select tool",
                    "criteria": {"search": "Search", "cancel": "Cancel"},
                },
                "doubt": {"type": "noul", "instructions": "Is anything missing?"},
            },
        }

    class ContextScorer:
        def score_many(self, requests: Sequence[Request]) -> Sequence[Sequence[float]]:
            return [
                (float(len(request.context)), float(len(request.query)))
                for request in requests
            ]

    model = Model(ContextScorer(), "generic-fixture")
    records = [record("Book a flight"), record("Cancel"), record("Book a flight")]
    for temperature in (None, 0.5):
        batched = model.predict_many(records, temperature=temperature)
        assert batched == tuple(
            model.predict(item, temperature=temperature) for item in records
        )
    assert batched[0] != batched[1]
    assert model.predict_many([]) == ()
    for invalid in ("records", {"state": "", "questions": {}}, iter(records)):
        with pytest.raises(ValueError, match="ordered sequence"):
            model.predict_many(invalid)  # type: ignore[arg-type]


def test_predict_answers_accessors_narrow_by_question_type() -> None:
    import wyat

    record = {
        "state": "Refund for order 42",
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Select tool",
                "criteria": {"search": "Search", "refund": "Refund"},
            },
            "risk": {"type": "score", "criteria": ["Low.", "High."]},
            "doubt": {"type": "noul", "instructions": "Is anything missing?"},
        },
    }
    model = Model(FixedScorer((1.0, 0.0)), "fixture")
    answers = model.predict(record)
    assert isinstance(answers, wyat.Answers) and isinstance(answers, dict)
    assert answers.choice("route") is answers["route"]
    assert answers.score("risk") is answers["risk"]
    assert answers.noul("doubt") is answers["doubt"]
    assert answers == dict(answers)
    assert all(isinstance(item, wyat.Answers) for item in model.predict_many([record]))
    with pytest.raises(TypeError, match="'risk' is score, not choice"):
        answers.choice("risk")
    with pytest.raises(TypeError, match="'route' is choice, not noul"):
        answers.noul("route")
    with pytest.raises(KeyError):
        answers.score("missing")


def test_decide_many_return_errors_isolates_refused_requests() -> None:
    from wyat.core import ModelInputError, ModelOutputError

    class RefusingScorer:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def score_many(self, requests: Sequence[Request]) -> Sequence[Sequence[float]]:
            self.batch_sizes.append(len(requests))
            if any(request.query == "bad" for request in requests):
                raise ModelInputError("too long", reason="input-length")
            return [(float(len(request.context)), 0.0) for request in requests]

    scorer = RefusingScorer()
    model = Model(scorer, "fixture")
    pair = (Candidate("a", "A"), Candidate("b", "B"))
    requests = [
        Request("bad" if index in (2, 5) else "good", pair, context="x" * index)
        for index in range(7)
    ]
    with pytest.raises(ModelInputError):
        model.decide_many(requests)
    results = model.decide_many(requests, errors="return")
    assert [type(result).__name__ for result in results] == [
        "Decision",
        "Decision",
        "ModelInputError",
        "Decision",
        "Decision",
        "ModelInputError",
        "Decision",
    ]
    refused = results[2]
    assert isinstance(refused, ModelInputError) and refused.reason == "input-length"
    for request, result in zip(requests, results, strict=True):
        if isinstance(result, Decision):
            assert result == model.decide(request)
    assert model.decide_many([], errors="return") == ()
    with pytest.raises(ValueError, match="errors must be"):
        model.decide_many(requests, errors="skip")  # type: ignore[call-overload]

    class BrokenScorer:
        def score_many(self, requests: Sequence[Request]) -> Sequence[Sequence[float]]:
            return [(1.0,) for _ in requests]

    with pytest.raises(ModelOutputError):
        Model(BrokenScorer(), "fixture").decide_many(requests, errors="return")


def test_predict_answers_carry_model_id_and_input_digest() -> None:
    import wyat
    from wyat import tasks

    def record(state: object, instructions: object = "Pick") -> dict[str, object]:
        return {
            "state": state,
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": instructions,
                    "criteria": {"a": "A", "b": "B"},
                },
                "doubt": {"type": "noul"},
            },
        }

    model = Model(FixedScorer((1.0, 0.0)), "fixture:v1")
    first = model.predict(record({"user": "x"}))
    route = first["route"]
    assert isinstance(route, wyat.ChoiceAnswer)
    assert route.model_id == "fixture:v1"
    assert len(route.input_sha256) == 64
    assert route.to_dict()["model_id"] == "fixture:v1"
    assert route.to_dict()["input_sha256"] == route.input_sha256
    assert first["doubt"].input_sha256 != route.input_sha256
    state, (parsed, _) = tasks.public_questions(record({"user": "x"}))
    assert route.input_sha256 == tasks.question_input_sha256(state, parsed)
    same = [
        model.predict(record({"user": "x"}), temperature=0.5)["route"],
        model.predict_many([record({"user": "x"})])[0]["route"],
        model.predict(record('{"user":"x"}'))["route"],
    ]
    assert {answer.input_sha256 for answer in same} == {route.input_sha256}
    assert model.predict(record({"user": "y"}))["route"].input_sha256 != (
        route.input_sha256
    )
    assert (
        model.predict(record({"user": "x"}, {"q": "Pick"}))["route"].input_sha256
        == model.predict(record({"user": "x"}, '{"q":"Pick"}'))["route"].input_sha256
    )
    plain = tasks.interpret(parsed, (1.0, 0.0))
    assert "model_id" not in plain.to_dict() and plain.model_id == ""
    with pytest.raises(ValueError, match="input_sha256"):
        tasks.ChoiceAnswer("q", ("a", "b"), (1.0, 0.0), (0.5, 0.5), "m", "ABC")
