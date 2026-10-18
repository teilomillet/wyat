# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Label-free recipe fixtures and independent analytic task expectations."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from wyat import ModelOutputError, Request
from wyat.tasks import (
    RECIPE,
    ChoiceAnswer,
    EncodedQuestion,
    NoulAnswer,
    Question,
    ScoreAnswer,
    encode_record,
    interpret,
    public_questions,
)


class CharacterTokenizer:
    def __init__(self) -> None:
        self.segments: list[str] = []

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        self.segments.append(text)
        return list(map(ord, text))


def record(kind: str = "choice", criteria: object = None) -> dict[str, Any]:
    if criteria is None and kind == "choice":
        criteria = {"café": "Wait.", "返金": "Refund."}
    body: dict[str, Any] = {"type": kind, "instructions": "Decide."}
    if criteria is not None:
        body["criteria"] = criteria
    return {"state": "  evidence\n", "questions": {"予約": body}}


def question(kind: str, criteria: object = None) -> Question:
    return public_questions(record(kind, criteria))[1][0]


def test_exact_segmented_fixture_and_readout_endpoints() -> None:
    tokenizer = CharacterTokenizer()
    rows = encode_record(tokenizer, record())
    expected = [
        "State:\n  evidence\n\n\nTask type: choice\nQuestion ID: 予約\n"
        "Instructions:\nDecide.\nOptions:",
        '\n<option>\n{"description":"Wait.","key":"café"}\n</option>',
        '\n<option>\n{"description":"Refund.","key":"返金"}\n</option>',
        "\n\nSelect the allowed option best supported by the state "
        "and instructions.\nDecision:",
    ]
    assert tokenizer.segments == expected
    assert len(rows) == 1
    assert rows[0].ids == tuple(map(ord, "".join(expected)))
    assert rows[0].candidate_positions == (
        len(expected[0]) + len(expected[1]) - 1,
        sum(map(len, expected[:3])) - 1,
    )
    assert rows[0].query_position == sum(map(len, expected)) - 1


def test_canonical_json_state_unicode_ids_defaults_and_snapshots() -> None:
    source = record(criteria={"é": "Composed.", "e\u0301": "Decomposed."})
    source["state"] = {"z": ["日本語", True, None], "a": "café استرداد"}
    del source["questions"]["予約"]["instructions"]
    state, questions = public_questions(source)
    assert state == '{"a":"café استرداد","z":["日本語",true,null]}'
    assert questions[0].instructions == "予約"
    assert questions[0].options == (("é", "Composed."), ("e\u0301", "Decomposed."))
    source["questions"]["予約"]["criteria"]["é"] = "Changed."
    assert questions[0].options[0] == ("é", "Composed.")
    with pytest.raises(FrozenInstanceError):
        questions[0].instructions = "Changed."  # type: ignore[misc]


def test_questions_are_independent_rows_in_saved_order() -> None:
    source = record()
    source["questions"]["rubric"] = {"type": "score", "criteria": ["Low.", "High."]}
    source["questions"]["binary"] = {"type": "noul"}
    rows = encode_record(CharacterTokenizer(), source)
    assert tuple(row.question.id for row in rows) == ("予約", "rubric", "binary")
    for row in rows:
        text = "".join(map(chr, row.ids))
        assert f"Question ID: {row.question.id}\n" in text
        for other in rows:
            if other is not row:
                assert f"Question ID: {other.question.id}\n" not in text
    modified = copy.deepcopy(source)
    modified["questions"]["binary"]["instructions"] = "Other question changes."
    assert encode_record(CharacterTokenizer(), modified)[0] == rows[0]


def test_choice_ties_preserve_order_and_detached_result_maps() -> None:
    value = question("choice", {"café": "Wait.", "返金": "Refund.", "استرداد": "Act."})
    values = [2.0, 2.0, 0.0]
    answer = interpret(value, values)
    assert isinstance(answer, ChoiceAnswer)
    assert answer.type == "choice"
    assert answer.question_id == "予約"
    assert answer.choice == "café"
    assert answer.ranking == ("café", "返金", "استرداد")
    assert answer.option_ids == ("café", "返金", "استرداد")
    assert answer.probabilities == pytest.approx(
        (
            1 / (2 + math.exp(-2)),
            1 / (2 + math.exp(-2)),
            math.exp(-2) / (2 + math.exp(-2)),
        )
    )
    values[0] = -9.0
    assert answer.logits == (2.0, 2.0, 0.0)
    mapping = answer.logits_by_id
    mapping["café"] = -100.0
    serialized = answer.to_dict()
    assert serialized == {
        "type": "choice",
        "choice": "café",
        "probabilities": answer.probabilities_by_id,
    }
    assert answer.logits_by_id["café"] == 2.0
    serialized["probabilities"]["café"] = 0.0  # type: ignore[index]
    assert answer.probabilities_by_id["café"] > 0
    with pytest.raises(FrozenInstanceError):
        answer.logits = (9.0, 0.0, 0.0)  # type: ignore[misc]


def test_score_is_expected_ordinal_index_not_argmax_or_logit() -> None:
    answer = interpret(
        question("score", ["Low.", "Medium.", "High."]), [0, math.log(3), math.log(2)]
    )
    assert isinstance(answer, ScoreAnswer)
    assert answer.option_ids == ("0", "1", "2")
    assert answer.probabilities == pytest.approx((1 / 6, 1 / 2, 1 / 3))
    assert answer.expected_ordinal_value == pytest.approx(7 / 6)
    assert answer.to_dict() == {
        "type": "score",
        "expected_ordinal_value": answer.expected_ordinal_value,
        "probabilities": answer.probabilities_by_id,
    }


@pytest.mark.parametrize("size", [2, 5, 255])
def test_score_complete_rubric_boundaries(size: int) -> None:
    value = question("score", [f"Level {index}." for index in range(size)])
    answer = interpret(value, [0.0] * size)
    assert isinstance(answer, ScoreAnswer)
    assert answer.expected_ordinal_value == pytest.approx((size - 1) / 2)
    assert answer.option_ids == tuple(map(str, range(size)))


@pytest.mark.parametrize(
    "criteria",
    [
        None,
        {"false": "Absent.", "true": "Present."},
        {"true": "Present.", "false": "Absent."},
    ],
)
def test_noul_true_probability_uses_saved_id_not_fixed_position(
    criteria: object,
) -> None:
    value = question("noul", criteria)
    logits = [math.log(3) if key == "true" else 0 for key, _ in value.options]
    answer = interpret(value, logits)
    assert isinstance(answer, NoulAnswer)
    assert answer.probability_true == pytest.approx(3 / 4)
    assert answer.to_dict() == {
        "type": "noul",
        "probability_true": answer.probability_true,
        "probabilities": answer.probabilities_by_id,
    }
    assert interpret(value, [0, 0]).probabilities_by_id["true"] == 0.5
    if criteria is None:
        assert value.options == (("false", "False."), ("true", "True."))


def test_finite_extreme_logits_are_stable() -> None:
    answer = interpret(question("choice"), [-1e38, 1e38])
    assert isinstance(answer, ChoiceAnswer)
    assert answer.choice == "返金"
    assert answer.probabilities == (0, 1)


def test_explicit_null_noul_criteria_use_exact_default_descriptions() -> None:
    source = record("noul")
    source["questions"]["予約"]["criteria"] = None
    assert public_questions(source)[1][0].options == (
        ("false", "False."),
        ("true", "True."),
    )


def test_fp32_rounding_tie_selects_first_option_before_softmax() -> None:
    answer = interpret(question("choice"), [1.0, 1.0 + 1e-9])
    assert isinstance(answer, ChoiceAnswer)
    assert answer.logits == (1.0, 1.0)
    assert answer.choice == "café"
    assert answer.ranking == ("café", "返金")
    assert answer.probabilities == (0.5, 0.5)


@pytest.mark.parametrize("values", [[1e39, 0], [-1e39, 0], [1e308, 0]])
def test_finite_double_logits_that_overflow_fp32_fail(values: list[float]) -> None:
    with pytest.raises(ModelOutputError, match="FP32"):
        interpret(question("choice"), values)


@pytest.mark.parametrize(
    "logits",
    [
        [],
        [0],
        [0, 1, 2],
        [True, 0],
        [None, 0],
        [[0], 0],
        [math.nan, 0],
        [math.inf, 0],
        [float("-inf"), 0],
        "01",
        {"a": 0, "b": 1},
        iter([0, 1]),
        [10**1000, 0],
    ],
)
def test_malformed_outputs_fail_without_repair(logits: Any) -> None:
    with pytest.raises(ModelOutputError):
        interpret(question("choice"), logits)


@pytest.mark.parametrize(
    "state",
    [
        float("nan"),
        float("inf"),
        {"nested": [float("nan")]},
        {3: "Nontext key."},
        (1, 2),
        "\ud800",
        {"\ud800": "key"},
        ["\ud800"],
    ],
)
def test_non_json_or_invalid_unicode_state_fails_before_tokenization(
    state: object,
) -> None:
    source = record()
    source["state"] = state
    tokenizer = CharacterTokenizer()
    with pytest.raises(ValueError):
        encode_record(tokenizer, source)
    assert tokenizer.segments == []


def test_cyclic_json_is_rejected_but_repeated_objects_are_allowed() -> None:
    cyclic: list[object] = []
    cyclic.append(cyclic)
    source = record()
    source["state"] = cyclic
    with pytest.raises(ValueError, match="finite JSON"):
        public_questions(source)
    shared = ["Valid."]
    source["state"] = [shared, shared]
    assert public_questions(source)[0] == '[["Valid."],["Valid."]]'


@pytest.mark.parametrize(
    "kind,criteria",
    [
        ("score", []),
        ("score", ["Only."]),
        ("score", ["Level."] * 256),
        ("score", {"0": "Low.", "1": "High."}),
        ("choice", ["A.", "B."]),
        ("choice", {"a": "Only."}),
        ("choice", {"": "Blank.", "b": "B."}),
        ("choice", {"a": "", "b": "B."}),
        ("choice", {"a": "A.", "\ud800": "B."}),
        ("noul", {"true": "Only."}),
        ("noul", {"yes": "Yes.", "no": "No."}),
        ("unknown", {"a": "A.", "b": "B."}),
    ],
)
def test_malformed_question_contract(kind: str, criteria: object) -> None:
    with pytest.raises(ValueError):
        public_questions(record(kind, criteria))


@pytest.mark.parametrize("location", ["record", "question"])
def test_labels_and_metadata_are_never_rendered(location: str) -> None:
    source = record()
    target = source if location == "record" else source["questions"]["予約"]
    target["expected"] = "返金"
    with pytest.raises(ValueError, match="permits only"):
        public_questions(source)


@pytest.mark.parametrize(
    "source",
    [
        None,
        [],
        {},
        {"state": "s", "questions": {}},
        '{"state":"s","questions":{},"questions":{}}',
    ],
)
def test_only_complete_decoded_public_records_are_accepted(source: object) -> None:
    with pytest.raises(ValueError):
        public_questions(source)


def test_exact_token_cap_passes_and_one_less_rejects_without_truncation() -> None:
    source = record()
    row = encode_record(CharacterTokenizer(), source)[0]
    assert (
        encode_record(CharacterTokenizer(), source, max_length=len(row.ids))[0] == row
    )
    with pytest.raises(ValueError, match="no truncation"):
        encode_record(CharacterTokenizer(), source, max_length=len(row.ids) - 1)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_max_length(limit: Any) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        encode_record(CharacterTokenizer(), record(), max_length=limit)


def test_byte_budget_rejects_before_tokenizer_and_operational_errors_propagate() -> (
    None
):
    tokenizer = CharacterTokenizer()
    with pytest.raises(ValueError, match="byte limit"):
        encode_record(tokenizer, record(), max_length=1)
    assert tokenizer.segments == []

    class BrokenTokenizer:
        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            raise RuntimeError("tokenizer unavailable")

    with pytest.raises(RuntimeError, match="tokenizer unavailable"):
        encode_record(BrokenTokenizer(), record())


@pytest.mark.parametrize("values", [[], [True], [-1], [1.5], "12"])
def test_invalid_tokenizer_results_are_rejected(values: Any) -> None:
    class InvalidTokenizer:
        def encode(self, text: str, *, add_special_tokens: bool) -> Any:
            return values

    with pytest.raises(ValueError, match="tokenizer"):
        encode_record(InvalidTokenizer(), record())


def test_direct_question_and_encoded_values_cannot_bypass_order_invariants() -> None:
    with pytest.raises(ValueError, match="unique"):
        Question("q", "choice", "Do.", (("same", "A."), ("same", "B.")))
    with pytest.raises(ValueError, match="ordered rubric"):
        Question("q", "score", "Do.", (("1", "Low."), ("0", "High.")))
    with pytest.raises(ValueError, match="exactly true"):
        Question("q", "noul", "Do.", (("yes", "Yes."), ("no", "No.")))
    value = question("choice")
    with pytest.raises(ValueError, match="endpoints"):
        EncodedQuestion(value, (1, 2, 3), (1, 0), 2)
    with pytest.raises(ValueError, match="suffix"):
        EncodedQuestion(value, (1, 2, 3), (0, 1), 1)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ScoreAnswer("q", ("a", "b"), (0.0, 0.0), (0.5, 0.5)),
        lambda: NoulAnswer("q", ("a", "b"), (0.0, 0.0), (0.5, 0.5)),
        lambda: NoulAnswer("q", ([], "true"), (0.0, 0.0), (0.5, 0.5)),  # type: ignore[arg-type]
        lambda: NoulAnswer("q", ("false", "true"), [0.0, 0.0], (0.5, 0.5)),  # type: ignore[arg-type]
        lambda: NoulAnswer("q", ("false", "true"), (0.0, 0.0), (-0.5, 1.5)),
        lambda: NoulAnswer("q", ("false", "true"), (0.0, 0.0), (0.5, 0.0)),
        lambda: ScoreAnswer("q", ("0", "1"), (0.0, 0.0), (math.nan, 0.5)),
        lambda: ChoiceAnswer("q", ("a", "b"), (0.0, math.inf), (0.5, 0.5)),
    ],
)
def test_direct_answers_reject_inconsistent_or_mutable_values(factory: Any) -> None:
    with pytest.raises(ValueError):
        factory()


def test_direct_answer_principal_values_are_derived_without_duplicate_state() -> None:
    choice = ChoiceAnswer("q", ("a", "b"), (1.0, 1.0 + 1e-9), (0.5, 0.5))
    assert choice.logits == (1.0, 1.0)
    assert choice.choice == "a"
    assert choice.ranking == ("a", "b")
    score = ScoreAnswer("q", ("0", "1", "2"), (0.0, 0.0, 0.0), (1 / 6, 1 / 2, 1 / 3))
    assert score.expected_ordinal_value == pytest.approx(7 / 6)
    noul = NoulAnswer("q", ("true", "false"), (0.0, 0.0), (0.75, 0.25))
    assert noul.probability_true == 0.75


def test_custom_output_sequence_is_snapshotted_once_and_actual_count_is_checked() -> (
    None
):
    from collections.abc import Iterator, Sequence

    class ChangingSequence(Sequence[float]):
        def __init__(self, values: list[float]) -> None:
            self.values = values
            self.iterations = 0

        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: Any) -> Any:
            return self.values[index]

        def __iter__(self) -> Iterator[float]:
            self.iterations += 1
            return iter(self.values if self.iterations == 1 else [math.nan])

    values = ChangingSequence([0.0, 1.0])
    answer = interpret(question("choice"), values)
    assert isinstance(answer, ChoiceAnswer)
    assert answer.choice == "返金"
    assert values.iterations == 1
    with pytest.raises(ModelOutputError, match="one value"):
        interpret(question("choice"), ChangingSequence([0.0, 1.0, 2.0]))


def test_custom_output_iteration_failure_propagates_unchanged() -> None:
    from collections.abc import Sequence

    problem = RuntimeError("output iteration unavailable")

    class BrokenSequence(Sequence[float]):
        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: Any) -> Any:
            if index == 0:
                return 0.0
            raise problem

    with pytest.raises(RuntimeError) as failure:
        interpret(question("choice"), BrokenSequence())
    assert failure.value is problem


def test_json_instructions_and_descriptions_become_canonical_text() -> None:
    source: dict[str, Any] = {
        "state": "evidence",
        "questions": {
            "find": {
                "type": "choice",
                "instructions": {"query": "café", "k": 2},
                "criteria": {"a": {"z": 1, "b": [True, None]}, "b": 7, "c": "text"},
            },
            "rank": {"type": "score", "criteria": [{"level": 0}, "middle", 2.5]},
        },
    }
    _, (find, rank) = public_questions(source)
    assert find.instructions == '{"k":2,"query":"café"}'
    assert find.options == (("a", '{"b":[true,null],"z":1}'), ("b", "7"), ("c", "text"))
    assert rank.options == (("0", '{"level":0}'), ("1", "middle"), ("2", "2.5"))
    # The same text as the canonical form the evaluation harness sent.
    plain = copy.deepcopy(source)
    plain["questions"]["find"]["instructions"] = find.instructions
    plain["questions"]["find"]["criteria"] = dict(find.options)
    plain["questions"]["rank"]["criteria"] = [text for _, text in rank.options]
    assert encode_record(CharacterTokenizer(), source) == encode_record(
        CharacterTokenizer(), plain
    )
    for invalid in ({"x": float("nan")}, {1: "key"}, ("tuple",), ""):
        broken = copy.deepcopy(source)
        broken["questions"]["find"]["instructions"] = invalid
        with pytest.raises(ValueError, match="instructions"):
            public_questions(broken)


def test_noul_with_empty_criteria_uses_default_options() -> None:
    assert (
        question("noul", {}).options
        == question("noul").options
        == (
            ("false", "False."),
            ("true", "True."),
        )
    )
    with pytest.raises(ValueError, match="exactly true and false"):
        question("noul", {"true": "Yes."})


@pytest.mark.parametrize("kind", ["choice", "score", "noul"])
def test_typed_answers_share_decision_confidence_metrics(kind: str) -> None:
    from wyat import Candidate, Request
    from wyat.core import decision_from_logits

    criteria = {"choice": None, "score": ["low", "mid", "high"], "noul": None}[kind]
    item = question(kind, criteria)
    logits = [0.7, -1.2, 2.4][: len(item.options)]
    answer = interpret(item, logits)
    request = Request(
        "Decide.",
        tuple(Candidate(key, text) for key, text in item.options),
        task=kind,
    )
    decision = decision_from_logits(request, logits, "fixture")
    # Same definitions; answers hold FP32 logit snapshots, decisions float64.
    for name in ("score_gap", "probability_margin", "entropy", "normalized_entropy"):
        assert getattr(answer, name) == pytest.approx(
            getattr(decision, name), abs=1e-6
        ), name
    best, second = sorted(answer.probabilities, reverse=True)[:2]
    assert answer.probability_margin == pytest.approx(best - second)
    assert 0.0 <= answer.normalized_entropy <= 1.0


# Exact model-facing encoding of every recipe, as a digest of the corpus below.
# A failure means the text the model reads changed. If intended, bump
# tasks.RECIPE and record the new digest beside the old one; artifacts trained
# on the old recipe must then stop loading. Never rewrite an existing entry.
ENCODING_GOLDEN = {
    # Recorded from tasks.py reference source sha256 dd4847e3.
    "wyat-native-typed-segmented-independent-questions-v1": (
        "42c7999f9d462f8a5b5421056a9f2230f6ef4474b9aca8bc27a3c5aa7bc70f98"
    ),
}

GOLDEN_CORPUS: tuple[dict[str, Any], ...] = (
    {
        "state": "plain text",
        "questions": {"q": {"type": "choice", "criteria": {"a": "A.", "b": "B."}}},
    },
    {
        "state": "  leading, trailing\n\twhitespace \n",
        "questions": {
            "q": {
                "type": "choice",
                "instructions": "Pick.",
                "criteria": {"b": "Second key first.", "a": "First."},
            }
        },
    },
    {
        "state": {"z": 1, "a": [True, False, None], "m": {"y": 0.1, "x": -2.5e-07}},
        "questions": {"q": {"type": "noul", "instructions": "Is it?"}},
    },
    {
        "state": [1, 2.0, 1e300, -0.0, 12345678901234567890, "é", " ", "\U0001f600"],
        "questions": {"q": {"type": "noul", "criteria": {}}},
    },
    {"state": None, "questions": {"q": {"type": "noul", "criteria": None}}},
    {
        "state": 0,
        "questions": {
            "q": {
                "type": "noul",
                "criteria": {"true": "Yes, it holds.", "false": "No."},
            }
        },
    },
    {
        "state": True,
        "questions": {
            "q": {"type": "noul", "criteria": {"false": "Nope.", "true": "Yep."}}
        },
    },
    {
        "state": "",
        "questions": {"q": {"type": "score", "criteria": ["none", "some", "all"]}},
    },
    {
        "state": "rubric",
        "questions": {
            "s": {
                "type": "score",
                "instructions": {"rule": "JSON", "n": 2},
                "criteria": [{"level": 0}, ["low"], "high"],
            }
        },
    },
    {
        "state": "json texts",
        "questions": {
            "c": {
                "type": "choice",
                "instructions": ["step", 1],
                "criteria": {"x": {"b": 2, "a": 1}, "y": 3.5, "z": None},
            }
        },
    },
    {
        "state": "multi",
        "questions": {
            "third": {"type": "choice", "criteria": {"1": "un — café", "返金": "two"}},
            "first": {"type": "score", "criteria": ["lo", "hi"]},
            "予約": {"type": "noul", "instructions": "多言語\nline two"},
        },
    },
    {
        "state": {"nested": [[[{"deep": "</option>"}]]]},
        "questions": {
            'q"id': {
                "type": "choice",
                "instructions": '<option>\nquote " and \\ backslash',
                "criteria": {"k\n": "multi\nline", "</option>": "tag-like"},
            }
        },
    },
)


def _golden_digest() -> str:
    tokenizer = CharacterTokenizer()
    rows = [
        [list(row.ids), list(row.candidate_positions), row.query_position]
        for item in GOLDEN_CORPUS
        for row in encode_record(tokenizer, item)
    ]
    payload = json.dumps(
        {"segments": tokenizer.segments, "rows": rows}, ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_encoding_matches_the_recipe_golden() -> None:
    assert RECIPE in ENCODING_GOLDEN, "record the golden digest of a new recipe"
    assert _golden_digest() == ENCODING_GOLDEN[RECIPE], (
        "model-facing encoding changed: restore it, or bump tasks.RECIPE"
    )


def test_state_nested_beyond_the_stack_is_rejected_as_invalid_input() -> None:
    nested: list[object] = []
    for _ in range(sys.getrecursionlimit()):
        nested = [nested]
    with pytest.raises(ValueError, match="nested too deeply"):
        public_questions({"state": nested, "questions": {"q": {"type": "noul"}}})
    with pytest.raises(ValueError, match="nested too deeply"):
        Request.noul("Holds?", context={"deep": nested})
