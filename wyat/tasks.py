# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Pure values, encoding and readout for the training-native typed recipe.

This module does not load a model. Its recipe is distinct from the single-choice
request/wire contract. Questions are independent causal rows sharing
weights, rather than jointly contextualized questions. Probabilities use stable
stdlib softmax; FP32 reference comparisons require declared numerical tolerance.
Use interpret for the recipe's logits-to-distribution transformation. Direct
answer construction validates supplied snapshots without recomputing softmax.
"""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import ClassVar, Literal, NotRequired, Protocol, TypedDict, cast

from wyat.core import ModelOutputError, _text

RECIPE = "wyat-native-typed-segmented-independent-questions-v1"

type JSONValue = (
    str | bool | int | float | None | list[JSONValue] | dict[str, JSONValue]
)
type TaskType = Literal["choice", "score", "noul"]


class ChoiceQuestion(TypedDict):
    """Ordered application IDs and their descriptions."""

    type: Literal["choice"]
    instructions: NotRequired[str]
    criteria: dict[str, str]


class ScoreQuestion(TypedDict):
    """Ordered rubric descriptions; option IDs are generated ordinal indices."""

    type: Literal["score"]
    instructions: NotRequired[str]
    criteria: list[str]


class NoulQuestion(TypedDict):
    """Binary evidence question; omitted criteria use False./True. descriptions."""

    type: Literal["noul"]
    instructions: NotRequired[str]
    criteria: NotRequired[dict[str, str] | None]


class PublicRecord(TypedDict):
    """Label-free inference inputs, independent of training references."""

    state: JSONValue
    questions: dict[str, ChoiceQuestion | ScoreQuestion | NoulQuestion]


@dataclass(frozen=True, slots=True)
class Question:
    """Validated snapshot of one question in its saved option order."""

    id: str
    type: TaskType
    instructions: str
    options: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _text(self.id, "question ID", nonempty=True)
        _text(self.instructions, "instructions", nonempty=True)
        if self.type not in ("choice", "score", "noul"):
            raise ValueError("unknown question type")
        if not isinstance(self.options, tuple) or not 2 <= len(self.options) <= 255:
            raise ValueError("questions require 2..255 complete options")
        for option in self.options:
            if not isinstance(option, tuple) or len(option) != 2:
                raise ValueError("options must contain immutable ID/description pairs")
            _text(option[0], "option ID", nonempty=True)
            _text(option[1], "description", nonempty=True)
        ids = tuple(key for key, _ in self.options)
        if len(set(ids)) != len(ids):
            raise ValueError("option IDs must be unique")
        if self.type == "score" and ids != tuple(map(str, range(len(ids)))):
            raise ValueError("score IDs must be consecutive ordered rubric indices")
        if self.type == "noul" and set(ids) != {"false", "true"}:
            raise ValueError("noul criteria must contain exactly true and false")


@dataclass(frozen=True, slots=True)
class EncodedQuestion:
    """Independent token row and exact candidate/query readout endpoints."""

    question: Question
    ids: tuple[int, ...]
    candidate_positions: tuple[int, ...]
    query_position: int

    def __post_init__(self) -> None:
        if not isinstance(self.question, Question):
            raise ValueError("question must be a Question")
        if (
            not isinstance(self.ids, tuple)
            or not self.ids
            or any(type(value) is not int or value < 0 for value in self.ids)
        ):
            raise ValueError("ids must be a nonempty tuple of nonnegative token IDs")
        if (
            not isinstance(self.candidate_positions, tuple)
            or len(self.candidate_positions) != len(self.question.options)
            or any(
                type(position) is not int or not 0 <= position < len(self.ids) - 1
                for position in self.candidate_positions
            )
            or any(
                left >= right
                for left, right in zip(
                    self.candidate_positions, self.candidate_positions[1:], strict=False
                )
            )
        ):
            raise ValueError("candidate positions must be ordered option endpoints")
        if (
            type(self.query_position) is not int
            or self.query_position != len(self.ids) - 1
        ):
            raise ValueError("query position must be the final suffix token")


class Tokenizer(Protocol):
    """Tokenizer configuration owns normalization; this recipe adds no tokens."""

    def encode(self, text: str, *, add_special_tokens: bool) -> Sequence[int]: ...


def _canonical_state(value: object) -> str:
    if isinstance(value, str):
        return _text(value, "state")

    # JSON has no cycles; repeated noncyclic objects are valid values.
    ancestors: set[int] = set()

    def validate(item: object) -> None:
        if item is None or type(item) in (bool, int):
            return
        if type(item) is str:
            _text(item, "state text")
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) not in (list, dict) or id(item) in ancestors:
            raise ValueError("state must be finite JSON or text")
        ancestors.add(id(item))
        try:
            if isinstance(item, dict):
                for key, child in item.items():
                    _text(key, "state object key")
                    validate(child)
            else:
                for child in cast(list[object], item):
                    validate(child)
        finally:
            ancestors.remove(id(item))

    validate(value)
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def public_questions(record: object) -> tuple[str, tuple[Question, ...]]:
    """Parse only decoded JSON state/questions, preserving all valid recipe text.

    The JSON transport must reject duplicate keys before building dictionaries;
    overwritten keys cannot be recovered here. References, labels and metadata
    are rejected at this boundary rather than entering model-facing text.
    """
    if type(record) is not dict or set(record) != {"state", "questions"}:
        raise ValueError("public record permits only state and questions")
    state = _canonical_state(record["state"])
    questions = record["questions"]
    if type(questions) is not dict or not questions:
        raise ValueError("questions must be a nonempty mapping")
    parsed = []
    for question_id, body in questions.items():
        _text(question_id, "question ID", nonempty=True)
        if type(body) is not dict or not set(body) <= {
            "type",
            "instructions",
            "criteria",
        }:
            raise ValueError("question permits only type, instructions and criteria")
        kind = body.get("type")
        if kind not in ("choice", "score", "noul"):
            raise ValueError("unknown question type")
        instructions = _text(
            body.get("instructions", question_id), "instructions", nonempty=True
        )
        criteria = body.get("criteria")
        if kind == "noul":
            if criteria is None:
                criteria = {"false": "False.", "true": "True."}
            if type(criteria) is not dict or set(criteria) != {"false", "true"}:
                raise ValueError("noul criteria must contain exactly true and false")
        if kind in ("choice", "noul"):
            if type(criteria) is not dict:
                raise ValueError("choice criteria must map option IDs to descriptions")
            options = tuple(
                (
                    _text(key, "option ID", nonempty=True),
                    _text(description, "description", nonempty=True),
                )
                for key, description in criteria.items()
            )
        else:
            if type(criteria) is not list:
                raise ValueError("score criteria must be an ordered description list")
            options = tuple(
                (str(index), _text(description, "description", nonempty=True))
                for index, description in enumerate(criteria)
            )
        parsed.append(
            Question(question_id, cast(TaskType, kind), instructions, options)
        )
    return state, tuple(parsed)


def encode_record(
    tokenizer: Tokenizer, record: object, *, max_length: int = 8192
) -> tuple[EncodedQuestion, ...]:
    """Encode prefix, complete options and suffix separately; never truncate.

    The executor separately owns the complete padded-question batch budget.
    Tokenizer errors propagate unchanged, rather than becoming output errors.
    """
    if type(max_length) is not int or max_length <= 0:
        raise ValueError("max_length must be a positive integer")
    state, questions = public_questions(record)

    def tokens(segment: str) -> tuple[int, ...]:
        if len(segment.encode("utf-8")) > max_length * 32:
            raise ValueError("segment exceeds tokenization byte limit")
        values = tokenizer.encode(segment, add_special_tokens=False)
        if (
            isinstance(values, (str, bytes, bytearray))
            or not isinstance(values, Sequence)
            or not values
            or any(type(value) is not int or value < 0 for value in values)
        ):
            raise ValueError("tokenizer must return nonempty nonnegative token IDs")
        if len(values) > max_length:
            raise ValueError("segment exceeds max_length; no truncation")
        return tuple(values)

    encoded = []
    for question in questions:
        prefix = (
            f"State:\n{state}\n\nTask type: {question.type}\n"
            f"Question ID: {question.id}\n"
            f"Instructions:\n{question.instructions}\nOptions:"
        )
        ids = list(tokens(prefix))
        positions = []
        for key, description in question.options:
            payload = json.dumps(
                {"key": key, "description": description},
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            ids.extend(tokens("\n<option>\n" + payload + "\n</option>"))
            positions.append(len(ids) - 1)
        ids.extend(
            tokens(
                "\n\nSelect the allowed option best supported by the state "
                "and instructions.\nDecision:"
            )
        )
        if len(ids) > max_length:
            raise ValueError(
                f"{len(ids)} tokens exceed max_length={max_length}; no truncation"
            )
        encoded.append(
            EncodedQuestion(question, tuple(ids), tuple(positions), len(ids) - 1)
        )
    return tuple(encoded)


@dataclass(frozen=True, slots=True)
class _Answer:
    """Readout logits are FP32 snapshots; probabilities use stdlib precision."""

    question_id: str
    option_ids: tuple[str, ...]
    logits: tuple[float, ...]
    probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        _text(self.question_id, "question ID", nonempty=True)
        if (
            not isinstance(self.option_ids, tuple)
            or not 2 <= len(self.option_ids) <= 255
        ):
            raise ValueError("option IDs must be an immutable unique ordered tuple")
        for key in self.option_ids:
            _text(key, "option ID", nonempty=True)
        if len(set(self.option_ids)) != len(self.option_ids):
            raise ValueError("option IDs must be unique")
        if not isinstance(self.logits, tuple) or not isinstance(
            self.probabilities, tuple
        ):
            raise ValueError("answer numbers must be immutable tuples")
        object.__setattr__(self, "logits", _logits(self.logits, len(self.option_ids)))
        probability = _numbers(
            self.probabilities, len(self.option_ids), "probabilities"
        )
        if any(not 0 <= value <= 1 for value in probability) or not math.isclose(
            math.fsum(probability), 1.0, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError("answer probabilities must form a complete distribution")
        object.__setattr__(self, "probabilities", probability)

    @property
    def logits_by_id(self) -> dict[str, float]:
        """Detached FP32 readout-logit mapping in saved option order."""
        return dict(zip(self.option_ids, self.logits, strict=True))

    @property
    def probabilities_by_id(self) -> dict[str, float]:
        """Detached distribution; this recipe makes no calibration claim."""
        return dict(zip(self.option_ids, self.probabilities, strict=True))


@dataclass(frozen=True, slots=True)
class ChoiceAnswer(_Answer):
    """First maximum option and stable descending ranking."""

    type: ClassVar[Literal["choice"]] = "choice"

    @property
    def ranking(self) -> tuple[str, ...]:
        indices = sorted(
            range(len(self.logits)), key=self.logits.__getitem__, reverse=True
        )
        return tuple(self.option_ids[index] for index in indices)

    @property
    def choice(self) -> str:
        return self.ranking[0]

    def to_dict(self) -> dict[str, object]:
        return {
            "type": self.type,
            "probabilities": self.probabilities_by_id,
            "choice": self.choice,
        }


@dataclass(frozen=True, slots=True)
class ScoreAnswer(_Answer):
    """Expected ordinal index; neither a raw logit nor the winning level."""

    type: ClassVar[Literal["score"]] = "score"

    def __post_init__(self) -> None:
        _Answer.__post_init__(self)
        if self.option_ids != tuple(map(str, range(len(self.option_ids)))):
            raise ValueError("score IDs must be consecutive ordered rubric indices")

    @property
    def expected_ordinal_value(self) -> float:
        return math.fsum(
            index * probability for index, probability in enumerate(self.probabilities)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "type": self.type,
            "probabilities": self.probabilities_by_id,
            "expected_ordinal_value": self.expected_ordinal_value,
        }


@dataclass(frozen=True, slots=True)
class NoulAnswer(_Answer):
    """Probability keyed true, without a hidden Boolean threshold."""

    type: ClassVar[Literal["noul"]] = "noul"

    def __post_init__(self) -> None:
        _Answer.__post_init__(self)
        if set(self.option_ids) != {"false", "true"}:
            raise ValueError("noul IDs must contain exactly true and false")

    @property
    def probability_true(self) -> float:
        return self.probabilities[self.option_ids.index("true")]

    def to_dict(self) -> dict[str, object]:
        return {
            "type": self.type,
            "probabilities": self.probabilities_by_id,
            "probability_true": self.probability_true,
        }


def _numbers(values: Sequence[float], count: int, name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ModelOutputError(
            f"{name} must be finite with one value per allowed option"
        )
    snapshot: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ModelOutputError(f"{name} must contain finite numbers")
        try:
            number = float(value)
        except (OverflowError, TypeError, ValueError) as error:
            raise ModelOutputError(f"{name} must be finite") from error
        if not math.isfinite(number):
            raise ModelOutputError(f"{name} must be finite")
        snapshot.append(number)
    if len(snapshot) != count:
        raise ModelOutputError(f"{name} must have one value per allowed option")
    return tuple(snapshot)


def _logits(values: Sequence[float], count: int) -> tuple[float, ...]:
    snapshot = _numbers(values, count, "logits")
    # The historical decoder casts to FP32 before both argmax and softmax.
    # Rounding can create exact ties; double-only argmax would change the ID.
    try:
        rounded = tuple(
            struct.unpack("<f", struct.pack("<f", value))[0] for value in snapshot
        )
    except (OverflowError, struct.error) as error:
        raise ModelOutputError("logits must remain finite in FP32") from error
    if not all(math.isfinite(value) for value in rounded):
        raise ModelOutputError("logits must remain finite in FP32")
    return rounded


def interpret(
    question: Question, logits: Sequence[float]
) -> ChoiceAnswer | ScoreAnswer | NoulAnswer:
    """Interpret ordered finite raw logits with no offsets or temperature.

    Convert to FP32 before ranking and readout, matching the native decoder.
    Choice preserves the first maximum at exact ties. Score averages ordered
    rubric indices. Noul looks up the actual true ID even in reversed menus.
    Stable double softmax comparisons to FP32 use 1e-6 absolute/relative
    tolerance; exact selected IDs and option order cannot use that tolerance.
    Backend consumers must detach tensors to FP32 CPU sequences before calling;
    this stdlib boundary does not import or inspect tensor runtimes.
    """
    if not isinstance(question, Question):
        raise ValueError("question must be a Question")
    option_ids = tuple(key for key, _ in question.options)
    values = _logits(logits, len(option_ids))
    maximum = max(values)
    weights = tuple(math.exp(value - maximum) for value in values)
    total = math.fsum(weights)
    probabilities = tuple(value / total for value in weights)
    if question.type == "choice":
        return ChoiceAnswer(question.id, option_ids, values, probabilities)
    if question.type == "score":
        return ScoreAnswer(question.id, option_ids, values, probabilities)
    return NoulAnswer(question.id, option_ids, values, probabilities)
