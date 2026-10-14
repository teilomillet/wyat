# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Immutable decision values and the sole runtime boundary: batches of logits."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal, Protocol, cast, overload


def _text(value: object, name: str, *, nonempty: bool = False) -> str:
    if not isinstance(value, str) or (nonempty and not value.strip()):
        raise ValueError(f"{name} must be {'nonempty ' if nonempty else ''}text")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{name} must contain valid Unicode") from error
    return value


def _canonical_schema(schema: object) -> str:
    if not isinstance(schema, Mapping) or not schema:
        raise ValueError("candidate schema must be a nonempty mapping")
    ancestors: set[int] = set()

    def normalize(item: object, *, root: bool = False) -> object:
        if item is None or isinstance(item, bool):
            return item
        if isinstance(item, int):
            return int(item)
        if isinstance(item, str):
            return _text(item, "candidate schema text")
        if isinstance(item, Real):
            number = float(item)
            if not math.isfinite(number):
                raise ValueError("candidate schema must contain finite JSON values")
            return number
        if id(item) in ancestors:
            raise ValueError("candidate schema must not contain cyclic references")
        if isinstance(item, Mapping):
            ancestors.add(id(item))
            try:
                return {
                    _text(key, "candidate schema key", nonempty=root): normalize(child)
                    for key, child in item.items()
                }
            finally:
                ancestors.remove(id(item))
        if isinstance(item, Sequence) and not isinstance(item, (bytes, bytearray)):
            ancestors.add(id(item))
            try:
                return [normalize(child) for child in item]
            finally:
                ancestors.remove(id(item))
        raise ValueError("candidate schema must contain finite JSON values")

    try:
        normalized = normalize(schema, root=True)
    except RecursionError as error:
        raise ValueError("candidate schema is nested too deeply") from error
    return json.dumps(
        normalized,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


@dataclass(frozen=True, slots=True)
class Candidate:
    """An application-owned candidate ID and the description the model reads."""

    id: str
    text: str

    def __post_init__(self) -> None:
        _text(self.id, "candidate.id", nonempty=True)
        _text(self.text, "candidate.text", nonempty=True)

    @classmethod
    def from_schema(cls, id: str, schema: Mapping[str, object]) -> Candidate:
        """Bind a structured tool or action schema directly to its candidate line."""
        return cls(id=id, text=_canonical_schema(schema))


_NOUL_CANDIDATES: tuple[Candidate, ...] = (
    Candidate("false", "False."),
    Candidate("true", "True."),
)


@dataclass(frozen=True, slots=True, init=False)
class Request:
    """One decision's inputs, independent of a model or HTTP connection.

    Query supplies the instruction; context supplies supporting information,
    as text or a JSON object or array stored in canonical JSON text.
    Ordered candidate sequences are stored as an immutable tuple with unique
    application-owned IDs.
    Construct with named fields to make the two text inputs easy to distinguish.
    The loaded scorer enforces its supported tasks and input-size limits.
    """

    query: str
    candidates: tuple[Candidate, ...]
    context: str = ""
    task: str = "choice"

    def __init__(
        self,
        query: str,
        candidates: Sequence[Candidate] = (),
        context: str | dict[str, Any] | list[Any] = "",
        task: str = "choice",
    ) -> None:
        if isinstance(candidates, (str, bytes, bytearray, Mapping)) or not isinstance(
            candidates, Sequence
        ):
            raise ValueError("candidates must be a nonempty ordered sequence")
        resolved = tuple(candidates)
        if not resolved and task == "noul":
            resolved = _NOUL_CANDIDATES
        object.__setattr__(self, "query", query)
        object.__setattr__(self, "candidates", resolved)
        if type(context) in (dict, list):
            # The same canonical text a PublicRecord state renders for predict.
            from . import tasks

            context = tasks._canonical_state(context)
        object.__setattr__(self, "context", context)
        object.__setattr__(self, "task", task)
        self.__post_init__()

    def __post_init__(self) -> None:
        _text(self.query, "query", nonempty=True)
        _text(self.context, "context")
        _text(self.task, "task", nonempty=True)
        if not isinstance(self.candidates, tuple) or not self.candidates:
            raise ValueError("candidates must be a nonempty ordered sequence")
        if not all(isinstance(candidate, Candidate) for candidate in self.candidates):
            raise ValueError("candidates must contain Candidate values")
        ids = [candidate.id for candidate in self.candidates]
        if len(set(ids)) != len(ids):
            raise ValueError("candidate IDs must be unique")

    @classmethod
    def choice(
        cls,
        query: str,
        candidates: Sequence[Candidate] | Mapping[str, str],
        *,
        context: str | dict[str, Any] | list[Any] = "",
    ) -> Request:
        """Construct a categorical choice request from Candidates or a mapping."""
        if isinstance(candidates, Mapping):
            resolved = tuple(
                Candidate(cand_id, text) for cand_id, text in candidates.items()
            )
            return cls(query, resolved, context=context, task="choice")
        return cls(query, candidates, context=context, task="choice")

    @classmethod
    def noul(
        cls,
        query: str,
        *,
        context: str | dict[str, Any] | list[Any] = "",
        candidates: Sequence[Candidate] | None = None,
        false_text: str = "False.",
        true_text: str = "True.",
    ) -> Request:
        """Construct a binary evidence / abstention (noul) request."""
        if candidates is not None:
            return cls(query, candidates, context=context, task="noul")
        if false_text == "False." and true_text == "True.":
            return cls(query, _NOUL_CANDIDATES, context=context, task="noul")
        return cls(
            query,
            (Candidate("false", false_text), Candidate("true", true_text)),
            context=context,
            task="noul",
        )

    @classmethod
    def score(
        cls,
        query: str,
        rubric: Sequence[Candidate | str],
        *,
        context: str | dict[str, Any] | list[Any] = "",
    ) -> Request:
        """Construct an ordinal rubric score request from descriptions or Candidates."""
        if (
            isinstance(rubric, (str, bytes, bytearray, Mapping))
            or not isinstance(rubric, Sequence)
            or len(rubric) < 2
        ):
            raise ValueError("score rubric must contain at least two ordered levels")
        resolved: list[Candidate] = []
        for index, item in enumerate(rubric):
            if isinstance(item, Candidate):
                resolved.append(item)
            elif isinstance(item, str):
                resolved.append(Candidate(str(index), item))
            else:
                raise ValueError(
                    "score rubric must contain Candidate or text descriptions"
                )
        return cls(query, tuple(resolved), context=context, task="score")


class Scorer(Protocol):
    """Return finite raw logits in request and candidate order.

    Sigmoid scores, embeddings, and calibrated probabilities require their own
    model contracts; they do not satisfy this interface by renaming outputs.
    Implementations own batching, model state, tokenization, and resource limits.
    """

    def score_many(self, requests: Sequence[Request]) -> Sequence[Sequence[float]]: ...


_MODEL_INPUT_REASONS = frozenset(
    {"unsupported-task", "candidate-count", "input-length", "input-segment"}
)


class ModelInputError(ValueError):
    """A model rejected input with a safe, application-readable reason.

    The message is for local diagnostics. Transports expose only the allowlisted
    reason, never arbitrary exception text or request content.
    """

    def __init__(
        self,
        message: str,
        *,
        reason: Literal[
            "unsupported-task", "candidate-count", "input-length", "input-segment"
        ],
    ) -> None:
        if type(reason) is not str or reason not in _MODEL_INPUT_REASONS:
            raise ValueError("unknown model input rejection reason")
        self._reason: str = reason
        super().__init__(message)

    @property
    def reason(self) -> str:
        return self._reason


class ModelOutputError(ValueError):
    """A scorer violated its output contract, rather than rejecting an input."""


def _score_gap(scores: Sequence[float]) -> float:
    """Highest minus second-highest raw score; callers ensure two values."""
    best, second = sorted((float(value) for value in scores), reverse=True)[:2]
    return best - second


def _probability_margin(probabilities: Sequence[float]) -> float:
    """Highest minus second-highest probability; callers ensure two values."""
    best, second = sorted(probabilities, reverse=True)[:2]
    return best - second


def _entropy(probabilities: Sequence[float]) -> float:
    """Shannon entropy in nats."""
    return -math.fsum(p * math.log(p) for p in probabilities if p > 0.0)


def _normalized_entropy(probabilities: Sequence[float]) -> float:
    """Entropy divided by ln(K), in [0, 1] for K >= 2 values."""
    return _entropy(probabilities) / math.log(len(probabilities))


@dataclass(frozen=True, slots=True)
class Decision:
    """Raw logits and probabilities follow input candidate order.

    Ranking contains candidate IDs in descending score order, preserving input
    order for exact ties. Derived candidate IDs and mappings rely on this stable
    ranking contract. Softmax probabilities are not calibrated confidence.
    """

    model_id: str
    scores: tuple[float, ...]
    probabilities: tuple[float, ...]
    ranking: tuple[str, ...]

    @property
    def selected(self) -> str:
        return self.ranking[0]

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        """Candidate IDs in original score order for a coherent result."""
        self._validate_ranking()
        # Invert the stable score ranking to preserve the four-field dataclass
        # and wire format, including coherent legacy four-argument results.
        order = sorted(
            range(len(self.scores)), key=self.scores.__getitem__, reverse=True
        )
        ids = [""] * len(order)
        for index, candidate_id in zip(order, self.ranking, strict=True):
            ids[index] = candidate_id
        return tuple(ids)

    def _validate_ranking(self) -> None:
        self._validate_numbers(self.scores, "scores")
        if not isinstance(self.ranking, Sequence) or isinstance(
            self.ranking, (str, bytes, bytearray)
        ):
            raise ValueError("decision.ranking must be a sequence of candidate IDs")
        if len(self.ranking) != len(self.scores):
            raise ValueError("decision ranking and scores must have equal lengths")
        for candidate_id in self.ranking:
            _text(candidate_id, "decision candidate ID", nonempty=True)
        if len(set(self.ranking)) != len(self.ranking):
            raise ValueError("decision candidate IDs must be unique")

    @property
    def scores_by_id(self) -> dict[str, float]:
        """A detached score dictionary in original candidate order."""
        return dict(zip(self.candidate_ids, self.scores, strict=True))

    @property
    def probabilities_by_id(self) -> dict[str, float]:
        """A detached probability dictionary in original candidate order."""
        ids = self.candidate_ids
        self._validate_probabilities()
        return dict(zip(ids, self.probabilities, strict=True))

    def _validate_probabilities(self) -> None:
        self._validate_numbers(self.probabilities, "probabilities")
        if len(self.probabilities) != len(self.scores):
            raise ValueError(
                "decision probabilities and scores must have equal lengths"
            )

    @property
    def score_gap(self) -> float:
        """Difference between the highest and second-highest raw candidate scores."""
        self._validate_numbers(self.scores, "scores")
        if len(self.scores) < 2:
            raise ValueError("decision.score_gap requires at least two candidates")
        return _score_gap(self.scores)

    @property
    def probability_margin(self) -> float:
        """Difference between the highest and second-highest probabilities."""
        probs = self._validated_distribution()
        if len(probs) < 2:
            raise ValueError(
                "decision.probability_margin requires at least two candidates"
            )
        return _probability_margin(probs)

    @property
    def entropy(self) -> float:
        """Shannon entropy in nats over the candidate probability distribution."""
        return _entropy(self._validated_distribution())

    @property
    def normalized_entropy(self) -> float:
        """Shannon entropy normalized by ln(K) into [0, 1] for K >= 2 candidates."""
        probs = self._validated_distribution()
        if len(probs) < 2:
            raise ValueError(
                "decision.normalized_entropy requires at least two candidates"
            )
        return _normalized_entropy(probs)

    @property
    def probability_true(self) -> float:
        """Probability of 'true' for a binary noul decision."""
        self._validated_distribution()
        by_id = self.probabilities_by_id
        if set(by_id) != {"false", "true"} or len(by_id) != 2:
            raise ValueError(
                "decision.probability_true requires binary "
                "'false' and 'true' candidates"
            )
        return by_id["true"]

    @property
    def expected_ordinal_value(self) -> float:
        """Probability-weighted expected ordinal index (0..K-1) for a score decision."""
        self._validated_distribution()
        by_id = self.probabilities_by_id
        if len(by_id) < 2:
            raise ValueError(
                "decision.expected_ordinal_value requires at least two candidates"
            )
        expected_ids = tuple(str(index) for index in range(len(by_id)))
        if tuple(by_id) != expected_ids:
            raise ValueError(
                "decision.expected_ordinal_value requires sequential "
                "'0'..'K-1' candidate IDs"
            )
        return math.fsum(
            index * probability for index, probability in enumerate(by_id.values())
        )

    def _validated_distribution(self) -> tuple[float, ...]:
        self._validate_numbers(self.probabilities, "probabilities")
        probs = tuple(float(v) for v in self.probabilities)
        if any(v < 0.0 or v > 1.0 for v in probs) or not math.isclose(
            math.fsum(probs), 1.0, rel_tol=1e-5, abs_tol=1e-5
        ):
            raise ValueError(
                "decision.probabilities must form a valid probability distribution"
            )
        return probs

    @staticmethod
    def _validate_numbers(values: Sequence[float], name: str) -> None:
        if (
            not isinstance(values, Sequence)
            or isinstance(values, (str, bytes, bytearray))
            or not values
        ):
            raise ValueError(f"decision.{name} must be a nonempty numeric sequence")
        for value in values:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"decision.{name} must contain finite numbers")
            try:
                finite = math.isfinite(value)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(
                    f"decision.{name} must contain finite numbers"
                ) from error
            if not finite:
                raise ValueError(f"decision.{name} must contain finite numbers")

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "selected": self.selected,
            "ranking": list(self.ranking),
            "scores": list(self.scores),
            "probabilities": list(self.probabilities),
        }


def _validate_temperature(temperature: object) -> float:
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, Real)
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("temperature must be a positive finite number")
    return float(temperature)


def decision_from_logits(
    request: Request,
    logits: Sequence[float],
    model_id: str,
    *,
    temperature: float = 1.0,
) -> Decision:
    """Pure interpretation; softmax values are relative, uncalibrated likelihoods."""
    temp = _validate_temperature(temperature)
    try:
        candidate_count = len(logits)
        values = iter(logits)
    except TypeError as error:
        raise ModelOutputError("scorer logits must be a sequence") from error
    if candidate_count != len(request.candidates):
        raise ModelOutputError("scorer returned a different candidate count")
    converted: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ModelOutputError("scorer logits must be numbers")
        try:
            converted.append(float(value))
        except (TypeError, ValueError, OverflowError) as error:
            raise ModelOutputError("scorer logits must be finite numbers") from error
    if len(converted) != len(request.candidates):
        raise ModelOutputError("scorer returned a different candidate count")
    scores = tuple(converted)
    if not all(math.isfinite(value) for value in scores):
        raise ModelOutputError("scorer logits must be finite")
    peak = max(scores)
    weights = tuple(math.exp((value - peak) / temp) for value in scores)
    total = math.fsum(weights)
    probabilities = tuple(weight / total for weight in weights)
    # Python's stable sort preserves caller order for exact ties.
    order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
    return Decision(
        model_id,
        scores,
        probabilities,
        tuple(request.candidates[index].id for index in order),
    )


class Model:
    """A loaded logit scorer. Loading and inference are separate lifetimes."""

    def __init__(
        self, scorer: Scorer, model_id: str, *, temperature: float = 1.0
    ) -> None:
        _text(model_id, "model_id", nonempty=True)
        self._scorer = scorer
        self._model_id = model_id
        self._temperature = _validate_temperature(temperature)

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def temperature(self) -> float:
        return self._temperature

    def decide(self, request: Request, *, temperature: float | None = None) -> Decision:
        """Score one validated input using this already-loaded model."""
        return self.decide_many((request,), temperature=temperature)[0]

    @overload
    def decide_many(
        self,
        requests: Sequence[Request],
        *,
        temperature: float | None = None,
        errors: Literal["raise"] = "raise",
    ) -> tuple[Decision, ...]: ...

    @overload
    def decide_many(
        self,
        requests: Sequence[Request],
        *,
        temperature: float | None = None,
        errors: Literal["return"],
    ) -> tuple[Decision | ModelInputError, ...]: ...

    def decide_many(
        self,
        requests: Sequence[Request],
        *,
        temperature: float | None = None,
        errors: Literal["raise", "return"] = "raise",
    ) -> tuple[Decision, ...] | tuple[Decision | ModelInputError, ...]:
        """Score one batch; results follow request order. Empty batches do no work.

        With ``errors="return"``, a request the model refuses gets its
        ModelInputError in its result slot and the other requests are still
        decided. Output-contract and backend exceptions still raise.
        """
        batch = tuple(requests)
        if not all(isinstance(request, Request) for request in batch):
            raise ValueError("batch must contain Request values")
        if errors not in ("raise", "return"):
            raise ValueError("errors must be 'raise' or 'return'")
        eff_temp = (
            self._temperature
            if temperature is None
            else _validate_temperature(temperature)
        )
        if not batch:
            return ()
        if errors == "return":
            return tuple(self._decide_isolating(batch, eff_temp))
        return self._decide_batch(batch, eff_temp)

    def _decide_isolating(
        self, batch: tuple[Request, ...], temperature: float
    ) -> list[Decision | ModelInputError]:
        # Bisect on refusals, as the server's microbatcher does, so a batch-level
        # refusal (such as a token budget) still decides every request it can.
        try:
            return list(self._decide_batch(batch, temperature))
        except ModelInputError as error:
            if len(batch) == 1:
                return [error]
            middle = len(batch) // 2
            return self._decide_isolating(
                batch[:middle], temperature
            ) + self._decide_isolating(batch[middle:], temperature)

    def _decide_batch(
        self, batch: tuple[Request, ...], eff_temp: float
    ) -> tuple[Decision, ...]:
        logits = self._scorer.score_many(batch)
        try:
            request_count = len(logits)
            rows = iter(logits)
        except TypeError as error:
            raise ModelOutputError(
                "scorer output must be a sequence of rows"
            ) from error
        if request_count != len(batch):
            raise ModelOutputError("scorer returned a different request count")
        decisions: list[Decision] = []
        for request in batch:
            try:
                scores = next(rows)
            except StopIteration as error:
                raise ModelOutputError(
                    "scorer returned a different request count"
                ) from error
            decisions.append(
                decision_from_logits(
                    request, scores, self.model_id, temperature=eff_temp
                )
            )
        try:
            next(rows)
        except StopIteration:
            return tuple(decisions)
        raise ModelOutputError("scorer returned a different request count")

    def logits(self, record: object) -> list[list[float]]:
        """Return one ordered FP32 logit vector per question in a PublicRecord."""
        from . import tasks

        scorer_logits = getattr(self._scorer, "logits", None)
        if callable(scorer_logits):
            return cast(list[list[float]], scorer_logits(record))
        state, questions = tasks.public_questions(record)
        requests = tuple(
            Request(
                question.instructions,
                tuple(Candidate(opt_id, desc) for opt_id, desc in question.options),
                context=state,
                task=question.type,
            )
            for question in questions
        )
        decisions = self.decide_many(requests, temperature=1.0)
        return [
            list(tasks._logits(decision.scores, len(question.options)))
            for question, decision in zip(questions, decisions, strict=True)
        ]

    def predict(
        self, record: object, *, temperature: float | None = None
    ) -> tasks.Answers:
        """Score a PublicRecord and return typed answers keyed by question ID."""
        from . import tasks

        eff_temp = (
            self._temperature
            if temperature is None
            else _validate_temperature(temperature)
        )
        scorer_predict = getattr(self._scorer, "predict", None)
        if eff_temp == 1.0 and callable(scorer_predict):
            return self._identified(
                record,
                cast(
                    dict[
                        str, tasks.ChoiceAnswer | tasks.ScoreAnswer | tasks.NoulAnswer
                    ],
                    scorer_predict(record),
                ),
            )
        records_logits = getattr(self._scorer, "_records_logits", None)
        if callable(records_logits):
            encoded, raw_rows = cast(
                list[tuple[tuple[tasks.EncodedQuestion, ...], list[list[float]]]],
                records_logits((record,)),
            )[0]
            questions = tuple(row.question for row in encoded)
        else:
            _, questions = tasks.public_questions(record)
            raw_rows = self.logits(record)
        return self._identified(record, _answers(questions, raw_rows, eff_temp))

    def predict_many(
        self, records: Sequence[object], *, temperature: float | None = None
    ) -> tuple[tasks.Answers, ...]:
        """Score PublicRecords together; answers follow record order.

        Native scorers encode every record before scoring any, so a refused
        record fails the call before compute. Each record is then scored in its
        own budgeted groups to keep its shared-prefix reuse. Other scorers score
        one record at a time.
        """
        from . import tasks

        if isinstance(records, (str, bytes, bytearray, Mapping)) or not isinstance(
            records, Sequence
        ):
            raise ValueError("records must be an ordered sequence")
        batch = tuple(records)
        eff_temp = (
            self._temperature
            if temperature is None
            else _validate_temperature(temperature)
        )
        if not batch:
            return ()
        records_logits = getattr(self._scorer, "_records_logits", None)
        if not callable(records_logits):
            return tuple(self.predict(record, temperature=eff_temp) for record in batch)
        scored = cast(
            list[tuple[tuple[tasks.EncodedQuestion, ...], list[list[float]]]],
            records_logits(batch),
        )
        return tuple(
            self._identified(
                record,
                _answers(tuple(row.question for row in encoded), rows, eff_temp),
            )
            for record, (encoded, rows) in zip(batch, scored, strict=True)
        )

    def _identified(
        self,
        record: object,
        answers: dict[str, tasks.ChoiceAnswer | tasks.ScoreAnswer | tasks.NoulAnswer],
    ) -> tasks.Answers:
        """Stamp answers with this model and a digest of the caller's input."""
        from dataclasses import replace

        from . import tasks

        state, questions = tasks.public_questions(record)
        digests = {
            question.id: tasks.question_input_sha256(state, question)
            for question in questions
        }
        return tasks.Answers(
            {
                key: replace(
                    answer,
                    model_id=self.model_id,
                    input_sha256=digests[answer.question_id],
                )
                for key, answer in answers.items()
            }
        )


def _answers(
    questions: Sequence[tasks.Question],
    raw_rows: Sequence[Sequence[float]],
    temperature: float,
) -> dict[str, tasks.ChoiceAnswer | tasks.ScoreAnswer | tasks.NoulAnswer]:
    """Interpret one record's logit rows as typed answers at a temperature."""
    from . import tasks

    answers: dict[str, tasks.ChoiceAnswer | tasks.ScoreAnswer | tasks.NoulAnswer] = {}
    for question, row_logits in zip(questions, raw_rows, strict=True):
        if temperature == 1.0:
            answers[question.id] = tasks.interpret(question, row_logits)
            continue
        option_ids = tuple(key for key, _ in question.options)
        values = tasks._logits(row_logits, len(option_ids))
        peak = max(values)
        weights = tuple(math.exp((value - peak) / temperature) for value in values)
        total = math.fsum(weights)
        probabilities = tuple(weight / total for weight in weights)
        if question.type == "choice":
            answers[question.id] = tasks.ChoiceAnswer(
                question.id, option_ids, values, probabilities
            )
        elif question.type == "score":
            answers[question.id] = tasks.ScoreAnswer(
                question.id, option_ids, values, probabilities
            )
        else:
            answers[question.id] = tasks.NoulAnswer(
                question.id, option_ids, values, probabilities
            )
    return answers


def request_from_dict(value: object) -> Request:
    if not isinstance(value, Mapping):
        raise ValueError("request must be an object")
    value = cast(Mapping[str, object], value)
    if set(value) - {"query", "candidates", "context", "task"}:
        raise ValueError("unknown request fields")
    if "query" not in value:
        raise ValueError("request requires query and candidates")
    task = _text(value.get("task", "choice"), "task", nonempty=True)
    if "candidates" not in value:
        if task == "noul":
            return Request(
                _text(value["query"], "query", nonempty=True),
                _NOUL_CANDIDATES,
                _text(value.get("context", ""), "context"),
                task,
            )
        raise ValueError("request requires query and candidates")
    items = value["candidates"]
    if not isinstance(items, list):
        raise ValueError("candidates must be an array")
    candidates: list[Candidate] = []
    for index, item in enumerate(items):
        if task == "score" and isinstance(item, str):
            candidates.append(
                Candidate(str(index), _text(item, "candidate.text", nonempty=True))
            )
            continue
        if not isinstance(item, Mapping) or set(item) != {"id", "text"}:
            raise ValueError("candidate requires exactly id and text")
        item = cast(Mapping[str, object], item)
        candidates.append(
            Candidate(
                _text(item["id"], "candidate.id", nonempty=True),
                _text(item["text"], "candidate.text", nonempty=True),
            )
        )
    return Request(
        _text(value["query"], "query", nonempty=True),
        tuple(candidates),
        _text(value.get("context", ""), "context"),
        task,
    )


from . import tasks  # noqa: E402, F401
