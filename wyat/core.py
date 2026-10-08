"""Immutable decision values and the sole runtime boundary: batches of logits."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Literal, Protocol, cast


def _text(value: object, name: str, *, nonempty: bool = False) -> str:
    if not isinstance(value, str) or (nonempty and not value.strip()):
        raise ValueError(f"{name} must be {'nonempty ' if nonempty else ''}text")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{name} must contain valid Unicode") from error
    return value


@dataclass(frozen=True, slots=True)
class Candidate:
    """An application-owned candidate ID and the description the model reads."""

    id: str
    text: str

    def __post_init__(self) -> None:
        _text(self.id, "candidate.id", nonempty=True)
        _text(self.text, "candidate.text", nonempty=True)


@dataclass(frozen=True, slots=True, init=False)
class Request:
    """One decision's inputs, independent of a model or HTTP connection.

    Query supplies the instruction; context supplies supporting information.
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
        candidates: Sequence[Candidate],
        context: str = "",
        task: str = "choice",
    ) -> None:
        if isinstance(candidates, (str, bytes, bytearray, Mapping)) or not isinstance(
            candidates, Sequence
        ):
            raise ValueError("candidates must be a nonempty ordered sequence")
        object.__setattr__(self, "query", query)
        object.__setattr__(self, "candidates", tuple(candidates))
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
        # Invert the stable score ranking to preserve the four-field dataclass
        # and wire format, including coherent legacy four-argument results.
        order = sorted(
            range(len(self.scores)), key=self.scores.__getitem__, reverse=True
        )
        ids = [""] * len(order)
        for index, candidate_id in zip(order, self.ranking, strict=True):
            ids[index] = candidate_id
        return tuple(ids)

    @property
    def scores_by_id(self) -> dict[str, float]:
        """A detached score dictionary in original candidate order."""
        return dict(zip(self.candidate_ids, self.scores, strict=True))

    @property
    def probabilities_by_id(self) -> dict[str, float]:
        """A detached probability dictionary in original candidate order."""
        ids = self.candidate_ids
        self._validate_numbers(self.probabilities, "probabilities")
        if len(self.probabilities) != len(ids):
            raise ValueError(
                "decision probabilities and scores must have equal lengths"
            )
        return dict(zip(ids, self.probabilities, strict=True))

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


def decision_from_logits(
    request: Request, logits: Sequence[float], model_id: str
) -> Decision:
    """Pure interpretation; softmax values are relative, uncalibrated likelihoods."""
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
    weights = tuple(math.exp(value - peak) for value in scores)
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

    def __init__(self, scorer: Scorer, model_id: str) -> None:
        _text(model_id, "model_id", nonempty=True)
        self._scorer = scorer
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    def decide(self, request: Request) -> Decision:
        """Score one validated input using this already-loaded model."""
        return self.decide_many((request,))[0]

    def decide_many(self, requests: Sequence[Request]) -> tuple[Decision, ...]:
        """Score one batch; results follow request order. Empty batches do no work."""
        batch = tuple(requests)
        if not all(isinstance(request, Request) for request in batch):
            raise ValueError("batch must contain Request values")
        if not batch:
            return ()
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
            decisions.append(decision_from_logits(request, scores, self.model_id))
        try:
            next(rows)
        except StopIteration:
            return tuple(decisions)
        raise ModelOutputError("scorer returned a different request count")


def request_from_dict(value: object) -> Request:
    if not isinstance(value, Mapping):
        raise ValueError("request must be an object")
    value = cast(Mapping[str, object], value)
    if set(value) - {"query", "candidates", "context", "task"}:
        raise ValueError("unknown request fields")
    if "query" not in value or "candidates" not in value:
        raise ValueError("request requires query and candidates")
    items = value["candidates"]
    if not isinstance(items, list):
        raise ValueError("candidates must be an array")
    candidates: list[Candidate] = []
    for item in items:
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
        _text(value.get("task", "choice"), "task", nonempty=True),
    )
