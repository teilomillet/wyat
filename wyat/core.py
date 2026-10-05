"""Immutable decision values and the sole runtime boundary: batches of logits."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Protocol, cast


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


@dataclass(frozen=True, slots=True)
class Request:
    """One decision's inputs, independent of a model or HTTP connection.

    Query supplies the instruction; context supplies supporting information.
    Candidates are an immutable ordered tuple with unique application-owned IDs.
    Construct with named fields to make the two text inputs easy to distinguish.
    The loaded scorer enforces its supported tasks and input-size limits.
    """

    query: str
    candidates: tuple[Candidate, ...]
    context: str = ""
    task: str = "choice"

    def __post_init__(self) -> None:
        _text(self.query, "query", nonempty=True)
        _text(self.context, "context")
        _text(self.task, "task", nonempty=True)
        if not isinstance(self.candidates, tuple) or not self.candidates:
            raise ValueError("candidates must be a nonempty tuple")
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


class ModelOutputError(ValueError):
    """A scorer violated its output contract, rather than rejecting an input."""


@dataclass(frozen=True, slots=True)
class Decision:
    """Raw logits and probabilities follow input candidate order.

    Ranking contains candidate IDs in descending score order, preserving input
    order for exact ties. Softmax probabilities are not calibrated confidence.
    """

    model_id: str
    scores: tuple[float, ...]
    probabilities: tuple[float, ...]
    ranking: tuple[str, ...]

    @property
    def selected(self) -> str:
        return self.ranking[0]

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
    if len(logits) != len(request.candidates):
        raise ModelOutputError("scorer returned a different candidate count")
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in logits):
        raise ModelOutputError("scorer logits must be numbers")
    scores = tuple(float(value) for value in logits)
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
        if len(logits) != len(batch):
            raise ModelOutputError("scorer returned a different request count")
        return tuple(
            decision_from_logits(request, scores, self.model_id)
            for request, scores in zip(batch, logits, strict=True)
        )


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
