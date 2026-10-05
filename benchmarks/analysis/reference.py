"""Verify frozen reference inputs without importing a tensor backend."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, cast

from wyat.backends.kai.encoding import EncodedRequest
from wyat.benchmark import BenchmarkCycle
from wyat.core import Request, decision_from_logits
from wyat.evaluation import EvaluationCase, EvaluationRow
from wyat.wire import read_object


class ReferenceEncoder(Protocol):
    model_id: str

    def encode_request(self, request: Request) -> EncodedRequest: ...


def balanced_orders(loads: Sequence[int], repeats: int) -> tuple[tuple[int, ...], ...]:
    """Rotate distinct loads so each occupies every measurement position equally."""
    values = tuple(loads)
    if (
        not values
        or any(type(value) is not int or value < 1 for value in values)
        or len(set(values)) != len(values)
        or type(repeats) is not int
        or repeats < len(values)
        or repeats % len(values)
    ):
        raise ValueError("Use unique positive loads and balanced repetition count")
    return tuple(
        values[offset:] + values[:offset]
        for repeat in range(repeats)
        for offset in (repeat % len(values),)
    )


def read_reference(
    path: Path,
    cases: tuple[EvaluationCase, ...],
    encoder: ReferenceEncoder,
    *,
    dataset_sha256: str,
) -> BenchmarkCycle:
    document = read_object(path)
    raw = document["rows"]
    if not isinstance(raw, list) or len(raw) != len(cases):
        raise ValueError("Reference must contain every declared case")
    if document["model_id"] != encoder.model_id:
        raise ValueError("Loaded model identity differs from frozen CPU reference")
    if document["dataset_sha256"] != dataset_sha256:
        raise ValueError("Reference dataset hash differs from declared cohort")
    rows: list[EvaluationRow] = []
    for case, item in zip(cases, raw, strict=True):
        if not isinstance(item, dict):
            raise ValueError("Reference row must be an object")
        encoded = encoder.encode_request(case.request)
        if (
            item["id"] != case.id
            or item["candidate_ids"]
            != [candidate.id for candidate in case.request.candidates]
            or item["token_ids"] != list(encoded.ids)
            or item["candidate_positions"] != list(encoded.candidate_positions)
            or item["query_position"] != encoded.query_position
        ):
            raise ValueError(f"Reference inputs differ for {case.id}")
        saved = item["legacy_logits"]
        if not isinstance(saved, list) or any(
            type(value) not in (int, float) for value in saved
        ):
            raise ValueError("Reference logits must be numbers")
        prediction = decision_from_logits(
            case.request,
            tuple(float(cast(float, value)) for value in saved),
            encoder.model_id,
        )
        rows.append(
            EvaluationRow(
                case.id,
                case.expected,
                "ok",
                prediction,
                prediction.selected in case.expected,
                0,
                None,
            )
        )
    return BenchmarkCycle(0, tuple(rows), 0, 0)
