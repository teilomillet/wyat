"""Frozen-input checks and measurement order, independent of tensor execution."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from benchmarks.analysis.reference import balanced_orders, read_reference
from wyat.backends.kai.encoding import EncodedRequest
from wyat.core import Candidate, Request
from wyat.evaluation import EvaluationCase


class Encoder:
    model_id = "test-artifact"

    def encode_request(self, request: Request) -> EncodedRequest:
        return EncodedRequest((1, 2, 3, 4), (1, 2), 3)


@given(
    st.lists(st.integers(1, 100), min_size=1, max_size=8, unique=True),
    st.integers(1, 4),
)
def test_rotations_balance_every_load_position(loads: list[int], rounds: int) -> None:
    orders = balanced_orders(loads, len(loads) * rounds)
    assert all(set(order) == set(loads) for order in orders)
    for position in range(len(loads)):
        assert Counter(order[position] for order in orders) == Counter(
            {load: rounds for load in loads}
        )


@pytest.mark.parametrize(
    "loads,repeats", [([], 6), ([1, 1], 2), ([0], 1), ([1, 2, 4], 2), ([1, 2], 3)]
)
def test_invalid_order_is_rejected(loads: list[int], repeats: int) -> None:
    with pytest.raises(ValueError):
        balanced_orders(loads, repeats)


def _case() -> EvaluationCase:
    return EvaluationCase(
        "one", Request("route", (Candidate("a", "A"), Candidate("b", "B"))), ("b",)
    )


def _document() -> dict[str, object]:
    return {
        "model_id": "test-artifact",
        "dataset_sha256": "declared",
        "rows": [
            {
                "id": "one",
                "candidate_ids": ["a", "b"],
                "token_ids": [1, 2, 3, 4],
                "candidate_positions": [1, 2],
                "query_position": 3,
                "legacy_logits": [-1.0, 1.0],
            }
        ],
    }


def test_reference_checks_tokens_and_reconstructs_decision(tmp_path: Path) -> None:
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(_document()))
    cycle = read_reference(path, (_case(),), Encoder(), dataset_sha256="declared")
    assert cycle.correct == 1
    assert cycle.failed == 0
    assert cycle.rows[0].prediction is not None
    assert cycle.rows[0].prediction.ranking == ("b", "a")


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("token_ids", [1, 2, 9, 4]),
        ("candidate_positions", [2, 1]),
        ("candidate_ids", ["b", "a"]),
        ("query_position", 2),
    ],
)
def test_reference_rejects_input_drift(
    tmp_path: Path, field: str, replacement: object
) -> None:
    document = _document()
    rows = document["rows"]
    assert isinstance(rows, list)
    rows[0][field] = replacement
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="Reference inputs differ"):
        read_reference(path, (_case(),), Encoder(), dataset_sha256="declared")
