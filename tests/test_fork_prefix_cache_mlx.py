# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""MLX option-fork retained state-block prefix cache: parity, hits, LRU."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
torch = pytest.importorskip("torch")
from test_fork_artifact import RECORD, _artifact, _config  # noqa: E402

import wyat  # noqa: E402
from wyat.loading import fork_artifact  # noqa: E402
from wyat.loading.fork import ForkScorer  # noqa: E402
from wyat.loading.fork_mlx import load_fork_mlx  # noqa: E402

DEVICES = [
    "cpu",
    pytest.param(
        "gpu",
        marks=pytest.mark.skipif(
            not mx.metal.is_available(), reason="requires Apple Metal"
        ),
    ),
]


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    # The Metal gated-delta kernel needs key heads at least 32 wide.
    config = _config(257)
    config.linear_key_head_dim = config.linear_value_head_dim = 32
    return _artifact(tmp_path_factory.mktemp("fork-prefix"), config=config)


FOLLOW_UP = {
    "state": copy.deepcopy(RECORD["state"]),
    "questions": {
        "followup": {
            "type": "noul",
            "instructions": "Should it proceed?",
            "criteria": {"false": "no", "true": "yes"},
        }
    },
}

OTHER_STATE = {
    "state": {"text": "Un état différent", "count": 11},
    "questions": {
        "other": {
            "type": "choice",
            "instructions": "Choose.",
            "criteria": {"a": "first", "b": "second"},
        }
    },
}


def _uncached(artifact: Path, device: str) -> object:
    return load_fork_mlx(artifact, device=device, prefix_cache_size=0)


def _cached(artifact: Path, device: str, size: int) -> object:
    return load_fork_mlx(artifact, device=device, prefix_cache_size=size)


def _assert_parity(
    uncached: object, cached: object, record: object, tolerance: float = 2e-5
) -> None:
    left = uncached.logits(record)  # type: ignore[attr-defined]
    right = cached.logits(record)  # type: ignore[attr-defined]
    assert len(left) == len(right)
    for expected, actual in zip(left, right, strict=True):
        torch.testing.assert_close(
            torch.tensor(actual), torch.tensor(expected), atol=tolerance, rtol=1e-4
        )
        assert max(range(len(expected)), key=expected.__getitem__) == max(
            range(len(actual)), key=actual.__getitem__
        )
        assert sorted(
            range(len(expected)), key=expected.__getitem__, reverse=True
        ) == sorted(range(len(actual)), key=actual.__getitem__, reverse=True)


@pytest.mark.parametrize("device", DEVICES)
def test_mlx_prefix_cache_multi_question_record(artifact: Path, device: str) -> None:
    uncached = _uncached(artifact, device)
    cached = _cached(artifact, device, 4)
    _assert_parity(uncached, cached, RECORD)
    # The three questions share one state block: one miss, two hits, one entry.
    assert cached.prefix_cache_misses == 1
    assert cached.prefix_cache_hits == 2
    assert cached.prefix_cache_currsize == 1


@pytest.mark.parametrize("device", DEVICES)
def test_mlx_prefix_cache_follow_up_and_repeat(artifact: Path, device: str) -> None:
    uncached = _uncached(artifact, device)
    cached = _cached(artifact, device, 4)
    _assert_parity(uncached, cached, RECORD)
    _assert_parity(uncached, cached, FOLLOW_UP)
    assert cached.prefix_cache_hits == 3
    # Repeating the same question is served from the same retained prefix.
    _assert_parity(uncached, cached, FOLLOW_UP)
    assert cached.prefix_cache_hits == 4
    assert cached.prefix_cache_misses == 1


@pytest.mark.parametrize("device", DEVICES)
def test_mlx_prefix_cache_distinct_states_do_not_contaminate(
    artifact: Path, device: str
) -> None:
    uncached = _uncached(artifact, device)
    cached = _cached(artifact, device, 4)
    _assert_parity(uncached, cached, RECORD)
    _assert_parity(uncached, cached, OTHER_STATE)
    _assert_parity(uncached, cached, FOLLOW_UP)
    _assert_parity(uncached, cached, RECORD)
    assert cached.prefix_cache_currsize <= 4


@pytest.mark.parametrize("device", DEVICES)
def test_mlx_prefix_cache_lru_eviction_is_bounded(artifact: Path, device: str) -> None:
    uncached = _uncached(artifact, device)
    cached = _cached(artifact, device, 1)
    for record in (RECORD, OTHER_STATE, RECORD, FOLLOW_UP):
        _assert_parity(uncached, cached, record)
        assert cached.prefix_cache_currsize <= 1


@pytest.mark.parametrize("device", DEVICES)
def test_mlx_prefix_cache_clear_resets(artifact: Path, device: str) -> None:
    cached = _cached(artifact, device, 4)
    cached.logits(RECORD)
    assert cached.prefix_cache_currsize == 1
    assert cached.prefix_cache_hits + cached.prefix_cache_misses == 3
    cached.clear_prefix_cache()
    assert cached.prefix_cache_currsize == 0
    assert cached.prefix_cache_hits == 0
    assert cached.prefix_cache_misses == 0


@pytest.mark.parametrize("device", DEVICES)
def test_mlx_prefix_cache_disabled_keeps_full_prefill(
    artifact: Path, device: str
) -> None:
    scorer = _uncached(artifact, device)
    _ = scorer.logits(RECORD)
    assert scorer.prefix_cache_currsize == 0
    assert scorer.prefix_cache_hits == 0
    assert scorer.prefix_cache_misses == 0


def test_reuse_prefix_is_an_exact_token_prefix(artifact: Path) -> None:
    scorer = load_fork_mlx(artifact, device="cpu", prefix_cache_size=0)
    for row in scorer.encode_record(RECORD):
        reused = fork_artifact.reuse_prefix(scorer.tokenizer, row)
        assert reused
        assert len(reused) < len(row.prefix_ids)
        assert row.prefix_ids[: len(reused)] == reused
        assert row.state
    empty = fork_artifact.ForkQuestion(
        scorer.encode_record(RECORD)[0].question, (1, 2, 3), ((4,),), ""
    )
    assert fork_artifact.reuse_prefix(scorer.tokenizer, empty) == ()


def test_fork_prefix_cache_size_is_validated(artifact: Path) -> None:
    for bad in (-1, True, 1.5):
        with pytest.raises(ValueError, match="prefix_cache_size"):
            load_fork_mlx(artifact, device="cpu", prefix_cache_size=bad)  # type: ignore[arg-type]


def test_public_load_wires_fork_prefix_cache_size(artifact: Path) -> None:
    default = wyat.load(artifact, backend="mlx", device="cpu")
    assert default._scorer.prefix_cache_size == 0  # type: ignore[attr-defined]
    enabled = wyat.load(artifact, backend="mlx", device="cpu", prefix_cache_size=2)
    assert enabled._scorer.prefix_cache_size == 2  # type: ignore[attr-defined]
    disabled = wyat.load(artifact, backend="mlx", device="cpu", prefix_cache_size=0)
    assert disabled._scorer.prefix_cache_size == 0  # type: ignore[attr-defined]


def test_mlx_prefix_cache_preserves_candidate_order(artifact: Path) -> None:
    scorer = _cached(artifact, device="cpu", size=4)
    swapped = json.loads(json.dumps(RECORD))
    swapped["questions"]["choice"]["criteria"] = {
        "c": "x",
        "a": "short",
        "b": "a much longer option " * 3,
    }
    left = scorer.logits(RECORD)[0]
    right = scorer.logits(swapped)[0]
    torch.testing.assert_close(
        torch.tensor(right),
        torch.tensor([left[2], left[0], left[1]]),
        atol=2e-5,
        rtol=1e-4,
    )


def test_torch_reference_still_available(artifact: Path) -> None:
    """The PyTorch runtime must remain usable as the parity reference."""
    scorer = ForkScorer.from_directory(artifact)
    assert len(scorer.logits(RECORD)) == 3
