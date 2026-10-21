# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""PyTorch option-fork cross-call prefix-state cache: parity and accounting.

The cache retains the hybrid backbone state after a question's rendered state
block. These tests hold it to the contract that matters: disabled is bitwise the
released behaviour, enabled is within tight tolerance of it, reuse is token
exact, and the LRU accounting cannot leak one record state into another.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
from test_fork_artifact import RECORD, _artifact  # noqa: E402
from transformers.cache_utils import (  # noqa: E402
    DynamicCache,
    LinearAttentionLayer,
)

from wyat.loading import fork_artifact  # noqa: E402
from wyat.loading.fork import ForkScorer, load_fork  # noqa: E402
from wyat.loading.fork_artifact import branch_groups  # noqa: E402

ATOL, RTOL = 2e-5, 1e-4

FOLLOW_UP = {
    "state": RECORD["state"],
    "questions": {
        "followup": {
            "type": "noul",
            "instructions": "True?",
            "criteria": {"false": "no", "true": "yes"},
        }
    },
}

MANY = {
    "state": "x" * 40,
    "questions": {
        f"q{index}": {
            "type": "choice",
            "instructions": f"Pick {index}.",
            "criteria": {f"o{i}": "w" * (1 + i % 9) for i in range(45)},
        }
        for index in range(2)
    },
}


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _artifact(tmp_path_factory.mktemp("fork-prefix-cache"))


def _scorer(artifact: Path, size: int = 0) -> ForkScorer:
    return ForkScorer.from_directory(artifact, prefix_cache_size=size)


def _assert_parity(actual: Any, expected: Any) -> None:
    assert len(actual) == len(expected)
    for left_row, right_row in zip(actual, expected, strict=True):
        left, right = torch.tensor(left_row), torch.tensor(right_row)
        torch.testing.assert_close(left, right, atol=ATOL, rtol=RTOL)
        assert int(left.argmax()) == int(right.argmax())
        order = torch.argsort(left, descending=True, stable=True)
        assert (
            order.tolist()
            == torch.argsort(right, descending=True, stable=True).tolist()
        )


def _legacy_question_logits(scorer: ForkScorer, row: Any) -> list[float]:
    """The released question_logits, before the cross-call prefix cache.

    Copied from the pre-cache implementation: one prefix forward and the same
    branch expansion/readout. A disabled cache must match it bitwise.
    """
    with torch.inference_mode():
        ids, mask = scorer._rows([row.prefix_ids])
        cache = scorer.backbone(
            input_ids=ids, attention_mask=mask, use_cache=True
        ).past_key_values
        prefix = len(row.prefix_ids)
        values = []
        for group in branch_groups(row.option_ids, prefix, scorer.max_batch_tokens):
            width = len(group)
            fork = DynamicCache(config=scorer.backbone.config)
            for source, layer in zip(cache.layers, fork.layers, strict=True):
                if isinstance(source, LinearAttentionLayer):
                    linear = layer
                    conv = source.conv_states[0].expand(width, -1, -1).clone()
                    state = source.recurrent_states[0].expand(width, -1, -1, -1).clone()
                    linear.lazy_initialization(conv_states=conv, recurrent_states=state)
                    linear.conv_states[0] = conv
                    linear.recurrent_states[0] = state
                    linear.is_conv_states_initialized[0] = True
                    linear.is_recurrent_states_initialized[0] = True
                    linear.has_previous_state[0] = True
                else:
                    attention = layer
                    keys = source.keys.expand(width, -1, -1, -1).clone()
                    vals = source.values.expand(width, -1, -1, -1).clone()
                    attention.lazy_initialization(keys, vals)
                    attention.keys, attention.values = keys, vals
            suffix_ids, suffix_mask = scorer._rows(group)
            positions = torch.arange(
                prefix, prefix + suffix_ids.shape[1], device=scorer.device
            )[None, :].expand(width, -1)
            output = scorer.backbone(
                input_ids=suffix_ids,
                attention_mask=torch.cat(
                    [
                        torch.ones(
                            (width, prefix),
                            dtype=suffix_mask.dtype,
                            device=scorer.device,
                        ),
                        suffix_mask,
                    ],
                    dim=1,
                ),
                position_ids=positions,
                past_key_values=fork,
                use_cache=True,
            )
            last = output.last_hidden_state[
                torch.arange(width, device=scorer.device),
                torch.tensor([len(s) - 1 for s in group], device=scorer.device),
            ]
            values.append(scorer.head(last))
        logits = torch.cat(values).float()
    return logits.cpu().tolist()


def test_disabled_cache_is_bitwise_the_released_implementation(
    artifact: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scorer = _scorer(artifact)
    assert scorer.prefix_cache_size == 0
    scorer.max_batch_tokens = 600  # force multiple branch groups on MANY
    rows = [*scorer.encode_record(RECORD), *scorer.encode_record(MANY)]
    assert len(list(branch_groups(rows[-1].option_ids, 1, 600))) > 2

    calls: list[int] = []
    original = scorer.backbone.forward

    def counted(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(scorer.backbone, "forward", counted)
    legacy = [_legacy_question_logits(scorer, row) for row in rows]
    legacy_calls = len(calls)
    calls.clear()

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("disabled cache must not tokenize a state header")

    monkeypatch.setattr("wyat.loading.fork.reuse_prefix", forbidden)
    for row, expected in zip(rows, legacy, strict=True):
        actual = torch.tensor(scorer.question_logits(row))
        assert torch.equal(actual, torch.tensor(expected))
    # Same one prefix forward plus one forward per branch group: no extra work.
    assert len(calls) == legacy_calls
    assert scorer.prefix_cache_currsize == 0
    assert scorer.prefix_cache_hits == 0
    assert scorer.prefix_cache_misses == 0
    assert load_fork(artifact).prefix_cache_size == 0


def test_cached_logits_match_uncached_with_ranks(artifact: Path) -> None:
    uncached = _scorer(artifact)
    cached = _scorer(artifact, 4)
    for record in (RECORD, FOLLOW_UP):
        _assert_parity(cached.logits(record), uncached.logits(record))
    # RECORD's three questions share its state: one miss then two hits, plus the
    # follow-up question on the same state.
    assert (cached.prefix_cache_misses, cached.prefix_cache_hits) == (1, 3)
    assert cached.prefix_cache_currsize == 1
    assert (uncached.prefix_cache_misses, uncached.prefix_cache_hits) == (0, 0)
    assert uncached.prefix_cache_currsize == 0


def test_followup_and_repeat_hit_the_state_block_key(artifact: Path) -> None:
    cached = _scorer(artifact, 4)
    rows = cached.encode_record(RECORD)
    boundary = fork_artifact.reuse_prefix(cached.tokenizer, rows[0])
    assert boundary and len(boundary) < len(rows[0].prefix_ids)
    assert rows[0].prefix_ids[: len(boundary)] == boundary

    cached.logits(RECORD)
    assert cached.prefix_cache_currsize == 1
    assert cached.prefix_cache_misses == 1
    assert cached.prefix_cache_hits == 2
    # The retained key is exactly the state-block token prefix.
    assert list(cached._prefix_cache) == [boundary]

    uncached = _scorer(artifact)
    _assert_parity(cached.logits(FOLLOW_UP), uncached.logits(FOLLOW_UP))
    assert (cached.prefix_cache_misses, cached.prefix_cache_hits) == (1, 3)
    # An exact repeat continues from the same stored state and stores nothing.
    _assert_parity(cached.logits(FOLLOW_UP), uncached.logits(FOLLOW_UP))
    assert (cached.prefix_cache_misses, cached.prefix_cache_hits) == (1, 4)
    assert cached.prefix_cache_currsize == 1


@pytest.mark.parametrize("size", [1, 2])
def test_lru_eviction_stays_bounded_and_exact(artifact: Path, size: int) -> None:
    uncached = _scorer(artifact)
    cached = _scorer(artifact, size)
    records = [
        {
            "state": f"distinct state {index} " + "s" * index,
            "questions": {
                "q": {
                    "type": "choice",
                    "instructions": f"Pick {index}.",
                    "criteria": {"a": "first", "b": "second", "c": "third"},
                }
            },
        }
        for index in range(4)
    ]
    for record in records:
        _assert_parity(cached.logits(record), uncached.logits(record))
        assert cached.prefix_cache_currsize <= size
    assert cached.prefix_cache_currsize == size
    assert cached.prefix_cache_misses == 4
    assert cached.prefix_cache_hits == 0
    # The first state was evicted, so asking for it again is an exacting miss.
    _assert_parity(cached.logits(records[0]), uncached.logits(records[0]))
    assert cached.prefix_cache_misses == 5
    assert cached.prefix_cache_currsize <= size


def test_distinct_states_do_not_contaminate_each_other(artifact: Path) -> None:
    uncached = _scorer(artifact)
    cached = _scorer(artifact, 4)
    alpha = {
        "state": "alpha state",
        "questions": {
            "q": {
                "type": "choice",
                "instructions": "Which?",
                "criteria": {"a": "short", "b": "a much longer option"},
            }
        },
    }
    beta = {
        "state": "beta state, different text",
        "questions": {
            "q": {
                "type": "score",
                "instructions": "Rate.",
                "criteria": ["no", "maybe", "yes", "certainly"],
            }
        },
    }
    expected_alpha = uncached.logits(alpha)
    expected_beta = uncached.logits(beta)
    _assert_parity(cached.logits(alpha), expected_alpha)  # miss, keeps alpha
    _assert_parity(cached.logits(beta), expected_beta)  # miss, keeps beta
    _assert_parity(cached.logits(alpha), expected_alpha)  # hit, alpha intact
    assert (cached.prefix_cache_misses, cached.prefix_cache_hits) == (2, 1)
    assert cached.prefix_cache_currsize == 2


def test_clear_resets_entries_and_counters(artifact: Path) -> None:
    uncached = _scorer(artifact)
    cached = _scorer(artifact, 4)
    cached.logits(RECORD)
    cached.logits(FOLLOW_UP)
    assert cached.prefix_cache_currsize == 1
    assert cached.prefix_cache_hits == 3

    cached.clear_prefix_cache()
    assert cached.prefix_cache_currsize == 0
    assert cached.prefix_cache_hits == 0
    assert cached.prefix_cache_misses == 0

    _assert_parity(cached.logits(FOLLOW_UP), uncached.logits(FOLLOW_UP))
    assert cached.prefix_cache_misses == 1
    assert cached.prefix_cache_hits == 0
    assert cached.prefix_cache_currsize == 1


def test_empty_state_is_a_miss_that_stores_nothing(artifact: Path) -> None:
    uncached = _scorer(artifact)
    cached = _scorer(artifact, 2)
    # A record with no state renders an empty state block: no reusable token
    # prefix exists, so every call is a plain miss that stores nothing.
    record = {
        "state": "",
        "questions": {
            "q": {
                "type": "choice",
                "instructions": "Which?",
                "criteria": {"a": "short", "b": "long"},
            }
        },
    }
    row = cached.encode_record(record)[0]
    assert row.state == ""
    assert fork_artifact.reuse_prefix(cached.tokenizer, row) == ()
    _assert_parity(cached.logits(record), uncached.logits(record))
    _assert_parity(cached.logits(record), uncached.logits(record))
    assert (cached.prefix_cache_misses, cached.prefix_cache_hits) == (2, 0)
    assert cached.prefix_cache_currsize == 0


def test_cached_branch_groups_match_uncached(artifact: Path) -> None:
    uncached = _scorer(artifact)
    cached = _scorer(artifact, 4)
    uncached.max_batch_tokens = cached.max_batch_tokens = 600
    rows = cached.encode_record(MANY)
    groups = branch_groups(rows[0].option_ids, len(rows[0].prefix_ids), 600)
    assert len(list(groups)) > 2

    expected = uncached.logits(MANY)
    _assert_parity(cached.logits(MANY), expected)  # miss then cached question
    assert (cached.prefix_cache_misses, cached.prefix_cache_hits) == (1, 1)
    _assert_parity(cached.logits(MANY), expected)  # both questions hit
    assert (cached.prefix_cache_misses, cached.prefix_cache_hits) == (1, 3)
    assert cached.prefix_cache_currsize == 1


@pytest.mark.parametrize("value", [-1, True, 1.5, "2", None])
def test_invalid_prefix_cache_size_is_refused(artifact: Path, value: Any) -> None:
    with pytest.raises(ValueError, match="prefix_cache_size"):
        ForkScorer.from_directory(artifact, prefix_cache_size=value)
    with pytest.raises(ValueError, match="prefix_cache_size"):
        load_fork(artifact, prefix_cache_size=value)


def test_constructor_validates_prefix_cache_size(artifact: Path) -> None:
    base = _scorer(artifact)
    kwargs = {
        "model_id": base.model_id,
        "max_length": base.max_length,
        "max_batch_tokens": base.max_batch_tokens,
        "input_rendering": None,
    }
    with pytest.raises(ValueError, match="prefix_cache_size"):
        ForkScorer(
            base.backbone, base.head, base.tokenizer, **kwargs, prefix_cache_size=-1
        )
    for size in (0, 3):
        scorer = ForkScorer(
            base.backbone, base.head, base.tokenizer, **kwargs, prefix_cache_size=size
        )
        assert scorer.prefix_cache_size == size
        assert load_fork(artifact, prefix_cache_size=size).prefix_cache_size == size
