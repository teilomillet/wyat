"""Maintained-cache mechanics on tiny asymmetric Qwen3 weights, CPU only."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

pytest.importorskip("mlx.core", exc_type=ImportError)
pytest.importorskip("mlx_lm", exc_type=ImportError)

import mlx.core as mx
from mlx_lm.models.qwen3 import ModelArgs, Qwen3Model

from benchmarks.analysis.mlx.prefix import PrefixScorer
from wyat.backends.wyat.encoding import WyatEncoder
from wyat.backends.wyat.mlx import MLXCandidateHead, MLXWyatScorer
from wyat.core import Candidate, Request


class CharacterTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return [ord(character) % 128 for character in text]


def request(choices: Sequence[str], *, context: str = "shared public state") -> Request:
    return Request(
        query="Pick the action.",
        context=context,
        candidates=tuple(
            Candidate(str(index), text) for index, text in enumerate(choices)
        ),
    )


@pytest.fixture
def base() -> MLXWyatScorer:
    args = ModelArgs(
        model_type="qwen3",
        hidden_size=8,
        num_hidden_layers=2,
        intermediate_size=12,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        rms_norm_eps=1e-6,
        vocab_size=128,
        max_position_embeddings=512,
        rope_theta=10000,
        tie_word_embeddings=True,
    )
    with mx.stream(mx.cpu):
        mx.random.seed(17)
        backbone = Qwen3Model(args)
        head = MLXCandidateHead(8, 4)
        backbone.eval()
        head.eval()
    return MLXWyatScorer(
        backbone,
        head,
        WyatEncoder(CharacterTokenizer(), max_length=512, max_options=8),
        model_id="owned-tiny-asymmetric-prefix-fixture",
        device=mx.Device(mx.cpu),
    )


@pytest.mark.parametrize("prefix_length", [1, 25])
def test_causal_suffix_matches_full_scalar_batch_and_interleaved_calls(
    base: MLXWyatScorer, prefix_length: int
) -> None:
    values = [request(("yes", "no")), request(("a", "b", "c"))]
    prefix = base.encode_request(values[0]).ids[:prefix_length]
    cached = PrefixScorer(base, prefix)
    assert base.backbone.layers[0].self_attn.q_proj.weight.shape == (16, 8)
    states_before = [
        (cast(list[object], keys.tolist()), cast(list[object], values.tolist()))
        for keys, values in cached._states
    ]
    for chunk in (values, values[:1], values[1:], values[::-1], values):
        actual = cached.score_many(chunk)
        expected = base.score_many(chunk)
        for row, control in zip(actual, expected, strict=True):
            assert row == pytest.approx(control, abs=2e-6, rel=1e-5)
    assert states_before == [
        (cast(list[object], keys.tolist()), cast(list[object], values.tolist()))
        for keys, values in cached._states
    ]
    for cache in cached._fork(3):
        assert cache.offset == prefix_length
        assert cache.state[0].shape[:3] == (3, 2, prefix_length)
    assert cached.prefix_state_bytes == 2 * 2 * prefix_length * 4 * 2 * 4
    assert cached.score_many([]) == []


def test_suffix_beyond_cache_growth_step_preserves_frozen_state(
    base: MLXWyatScorer,
) -> None:
    value = request(("long candidate " * 8, "short"))
    encoded = base.encode_request(value)
    assert len(encoded.ids) - 25 > 256
    cached = PrefixScorer(base, encoded.ids[:25])
    for _ in range(2):
        assert cached.score_many([value])[0] == pytest.approx(
            base.score_many([value])[0], abs=2e-6, rel=1e-5
        )
        assert all(
            keys.shape[2] == values.shape[2] == 25 for keys, values in cached._states
        )


def test_mixed_prefix_batch_falls_back_intact(base: MLXWyatScorer) -> None:
    admitted = request(("a", "b"))
    mismatched = request(("a", "b"), context="different public state")
    cached = PrefixScorer(base, base.encode_request(admitted).ids[:25])
    assert cached._admits(base.encode_request(admitted))
    assert not cached._admits(base.encode_request(mismatched))
    # Fallback reuses exactly the baseline path, not a partially split batch.
    assert cached.score_many([admitted, mismatched]) == base.score_many(
        [admitted, mismatched]
    )


@given(
    menus=st.lists(
        st.lists(
            st.text(alphabet="abé", min_size=1, max_size=7), min_size=2, max_size=5
        ),
        min_size=1,
        max_size=3,
    )
)
@settings(
    max_examples=8,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_generated_causal_batches_preserve_readouts(
    base: MLXWyatScorer, menus: list[list[str]]
) -> None:
    values = [request(menu) for menu in menus]
    cached = PrefixScorer(base, base.encode_request(values[0]).ids[:25])
    for row, control in zip(
        cached.score_many(values), base.score_many(values), strict=True
    ):
        assert row == pytest.approx(control, abs=2e-6, rel=1e-5)


def test_prefix_overlapping_candidate_readout_falls_back(base: MLXWyatScorer) -> None:
    value = request(("a", "b"))
    encoded = base.encode_request(value)
    prefix_length = encoded.candidate_positions[0] + 1
    cached = PrefixScorer(base, encoded.ids[:prefix_length])
    assert not cached._admits(encoded)
    assert cached.score_many([value]) == base.score_many([value])


@pytest.mark.parametrize("prefix", [(), (-1,), (128,), (True,)])
def test_invalid_prefix_ids_rejected(
    base: MLXWyatScorer, prefix: tuple[int, ...]
) -> None:
    with pytest.raises(ValueError, match="Prefix"):
        PrefixScorer(base, prefix)
