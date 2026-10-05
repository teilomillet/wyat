"""One-pass cache ownership and maintained tiny Qwen3 parity, CPU only."""

from __future__ import annotations

import copy
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import cast

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from torch import nn
from transformers import Qwen3Config, Qwen3Model
from transformers.cache_utils import DynamicCache, DynamicLayer
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast

from benchmarks.analysis.torch.prefix.lifetime import KVState, OneUseCache


def config() -> Qwen3Config:
    value = Qwen3Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        rope_parameters={"rope_type": "default", "rope_theta": 1e6},
    )
    value._attn_implementation = "sdpa"
    return value


def seeds(value: Qwen3Config, batch: int = 1, prefix: int = 3) -> tuple[KVState, ...]:
    return tuple(
        (
            torch.full((batch, 2, prefix, 8), float(index)),
            torch.full((batch, 2, prefix, 8), float(index + 1)),
        )
        for index in range(value.num_hidden_layers)
    )


def layers(cache: DynamicCache) -> list[DynamicLayer]:
    assert all(isinstance(layer, DynamicLayer) for layer in cache.layers)
    return cast(list[DynamicLayer], cache.layers)


@contextmanager
def storage_alive(tensor: torch.Tensor) -> Iterator[Callable[[], bool]]:
    # Intrusive storage weak handle observes backing storage, rather than a Python
    # storage wrapper that can die while its tensor still owns the C++ allocation.
    pointer = cast(int, tensor.untyped_storage()._weak_ref())
    # The suspended observer frame must not itself retain the observed tensor.
    del tensor
    try:
        yield lambda: not torch.UntypedStorage._expired(pointer)
    finally:
        cast(Callable[[int], None], torch.UntypedStorage._free_weak_ref)(pointer)


def test_seed_copies_and_returned_storage_survive_reset_until_release() -> None:
    value = config()
    with torch.inference_mode():
        original = seeds(value)
        cache = OneUseCache(value, original)
        assert cache.get_seq_length() == 3 and not cache.consumed
        seeded = layers(cache)[0].keys
        assert isinstance(seeded, torch.Tensor)
        assert seeded.data_ptr() != original[0][0].data_ptr()
        seeded_ref = weakref.ref(seeded)
        with storage_alive(seeded) as seed_storage_alive:
            del seeded
            suffix = torch.ones(1, 2, 4, 8)
            returned = cache.update(suffix, suffix, 0)
            assert seeded_ref() is None and not seed_storage_alive()
        for tensor in returned:
            assert tensor.shape == (1, 2, 7, 8) and tensor.stride(-1) == 1
        del tensor
        assert layers(cache)[0].keys is None and layers(cache)[0].values is None
        assert layers(cache)[1].keys is not None
        returned_ref = weakref.ref(returned[0])
        with storage_alive(returned[0]) as return_storage_alive:
            # Consume the local returned tensors after cache.reset: no zeroing.
            assert return_storage_alive() and returned_ref() is not None
            torch.testing.assert_close(returned[0][:, :, 3:], suffix)
            del returned
            assert returned_ref() is None and not return_storage_alive()
        assert torch.equal(original[0][0], torch.zeros(1, 2, 3, 8))


def test_exact_order_single_pass_and_suffix_geometry_are_atomic() -> None:
    value = config()
    with torch.inference_mode():
        cache = OneUseCache(value, seeds(value))
        suffix = torch.ones(1, 2, 4, 8)
        before = layers(cache)[0].keys
        for index in (1, -1, True):
            with pytest.raises(RuntimeError, match="ordered"):
                cache.update(suffix, suffix, index)
        assert layers(cache)[0].keys is before
        cache.update(suffix, suffix, 0)
        with pytest.raises(ValueError, match="consistent"):
            cache.update(suffix[:, :, :2], suffix[:, :, :2], 1)
        assert layers(cache)[1].keys is not None
        for index in (1, 2):
            cache.update(suffix, suffix, index)
        assert cache.consumed
        assert all(
            layer.keys is None and layer.values is None for layer in layers(cache)
        )
        with pytest.raises(RuntimeError, match="ordered"):
            cache.update(suffix, suffix, 0)


def test_inference_mode_required_at_construction_and_update() -> None:
    value = config()
    original = seeds(value)
    with pytest.raises(RuntimeError, match="inference_mode"):
        OneUseCache(value, original)
    with torch.no_grad(), pytest.raises(RuntimeError, match="inference_mode"):
        OneUseCache(value, original)
    with torch.inference_mode():
        cache = OneUseCache(value, original)
    with pytest.raises(RuntimeError, match="inference_mode"):
        cache.update(torch.ones(1, 2, 2, 8), torch.ones(1, 2, 2, 8), 0)
    assert layers(cache)[0].keys is not None


@pytest.mark.parametrize("boundary", ["sliding", "rope", "dtype", "device"])
def test_scope_rejected_before_seed_mutation(boundary: str) -> None:
    value = config()
    original = seeds(value)
    if boundary == "sliding":
        value.layer_types = ["sliding_attention"] * 3
    elif boundary == "rope":
        value.rope_parameters = {"rope_type": "linear", "factor": 2}
    else:

        def transform(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.double() if boundary == "dtype" else tensor.to("meta")

        original = tuple((transform(key), transform(value)) for key, value in original)

    with torch.inference_mode(), pytest.raises(ValueError):
        OneUseCache(value, original)


@given(prefix=st.integers(1, 5), suffix=st.integers(2, 8), batch=st.integers(1, 3))
@settings(max_examples=6, deadline=None)
def test_maintained_tiny_model_parity_mask_and_no_returned_cache(
    prefix: int, suffix: int, batch: int
) -> None:
    value = config()
    with torch.random.fork_rng():
        torch.manual_seed(59)
        model = Qwen3Model(copy.deepcopy(value))
    cast(nn.Module, model).eval()
    full_ids = torch.arange(prefix + suffix)[None].expand(batch, -1) % 64
    with torch.inference_mode():
        ordinary = cast(Callable[..., DynamicCache], DynamicCache)(config=value)
        model(input_ids=full_ids[:, :prefix], past_key_values=ordinary, use_cache=True)
        states = tuple(
            (cast(torch.Tensor, layer.keys), cast(torch.Tensor, layer.values))
            for layer in layers(ordinary)
        )
        cache = OneUseCache(value, states)
        mask = create_causal_mask(
            config=value,
            inputs_embeds=torch.zeros(batch, suffix, 16),
            past_key_values=cache,
            attention_mask=None,
        )
        assert isinstance(mask, torch.Tensor)
        oracle = (
            torch.arange(prefix + suffix)[None]
            <= prefix + torch.arange(suffix)[:, None]
        )
        assert torch.equal(mask[0, 0], oracle)
        expected = cast(
            BaseModelOutputWithPast, model(input_ids=full_ids, use_cache=False)
        ).last_hidden_state
        actual = cast(
            BaseModelOutputWithPast,
            model(
                input_ids=full_ids[:, prefix:], past_key_values=cache, use_cache=False
            ),
        )
        assert actual.past_key_values is None and cache.consumed
        assert expected is not None and actual.last_hidden_state is not None
        torch.testing.assert_close(
            actual.last_hidden_state, expected[:, prefix:], atol=1e-6, rtol=1e-5
        )
        assert all(
            layer.keys is None and layer.values is None for layer in layers(cache)
        )
        # The maintained reusable cache/prefix tensors remain unchanged.
        assert ordinary.get_seq_length() == prefix
