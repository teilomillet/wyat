"""Stock tiny Qwen3 oracles; synthetic mechanics do not establish learned quality."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from torch import nn
from transformers import Qwen3Config, Qwen3Model
from transformers.cache_utils import DynamicCache, DynamicLayer
from transformers.integrations import sdpa_attention
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3RMSNorm,
)

from benchmarks.analysis.torch.readout import block as experiment


def tiny() -> Qwen3Model:
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        attention_dropout=0.0,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    config._attn_implementation = "sdpa"
    with torch.random.fork_rng():
        torch.manual_seed(21)
        model = Qwen3Model(config)
        cast(nn.Module, model).eval()
        return model


def seed(
    model: Qwen3Model, length: int
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    with torch.inference_mode():
        output = cast(
            BaseModelOutputWithPast,
            model(
                input_ids=torch.arange(length)[None] % 32,
                use_cache=True,
            ),
        )
        assert type(output.past_key_values) is DynamicCache
        return tuple(
            (layer.keys.clone(), layer.values.clone())
            for layer in output.past_key_values.layers
            if isinstance(layer, DynamicLayer)
            and layer.keys is not None
            and layer.values is not None
        )


def fork(
    model: Qwen3Model,
    states: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    batch: int,
) -> DynamicCache:
    with torch.inference_mode():
        return DynamicCache(
            config=model.config,
            ddp_cache_data=tuple(
                (k.expand(batch, -1, -1, -1), v.expand(batch, -1, -1, -1))
                for k, v in states
            ),
        )


def prepare(
    model: Qwen3Model,
    suffix: torch.Tensor,
    cache: DynamicCache,
    prefix: int,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Maintained lower blocks/mask/RoPE before the final-layer experiment."""
    with torch.inference_mode():
        hidden = model.embed_tokens(suffix)
        positions = torch.arange(prefix, prefix + suffix.shape[1])[None]
        embeddings = cast(
            tuple[torch.Tensor, torch.Tensor], model.rotary_emb(hidden, positions)
        )
        mask = create_causal_mask(
            config=model.config,
            inputs_embeds=hidden,
            attention_mask=None,
            past_key_values=cache,
            position_ids=positions,
        )
        assert isinstance(mask, torch.Tensor)
        for layer in model.layers[:-1]:
            hidden = layer(
                hidden,
                attention_mask=mask,
                position_embeddings=embeddings,
                position_ids=positions,
                past_key_values=cache,
                use_cache=True,
            )
        return hidden, embeddings, mask


def evaluate(
    model: Qwen3Model,
    states: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    suffix: torch.Tensor,
    positions: torch.Tensor,
    prefix: int,
) -> tuple[torch.Tensor, DynamicCache]:
    cache = fork(model, states, suffix.shape[0])
    hidden, embeddings, _ = prepare(model, suffix, cache, prefix)
    assert cache.get_seq_length(0) == prefix + suffix.shape[1]
    assert cache.get_seq_length(1) == prefix
    result = experiment.selected_last_block(
        cast(Qwen3DecoderLayer, model.layers[-1]),
        model.norm,
        hidden,
        positions,
        embeddings,
        cache,
        prefix_length=prefix,
    )
    return result, cache


@settings(
    max_examples=12,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(prefix=st.integers(1, 5), length=st.integers(2, 12), batch=st.integers(1, 3))
def test_stock_full_hidden_oracle_ragged_duplicates_and_absolute_rope(
    tmp_path: Path,
    prefix: int,
    length: int,
    batch: int,
) -> None:
    model = tiny()
    states = seed(model, prefix)
    suffix = (torch.arange(batch * length).reshape(batch, length) + 7) % 32
    selected = torch.tensor(
        [
            [0, max(0, length - row - 2), max(0, length - row - 2), 0]
            for row in range(batch)
        ]
    )
    # Row-dependent right padding follows every real readout, including query overlap.
    for row in range(batch):
        suffix[row, max(1, length - row) :] = 0
    actual, cache = evaluate(model, states, suffix, selected, prefix)
    full_ids = torch.cat(
        (torch.arange(prefix)[None].expand(batch, -1) % 32, suffix), dim=1
    )
    with torch.inference_mode():
        full = model(input_ids=full_ids, use_cache=False).last_hidden_state
        expected = full[torch.arange(batch)[:, None], prefix + selected]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert cache.get_seq_length(1) == prefix + length
    assert torch.equal(actual[:, 1], actual[:, 2])
    assert torch.equal(actual[:, 0], actual[:, 3])
    changed = suffix.clone()
    changed[:, -1] = (changed[:, -1] + 1) % 32
    causal, _ = evaluate(model, states, changed, selected, prefix)
    torch.testing.assert_close(causal, actual, atol=0, rtol=0)
    evidence = {
        "prefix": prefix,
        "suffix_length": length,
        "batch": batch,
        "positions": selected.tolist(),
        "expected": expected.tolist(),
        "actual": actual.tolist(),
        "future_perturbed": causal.tolist(),
        "max_abs_difference": (actual - expected).abs().max().item(),
    }
    (tmp_path / f"p{prefix}-s{length}-b{batch}.json").write_text(json.dumps(evidence))


def test_full_stock_final_block_and_batch_specific_rope_oracle() -> None:
    model = tiny()
    states = seed(model, 3)
    suffix = torch.tensor([[4, 5, 6, 7], [7, 6, 5, 4]])
    positions = torch.tensor([[3, 0, 3], [1, 2, 1]])
    control = fork(model, states, 2)
    hidden, embeddings, mask = prepare(model, suffix, control, 3)
    embeddings = (
        embeddings[0].expand(2, -1, -1).contiguous(),
        embeddings[1].expand(2, -1, -1).contiguous(),
    )
    with torch.inference_mode():
        expected = model.norm(
            model.layers[-1](
                hidden,
                attention_mask=mask,
                position_embeddings=embeddings,
                past_key_values=control,
                use_cache=True,
            )
        )[torch.arange(2)[:, None], positions]
        actual = experiment.selected_last_block(
            cast(Qwen3DecoderLayer, model.layers[-1]),
            model.norm,
            hidden,
            positions,
            embeddings,
            fork(model, states, 2),
            prefix_length=3,
        )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_snapshot_interleaving_consumed_owner_and_failure_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tiny()
    states = seed(model, 2)
    copies = tuple((k.clone(), v.clone()) for k, v in states)
    suffix = torch.tensor([[4, 5, 6]])
    positions = torch.tensor([[0, 2, 0]])
    before, cache = evaluate(model, states, suffix, positions, 2)
    evaluate(model, states, torch.tensor([[9, 8], [5, 4]]), torch.tensor([[0], [1]]), 2)
    after, _ = evaluate(model, states, suffix, positions, 2)
    torch.testing.assert_close(after, before, atol=0, rtol=0)
    hidden, embeddings, _ = prepare(model, suffix, fork(model, states, 1), 2)
    with pytest.raises(ValueError, match="fresh"):
        experiment.selected_last_block(
            cast(Qwen3DecoderLayer, model.layers[-1]),
            model.norm,
            hidden,
            positions,
            embeddings,
            cache,
            prefix_length=2,
        )
    failed_cache = fork(model, states, 1)
    original = experiment.sdpa_attention_forward

    def fail(*args: object, **kwargs: object) -> tuple[torch.Tensor, None]:
        raise RuntimeError("suffix failure after append")

    monkeypatch.setattr(experiment, "sdpa_attention_forward", fail)
    with pytest.raises(RuntimeError, match="suffix failure"):
        experiment.selected_last_block(
            cast(Qwen3DecoderLayer, model.layers[-1]),
            model.norm,
            hidden,
            positions,
            embeddings,
            failed_cache,
            prefix_length=2,
        )
    assert failed_cache.get_seq_length(1) == 5
    monkeypatch.setattr(experiment, "sdpa_attention_forward", original)
    recovered, _ = evaluate(model, states, suffix, positions, 2)
    torch.testing.assert_close(recovered, before, atol=0, rtol=0)
    assert all(
        torch.equal(a, b)
        for pair, saved in zip(states, copies, strict=True)
        for a, b in zip(pair, saved, strict=True)
    )


@pytest.mark.parametrize(
    "kind",
    [
        "autocast",
        "training",
        "hook",
        "dtype",
        "rope",
        "index",
        "cache",
        "embeddings",
        "norm",
        "gradient",
        "index_dtype",
    ],
)
def test_unsupported_before_projection_or_cache_mutation(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    model = tiny()
    states = seed(model, 2)
    cache = fork(model, states, 1)
    hidden, embeddings, _ = prepare(model, torch.tensor([[4, 5, 6]]), cache, 2)
    positions = torch.tensor([[0, 2]])
    if kind == "autocast":
        monkeypatch.setattr(torch, "is_autocast_enabled", lambda device: True)
    elif kind == "training":
        model.layers[-1].train()
    elif kind == "hook":
        model.layers[-1].register_forward_hook(lambda *args: None)
    elif kind == "dtype":
        hidden = hidden.double()
    elif kind == "rope":
        model.config.rope_parameters = {"rope_type": "dynamic", "rope_theta": 10000.0}
    elif kind == "index":
        positions[0, 1] = 3
    elif kind == "cache":

        class Foreign(DynamicCache):
            pass

        cache = Foreign(config=model.config)

    norm = model.norm
    if kind == "embeddings":
        embeddings = cast(tuple[torch.Tensor, torch.Tensor], (embeddings[0],))
    elif kind == "norm":
        norm = Qwen3RMSNorm(8).eval()
    elif kind == "gradient":
        hidden = hidden.clone().requires_grad_()
    elif kind == "index_dtype":
        positions = positions.int()

    def forbidden(*args: object, **kwargs: object) -> torch.Tensor:
        pytest.fail("unsupported candidate reached projection")

    monkeypatch.setattr(
        cast(Qwen3DecoderLayer, model.layers[-1]).self_attn.q_proj, "forward", forbidden
    )
    with pytest.raises(ValueError):
        experiment.selected_last_block(
            cast(Qwen3DecoderLayer, model.layers[-1]),
            norm,
            hidden,
            positions,
            embeddings,
            cache,
            prefix_length=2,
        )
    assert all(
        cache.get_seq_length(index) == value
        for index, value in enumerate((5, 2) if kind != "cache" else (0, 0))
    )


@pytest.mark.skipif(
    not (
        sdpa_attention._is_torch_mps_available
        and sdpa_attention._is_torch_greater_or_equal_than_2_13
    ),
    reason="Native Mac masked-GQA observation unsupported; mechanics still tested",
)
def test_native_short_query_masked_cpu_flash_gqa_trace(tmp_path: Path) -> None:
    """Observe one tiny native call; no flag patches or performance claim."""
    model = tiny()
    states = seed(model, 3)
    cache = fork(model, states, 2)
    hidden, embeddings, _ = prepare(
        model, torch.tensor([[4, 5, 6, 7], [8, 9, 10, 11]]), cache, 3
    )
    positions = torch.tensor([[0, 2, 0], [3, 1, 3]])
    observed: dict[str, object] = {}
    original = experiment.sdpa_attention_forward

    def observe(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        **kwargs: object,
    ) -> tuple[torch.Tensor, None]:
        observed.update(
            query_shape=list(query.shape),
            key_shape=list(key.shape),
            value_shape=list(value.shape),
            mask=mask.tolist(),
            mask_shape=list(mask.shape),
            qkv_last_strides=[x.stride(-1) for x in (query, key, value)],
            mask_dtype=str(mask.dtype),
            flags=kwargs,
        )
        return cast(Callable[..., tuple[torch.Tensor, None]], original)(
            module, query, key, value, mask, **kwargs
        )

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(experiment, "sdpa_attention_forward", observe)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True
        ) as profile:
            output = experiment.selected_last_block(
                cast(Qwen3DecoderLayer, model.layers[-1]),
                model.norm,
                hidden,
                positions,
                embeddings,
                cache,
                prefix_length=3,
            )
    operators = {event.key: event.count for event in profile.key_averages()}
    flash_shapes = [
        event.input_shapes
        for event in profile.events()
        if event.name == "aten::_scaled_dot_product_flash_attention_for_cpu"
    ]
    observed["actual_flash_input_shapes"] = flash_shapes
    assert len(flash_shapes) == 1 and flash_shapes[0][:3] == [
        [2, 4, 3, 8],
        [2, 2, 7, 8],
        [2, 2, 7, 8],
    ]
    observed.update(
        operators=operators,
        mps_snapshot=sdpa_attention._is_torch_mps_available,
        torch_gqa_snapshot=sdpa_attention._is_torch_greater_or_equal_than_2_13,
        output=output.tolist(),
    )
    (tmp_path / "native-dispatch.json").write_text(json.dumps(observed, indent=2))
    assert operators.get("aten::_scaled_dot_product_flash_attention_for_cpu") == 1
    assert operators.get("aten::_scaled_dot_product_attention_math", 0) == 0
    assert observed["key_shape"] == [2, 2, 7, 8]
    expected = [
        [[k <= 3 + q for k in range(7)] for q in row] for row in positions.tolist()
    ]
    assert observed["mask"] == [[row] for row in expected]
    # This native dispatch observation is not transferable to other contexts.
    assert (
        sdpa_attention._is_torch_mps_available
        and sdpa_attention._is_torch_greater_or_equal_than_2_13
    )


@pytest.mark.skipif(
    not (
        sdpa_attention._is_torch_mps_available
        and sdpa_attention._is_torch_greater_or_equal_than_2_13
    ),
    reason="Native Mac masked-GQA observation unsupported; mechanics still tested",
)
def test_representative_native_masked_gqa_geometry(tmp_path: Path) -> None:
    """Synthetic real head geometry; no trained model, performance or quality claim."""
    config = Qwen3Config(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
    )
    attention = Qwen3Attention(config, 0).eval()
    query = torch.zeros(8, 16, 7, 128)
    key = torch.zeros(8, 8, 192, 128)
    value = (
        torch.arange(192)[None, None, :, None].float()
        + torch.arange(8)[None, :, None, None].float() * 100
    )
    value = value.expand(8, 8, 192, 128).contiguous()
    positions = torch.tensor([6, 40, 65, 95, 125, 160, 166])
    mask = (
        torch.arange(192)[None, None, None] <= (25 + positions[None, None, :, None])
    ).expand(8, 1, 7, 192)
    with (
        torch.inference_mode(),
        torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
        ) as profile,
    ):
        output, _ = sdpa_attention.sdpa_attention_forward(
            attention,
            query,
            key,
            value,
            mask,
            dropout=0.0,
            scaling=128**-0.5,
            is_causal=False,
        )
    expected = (
        (25 + positions)[None, :, None, None].float() / 2
        + (torch.arange(16) // 2)[None, None, :, None].float() * 100
    ).expand(8, 7, 16, 128)
    torch.testing.assert_close(output, expected, atol=1e-3, rtol=0)
    shapes = [
        event.input_shapes
        for event in profile.events()
        if event.name == "aten::_scaled_dot_product_flash_attention_for_cpu"
    ]
    operators = {event.key: event.count for event in profile.key_averages()}
    evidence = {
        "actual_flash_input_shapes": shapes,
        "operators": operators,
        "mask_shape": list(mask.shape),
        "mask": mask[0, 0].tolist(),
        "mps_snapshot": sdpa_attention._is_torch_mps_available,
        "torch_gqa_snapshot": sdpa_attention._is_torch_greater_or_equal_than_2_13,
        "max_abs_mean_oracle_error": (output - expected).abs().max().item(),
    }
    (tmp_path / "representative-dispatch.json").write_text(
        json.dumps(evidence, indent=2)
    )
    assert len(shapes) == 1 and shapes[0][:3] == [
        [8, 16, 7, 128],
        [8, 8, 192, 128],
        [8, 8, 192, 128],
    ]
    assert operators.get("aten::_scaled_dot_product_attention_math", 0) == 0
