# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Tiny local causal and ownership oracles for explicit CPU prefix reuse."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest
import torch
from fixtures.wyat import create_wyat_bundle
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from torch import nn
from transformers.cache_utils import DynamicLayer
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast

from wyat import Candidate, Model, Request
from wyat.backends.wyat.cpu import prefix as runtime
from wyat.backends.wyat.cpu.prefix import PrefixScorer
from wyat.backends.wyat.encoding import EncodedRequest
from wyat.backends.wyat.torch import WyatScorer


@pytest.fixture
def base(tmp_path: Path) -> WyatScorer:
    return WyatScorer.from_directory(
        create_wyat_bundle(tmp_path / "wyat"), max_length=512
    )


def request(query: str = "read?", count: int = 2) -> Request:
    return Request(
        query,
        tuple(Candidate(str(i), "read " * (i + 1)) for i in range(count)),
        context="Bonjour",
    )


def suffix_hidden(
    base: WyatScorer, prefix: PrefixScorer, ids: torch.Tensor
) -> torch.Tensor:
    with torch.inference_mode():
        result = cast(
            BaseModelOutputWithPast,
            base.backbone(
                input_ids=ids,
                past_key_values=prefix._fork(ids.shape[0]),
                use_cache=True,
            ),
        )
    assert result.last_hidden_state is not None
    return result.last_hidden_state


@settings(
    max_examples=8,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    prefix_length=st.integers(1, 5),
    suffix_length=st.integers(2, 9),
    batch=st.integers(1, 3),
)
def test_absolute_offset_mask_and_suffix_match_full_causal_oracle(
    base: WyatScorer, prefix_length: int, suffix_length: int, batch: int
) -> None:
    vocab = base.backbone.config.vocab_size
    tokens = tuple((i + 1) % vocab for i in range(prefix_length))
    retained = PrefixScorer(base, tokens)
    suffix = (torch.arange(batch * suffix_length).reshape(batch, -1) + 3) % vocab
    full = torch.cat((torch.tensor([tokens]).expand(batch, -1), suffix), dim=1)
    with torch.inference_mode():
        expected = base.backbone(input_ids=full, use_cache=False).last_hidden_state[
            :, prefix_length:
        ]
        mask = create_causal_mask(
            config=base.backbone.config,
            inputs_embeds=torch.zeros(batch, suffix_length, 16),
            attention_mask=None,
            past_key_values=retained._fork(batch),
        )
    oracle = torch.tensor(
        [
            [k <= prefix_length + q for k in range(prefix_length + suffix_length)]
            for q in range(suffix_length)
        ]
    )
    assert isinstance(mask, torch.Tensor) and torch.equal(
        mask, oracle[None, None].expand(batch, 1, -1, -1)
    )
    actual = suffix_hidden(base, retained, suffix)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    future = suffix.clone()
    future[:, -1] = (future[:, -1] + 1) % vocab
    changed = suffix_hidden(base, retained, future)
    torch.testing.assert_close(changed[:, :-1], actual[:, :-1], atol=0, rtol=0)
    # The rectangular upper-left rule excludes already retained tokens incorrectly.
    wrong = torch.ones(
        suffix_length, prefix_length + suffix_length, dtype=torch.bool
    ).tril()
    assert not torch.equal(wrong, oracle)


def test_ragged_public_decisions_preserve_candidate_order_and_base_defaults(
    base: WyatScorer,
) -> None:
    inputs = (request(), request("write? " * 3, 3), request("é 世界?", 4))
    retained = PrefixScorer(base, base.encode_request(inputs[0]).ids[:2])
    state = {name: value.clone() for name, value in base.backbone.state_dict().items()}
    head_state = {name: value.clone() for name, value in base.head.state_dict().items()}
    config = base.backbone.config.to_dict()
    device = base.device
    threads = torch.get_num_threads()
    ordinary, cached = Model(base, base.model_id), Model(retained, retained.model_id)
    for batch in (inputs, inputs[::-1], inputs[:1], inputs):
        for actual, expected in zip(
            cached.decide_many(batch), ordinary.decide_many(batch), strict=True
        ):
            torch.testing.assert_close(
                torch.tensor(actual.scores),
                torch.tensor(expected.scores),
                atol=1e-6,
                rtol=1e-5,
            )
            assert (
                actual.ranking == expected.ranking
                and actual.selected == expected.selected
            )
    assert retained.score_many(()) == [] and cached.decide_many(()) == ()
    assert base.device == device and base.backbone.config.to_dict() == config
    assert torch.get_num_threads() == threads
    assert all(
        torch.equal(value, state[name])
        for name, value in base.backbone.state_dict().items()
    )
    assert all(
        torch.equal(value, head_state[name])
        for name, value in base.head.state_dict().items()
    )


def test_fresh_cache_storage_and_interleaving_do_not_mutate_prefix(
    base: WyatScorer,
) -> None:
    retained = PrefixScorer(base, (1, 2, 3))
    snapshots = tuple((k.clone(), v.clone()) for k, v in retained._states)
    a, b = retained._fork(2), retained._fork(1)
    for first, second, seed in zip(a.layers, b.layers, retained._states, strict=True):
        assert type(first) is DynamicLayer and type(second) is DynamicLayer
        assert (
            first.keys is not None
            and first.values is not None
            and second.keys is not None
        )
        assert first.keys.data_ptr() != second.keys.data_ptr() != seed[0].data_ptr()
        assert first.keys.data_ptr() != seed[0].data_ptr()
    suffix = torch.tensor([[4, 5, 6], [7, 8, 9]])
    initial = suffix_hidden(base, retained, suffix)
    suffix_hidden(base, retained, torch.tensor([[3, 2]]))
    torch.testing.assert_close(
        suffix_hidden(base, retained, suffix), initial, atol=0, rtol=0
    )
    assert a.get_seq_length() == b.get_seq_length() == 3
    assert retained.prefix == (1, 2, 3)
    assert retained.prefix_state_bytes == sum(
        t.numel() * t.element_size() for pair in snapshots for t in pair
    )
    assert all(
        torch.equal(actual, expected)
        for pair, snapshot in zip(retained._states, snapshots, strict=True)
        for actual, expected in zip(pair, snapshot, strict=True)
    )


@pytest.mark.parametrize("kind", ["mismatch", "candidate", "query", "empty"])
def test_one_nonadmitted_member_uses_whole_original_batch(
    base: WyatScorer, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    inputs = (request(), request("write?"))
    row = base.encode_request(inputs[0])
    retained = PrefixScorer(base, row.ids[:2])
    nonadmitted = {
        "mismatch": EncodedRequest(
            (row.ids[0] + 1, *row.ids[1:]), row.candidate_positions, row.query_position
        ),
        "candidate": EncodedRequest(
            row.ids, (1, row.candidate_positions[1]), row.query_position
        ),
        "query": EncodedRequest(row.ids, row.candidate_positions, 1),
        "empty": EncodedRequest(retained.prefix, (0, 1), 1),
    }[kind]
    calls: list[tuple[EncodedRequest, ...]] = []

    def encode(value: Request) -> EncodedRequest:
        return row if value is inputs[0] else nonadmitted

    def score(encoded: Sequence[EncodedRequest], **kwargs: object) -> list[list[float]]:
        assert kwargs == {}, "Fallback must retain ordinary full prefill"
        calls.append(tuple(encoded))
        return [[1.0, 2.0], [3.0, 4.0]]

    monkeypatch.setattr(base, "encode_request", encode)
    monkeypatch.setattr(base, "_score_encoded", score)
    assert retained.score_many(inputs) == [[1.0, 2.0], [3.0, 4.0]]
    assert calls == [(row, nonadmitted)]


@pytest.mark.parametrize(
    "invalid",
    [
        request("read " * 600),
        Request("read?", request().candidates, task="score"),
        Request("read?", (Candidate("a", "read"),)),
    ],
)
def test_complete_original_input_admission_precedes_prefix_removal(
    base: WyatScorer, monkeypatch: pytest.MonkeyPatch, invalid: Request
) -> None:
    retained = PrefixScorer(base, base.encode_request(request()).ids[:2])

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("No model call before complete input admission")

    monkeypatch.setattr(base.backbone, "forward", forbidden)
    with pytest.raises(ValueError):
        retained.score_many((request(), invalid))


@pytest.mark.parametrize(
    "tokens", [[], (), (True,), (-1,), (100000,), (1.0,), (1,) * 512]
)
def test_invalid_prefix_is_rejected_before_any_backbone_call(
    base: WyatScorer, monkeypatch: pytest.MonkeyPatch, tokens: object
) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("No prefill for invalid prefix")

    monkeypatch.setattr(base.backbone, "forward", forbidden)
    with pytest.raises((ValueError, TypeError)):
        PrefixScorer(base, cast(tuple[int, ...], tokens))


@pytest.mark.parametrize("when", ["construction", "call"])
def test_ambient_cpu_autocast_is_rejected_and_caller_context_preserved(
    base: WyatScorer, monkeypatch: pytest.MonkeyPatch, when: str
) -> None:
    retained = (
        PrefixScorer(base, base.encode_request(request()).ids[:2])
        if when == "call"
        else None
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("No prefill/suffix under ambient CPU autocast")

    monkeypatch.setattr(base.backbone, "forward", forbidden)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        assert torch.is_autocast_enabled("cpu")
        with pytest.raises(ValueError, match="autocast"):
            if retained is None:
                PrefixScorer(base, (1, 2))
            else:
                retained.score_many((request(),))
        assert torch.is_autocast_enabled("cpu")
    assert not torch.is_autocast_enabled("cpu")


@pytest.mark.parametrize("when", ["construction", "call"])
def test_backbone_hooks_reject_before_cached_extra_effects(
    base: WyatScorer, when: str
) -> None:
    retained = (
        PrefixScorer(base, base.encode_request(request()).ids[:2])
        if when == "call"
        else None
    )
    events: list[str] = []

    def hook(module: nn.Module, args: tuple[object, ...], result: object) -> None:
        events.append("effect")

    handle = base.backbone.layers[0].register_forward_hook(hook)
    try:
        with pytest.raises(ValueError, match="hook"):
            if retained is None:
                PrefixScorer(base, (1, 2))
            else:
                retained.score_many((request(),))
        assert events == []
    finally:
        handle.remove()


def test_ordinary_shared_helper_keeps_exact_forward_arguments(
    base: WyatScorer, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []
    ordinary = base.backbone.forward

    def observe(*args: object, **kwargs: object) -> BaseModelOutputWithPast:
        assert args == () and set(kwargs) == {"input_ids", "use_cache"}
        assert kwargs["use_cache"] is False
        calls.append(kwargs)
        return cast(
            BaseModelOutputWithPast,
            ordinary(
                input_ids=cast(torch.Tensor, kwargs["input_ids"]), use_cache=False
            ),
        )

    monkeypatch.setattr(base.backbone, "forward", observe)
    assert len(base.score_many((request(), request("write?", 3)))) == 2
    assert base.score_many(()) == [] and len(calls) == 1


def test_suffix_failure_does_not_poison_retained_cache(
    base: WyatScorer, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = request()
    retained = PrefixScorer(base, base.encode_request(value).ids[:2])
    expected = retained.score_many((value,))
    original = base.backbone.forward
    snapshots = tuple((k.clone(), v.clone()) for k, v in retained._states)

    def fail(*args: object, **kwargs: object) -> object:
        original(
            input_ids=cast(torch.Tensor, kwargs["input_ids"]),
            past_key_values=kwargs["past_key_values"],
            use_cache=True,
        )
        raise RuntimeError("failure after private cache update")

    with monkeypatch.context() as scoped:
        scoped.setattr(base.backbone, "forward", fail)
        with pytest.raises(RuntimeError, match="private cache"):
            retained.score_many((value,))
    assert retained.score_many((value,)) == expected
    assert all(
        torch.equal(actual, previous)
        for pair, snapshot in zip(retained._states, snapshots, strict=True)
        for actual, previous in zip(pair, snapshot, strict=True)
    )


@pytest.mark.parametrize("condition", ["train", "half", "eager", "rope", "sliding"])
def test_unsupported_runtime_rejects_before_prefill(
    base: WyatScorer, monkeypatch: pytest.MonkeyPatch, condition: str
) -> None:
    if condition == "train":
        cast(nn.Module, base.backbone).train()
    elif condition == "half":
        cast(nn.Module, base.backbone).half()
    elif condition == "eager":
        base.backbone.config._attn_implementation = "eager"
    elif condition == "rope":
        base.backbone.config.rope_parameters = {"rope_type": "dynamic", "factor": 2.0}
    else:
        base.backbone.config.layer_types = ["sliding_attention"]
        base.backbone.has_sliding_layers = True

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("Unsupported capability must fail before model prefill")

    monkeypatch.setattr(base.backbone, "forward", forbidden)
    with pytest.raises(ValueError):
        PrefixScorer(base, (1, 2))


def test_cached_shared_helper_rejects_non_cpu_before_tensor_allocation(
    base: WyatScorer, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = base.encode_request(request())
    retained = PrefixScorer(base, row.ids[:2])
    cache = retained._fork(1)
    base.device = torch.device("cuda")

    def forbidden(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("Rejected cached CUDA must allocate no tensor")

    monkeypatch.setattr(torch, "full", forbidden)
    with pytest.raises(ValueError, match="CPU"):
        base._score_encoded((row,), prefix_length=2, cache=cache)


@pytest.mark.parametrize("fault", ["constructor", "subclass", "alias", "discard"])
def test_incompatible_cache_semantics_reject_before_model_prefill(
    base: WyatScorer, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    from transformers.cache_utils import DynamicCache

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError(
            "Cache capability must be established before model prefill"
        )

    monkeypatch.setattr(base.backbone, "forward", forbidden)
    if fault == "constructor":

        def old_constructor() -> DynamicCache:
            return DynamicCache()

        monkeypatch.setattr(runtime, "new_cache", old_constructor)
    elif fault == "subclass":

        class AlternateCache(DynamicCache):
            pass

        def alternate_constructor(**kwargs: object) -> DynamicCache:
            return AlternateCache(config=base.backbone.config)

        monkeypatch.setattr(runtime, "new_cache", alternate_constructor)
    else:
        ordinary_update = DynamicLayer.update

        def incompatible_update(
            layer: DynamicLayer,
            keys: torch.Tensor,
            values: torch.Tensor,
            *args: object,
            **kwargs: object,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if fault == "alias":
                layer.keys, layer.values = keys, values
                layer.is_initialized = True
                return keys, values
            if layer.is_initialized:
                assert layer.keys is not None and layer.values is not None
                return (
                    layer.keys,
                    layer.values,
                )  # A cache that discards appended tokens.
            return ordinary_update(layer, keys, values)

        monkeypatch.setattr(DynamicLayer, "update", incompatible_update)
    with pytest.raises(ValueError, match="cache|Cache"):
        PrefixScorer(base, (1, 2))


def test_real_overlapping_candidate_readout_falls_back_to_full_prefill(
    base: WyatScorer, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = request()
    row = base.encode_request(value)
    retained = PrefixScorer(base, row.ids[: row.candidate_positions[0] + 1])
    expected = base.score_many((value,))
    ordinary = base.backbone.forward
    calls: list[bool] = []

    def observe(*args: object, **kwargs: object) -> BaseModelOutputWithPast:
        assert args == () and set(kwargs) == {"input_ids", "use_cache"}
        calls.append(kwargs["use_cache"] is False)
        return cast(
            BaseModelOutputWithPast,
            ordinary(
                input_ids=cast(torch.Tensor, kwargs["input_ids"]), use_cache=False
            ),
        )

    monkeypatch.setattr(base.backbone, "forward", observe)
    assert retained.score_many((value,)) == expected and calls == [True]


def test_head_hooks_run_on_each_readout_without_prefix_effects(
    base: WyatScorer,
) -> None:
    effects: list[str] = []

    def hook(module: nn.Module, args: tuple[object, ...], result: object) -> None:
        effects.append("readout")

    handle = base.head.register_forward_hook(hook)
    try:
        value = request()
        retained = PrefixScorer(base, base.encode_request(value).ids[:2])
        assert effects == []
        actual = retained.score_many((value,))
        expected = base.score_many((value,))
        torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected))
        assert effects == ["readout", "readout"]
    finally:
        handle.remove()
