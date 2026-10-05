"""Independent tiny causal/parity and cache-ownership challenges; no real model."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from torch import nn
from transformers import Qwen3Model
from transformers.cache_utils import DynamicLayer
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast

from benchmarks.analysis.torch.prefix.cache import PrefixScorer
from benchmarks.analysis.torch.prefix.study import study
from benchmarks.fixtures.kai import create_kai_bundle
from wyat import Candidate, Request
from wyat.backends.kai.encoding import EncodedRequest
from wyat.backends.kai.torch import KaiScorer
from wyat.wire import read_object


@pytest.fixture
def base(tmp_path: Path) -> KaiScorer:
    return KaiScorer.from_directory(create_kai_bundle(tmp_path / "kai"), max_length=512)


def request(query: str = "read?", count: int = 2) -> Request:
    return Request(
        query,
        tuple(Candidate(str(index), "read " * (index + 1)) for index in range(count)),
        context="Bonjour",
    )


def hidden(base: KaiScorer, ids: torch.Tensor, candidate: PrefixScorer) -> torch.Tensor:
    with torch.inference_mode():
        output = cast(
            BaseModelOutputWithPast,
            base.backbone(
                input_ids=ids,
                past_key_values=candidate.fork(ids.shape[0]),
                use_cache=True,
            ),
        )
        assert output.last_hidden_state is not None
        return output.last_hidden_state


@settings(
    max_examples=10,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    prefix_length=st.integers(1, 7),
    suffix_length=st.integers(2, 12),
    batch=st.integers(1, 3),
)
def test_cached_suffix_matches_independent_full_model_and_future_causality(
    base: KaiScorer, prefix_length: int, suffix_length: int, batch: int
) -> None:
    vocab = base.backbone.config.vocab_size
    prefix = tuple((index * 3 + 1) % vocab for index in range(prefix_length))
    candidate = PrefixScorer(base, prefix)
    suffix = (
        torch.arange(batch * suffix_length).reshape(batch, suffix_length) + 2
    ) % vocab
    full = torch.cat((torch.tensor([prefix]).expand(batch, -1), suffix), dim=1)
    with torch.inference_mode():
        expected = base.backbone(input_ids=full, use_cache=False).last_hidden_state[
            :, prefix_length:
        ]
        actual = hidden(base, suffix, candidate)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    perturbed = suffix.clone()
    perturbed[:, -1] = (perturbed[:, -1] + 1) % vocab
    changed = hidden(base, perturbed, candidate)
    # An independent future-token intervention cannot affect earlier readouts.
    torch.testing.assert_close(changed[:, :-1], actual[:, :-1], atol=0, rtol=0)


def test_fresh_owners_and_interleaved_calls_preserve_exact_prefix(
    base: KaiScorer,
) -> None:
    candidate = PrefixScorer(base, (1, 2, 3))
    snapshots = tuple((key.clone(), value.clone()) for key, value in candidate._states)
    first, second = candidate.fork(2), candidate.fork(1)
    for layer, other, original in zip(
        first.layers, second.layers, candidate._states, strict=True
    ):
        assert isinstance(layer, DynamicLayer) and isinstance(other, DynamicLayer)
        assert layer.keys is not None and layer.values is not None
        assert other.keys is not None
        assert layer.keys.data_ptr() != other.keys.data_ptr()
        assert layer.keys.data_ptr() != original[0].data_ptr()
        assert layer.keys.stride(-1) == layer.values.stride(-1) == 1
    ids = torch.tensor([[4, 5, 6], [7, 8, 9]])
    a = hidden(base, ids, candidate)
    hidden(base, torch.tensor([[10, 11]]), candidate)
    again = hidden(base, ids, candidate)
    torch.testing.assert_close(a, again, atol=0, rtol=0)
    for actual, expected in zip(candidate._states, snapshots, strict=True):
        assert torch.equal(actual[0], expected[0]) and torch.equal(
            actual[1], expected[1]
        )
    assert first.get_seq_length() == second.get_seq_length() == 3


def test_multiple_layers_reuse_state_without_baseline_changes(base: KaiScorer) -> None:
    config = copy.deepcopy(base.backbone.config)
    config.num_hidden_layers = 2
    config.layer_types = ["full_attention", "full_attention"]
    with torch.random.fork_rng():
        torch.manual_seed(19)
        backbone = Qwen3Model(config)
    cast(nn.Module, backbone).eval()
    scorer = KaiScorer(
        backbone,
        base.head,
        base.tokenizer,
        device=base.device,
        max_length=base.max_length,
        max_options=base.max_options,
        token_cache_size=0,
        model_id="tiny-two-layer",
    )
    prefix = (1, 2, 3)
    candidate = PrefixScorer(scorer, prefix)
    assert len(candidate._states) == 2
    suffix = torch.tensor([[4, 5], [5, 4]])
    with torch.inference_mode():
        expected = backbone(
            torch.cat((torch.tensor([prefix]).expand(2, -1), suffix), 1),
            use_cache=False,
        ).last_hidden_state[:, 3:]
    torch.testing.assert_close(
        hidden(scorer, suffix, candidate), expected, atol=1e-6, rtol=1e-5
    )


def test_maintained_mask_is_exact_absolute_offset_oracle(base: KaiScorer) -> None:
    candidate = PrefixScorer(base, (1, 2, 3, 4, 5))
    cache = candidate.fork(2)
    with torch.inference_mode():
        mask = create_causal_mask(
            config=base.backbone.config,
            inputs_embeds=torch.zeros(2, 6, base.backbone.config.hidden_size),
            attention_mask=None,
            past_key_values=cache,
        )
    assert isinstance(mask, torch.Tensor) and mask.dtype == torch.bool
    expected = torch.tensor(
        [[key <= 5 + query for key in range(11)] for query in range(6)]
    )
    assert torch.equal(mask, expected[None, None].expand(2, 1, -1, -1))
    assert mask[0, 0, 0, :6].all().item() and not mask[0, 0, 0, 6:].any().item()


def test_ragged_scorer_parity_order_and_baseline_ownership(base: KaiScorer) -> None:
    requests = (request(), request("write? " * 4, 3))
    prefix = base.encode_request(requests[0]).ids[:2]
    candidate = PrefixScorer(base, prefix)
    before = {key: value.clone() for key, value in base.backbone.state_dict().items()}
    config_cache = base.backbone.config.use_cache
    for batch in (requests, requests[::-1], requests[:1], requests):
        actual, expected = candidate.score_many(batch), base.score_many(batch)
        for a, e in zip(actual, expected, strict=True):
            torch.testing.assert_close(
                torch.tensor(a), torch.tensor(e), atol=1e-6, rtol=1e-5
            )
    assert candidate.score_many(()) == []
    assert base.backbone.config.use_cache == config_cache
    assert all(
        torch.equal(value, before[key])
        for key, value in base.backbone.state_dict().items()
    )


@pytest.mark.parametrize("overlap", ["mismatch", "candidate", "query", "empty"])
def test_one_nonadmitted_row_falls_back_whole_batch(
    base: KaiScorer, monkeypatch: pytest.MonkeyPatch, overlap: str
) -> None:
    requests = (request(), request("write?"))
    row = base.encode_request(requests[0])
    candidate = PrefixScorer(base, row.ids[:2])
    invalid = {
        "mismatch": EncodedRequest(
            (row.ids[0] + 1, *row.ids[1:]), row.candidate_positions, row.query_position
        ),
        "candidate": EncodedRequest(
            row.ids, (1, row.candidate_positions[1]), row.query_position
        ),
        "query": EncodedRequest(row.ids, row.candidate_positions, 1),
        "empty": EncodedRequest(candidate.prefix, (0, 1), 1),
    }[overlap]

    def encode(value: Request) -> EncodedRequest:
        return row if value is requests[0] else invalid

    calls: list[tuple[Request, ...]] = []

    def fallback(values: Sequence[Request]) -> list[list[float]]:
        calls.append(tuple(values))
        return [[1.0, 2.0], [3.0, 4.0]]

    monkeypatch.setattr(base, "encode_request", encode)
    monkeypatch.setattr(base, "score_many", fallback)
    assert candidate.score_many(requests) == [[1.0, 2.0], [3.0, 4.0]]
    assert calls == [requests]


def test_original_admission_is_not_bypassed_by_prefix(base: KaiScorer) -> None:
    candidate = PrefixScorer(base, base.encode_request(request()).ids[:2])
    with pytest.raises(ValueError, match="exceeds"):
        candidate.score_many((request(), request("read " * 600)))
    with pytest.raises(ValueError, match="choice"):
        candidate.score_many((Request("read?", request().candidates, task="score"),))


@pytest.mark.parametrize(
    "prefix", [[], (), (True,), (-1,), (10**8,), (1.0,), (1,) * 512]
)
def test_invalid_tuple_prefix_rejected(base: KaiScorer, prefix: object) -> None:
    with pytest.raises((TypeError, ValueError), match="Prefix"):
        PrefixScorer(base, cast(tuple[int, ...], prefix))


def test_nondefault_rope_rejected_before_prefill(base: KaiScorer) -> None:
    base.backbone.config.rope_parameters = {
        "rope_type": "dynamic",
        "factor": 2.0,
        "rope_theta": 10000.0,
    }
    with pytest.raises(ValueError, match="default unscaled RoPE"):
        PrefixScorer(base, (1, 2))


@pytest.mark.parametrize("condition", ["train", "half", "eager"])
def test_invalid_runtime_rejected_before_prefill(
    base: KaiScorer, condition: str
) -> None:
    if condition == "train":
        cast(nn.Module, base.backbone).train()
    elif condition == "half":
        cast(nn.Module, base.backbone).half()
    else:
        base.backbone.config._attn_implementation = "eager"
    with pytest.raises(ValueError, match="eval FP32 CPU SDPA"):
        PrefixScorer(base, (1, 2))


def test_native_flags_failure_retained_before_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from transformers.integrations import sdpa_attention

    monkeypatch.setattr(sdpa_attention, "_is_torch_mps_available", False)

    def forbidden_load(*args: object, **kwargs: object) -> None:
        raise AssertionError("No model load before native flags")

    monkeypatch.setattr(KaiScorer, "from_directory", forbidden_load)
    prior = torch.get_num_threads()
    output = tmp_path / "screen.json"
    with pytest.raises(ValueError, match="native HF"):
        study(tmp_path / "model", output)
    receipt = read_object(output)
    assert (
        receipt["completed"] is receipt["screen_passed"] is receipt["accepted"] is False
    )
    assert receipt["phase"] == "failed" and torch.get_num_threads() == prior


def test_loading_failure_retains_flags_and_restores_threads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from transformers.integrations import sdpa_attention

    monkeypatch.setattr(sdpa_attention, "_is_torch_mps_available", True)
    monkeypatch.setattr(sdpa_attention, "_is_torch_greater_or_equal_than_2_13", True)

    def failed_load(*args: object, **kwargs: object) -> None:
        raise ValueError("deliberate artifact failure")

    monkeypatch.setattr(KaiScorer, "from_directory", failed_load)
    prior = torch.get_num_threads()
    output = tmp_path / "screen.json"
    with pytest.raises(ValueError, match="deliberate artifact failure"):
        study(tmp_path / "model", output)
    receipt = read_object(output)
    assert (
        receipt["completed"]
        is receipt["quality_passed"]
        is receipt["accepted"]
        is False
    )
    assert receipt["phase"] == "failed" and torch.get_num_threads() == prior
    assert receipt["sdpa_flags"] is not None and receipt["source_archive"] is not None
