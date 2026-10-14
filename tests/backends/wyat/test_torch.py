# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Independent token/head oracles and real local-load failure exercises."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch
from fixtures.wyat import create_wyat_bundle
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from safetensors.torch import load_file, save_file
from torch.nn import functional as F

from wyat import Candidate, Request
from wyat.backends.wyat.torch import CandidateHead, WyatScorer, _config


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    return create_wyat_bundle(tmp_path / "wyat")


@pytest.fixture
def scorer(bundle: Path) -> WyatScorer:
    return WyatScorer.from_directory(bundle, max_length=512, token_cache_size=256)


def request(
    query: str = "read?",
    context: str = "Bonjour",
    choices: Sequence[str] = ("yes", "no"),
) -> Request:
    return Request(
        query=query,
        context=context,
        task="choice",
        candidates=tuple(
            Candidate(id=str(index), text=text) for index, text in enumerate(choices)
        ),
    )


def reference_encoding(
    scorer: WyatScorer, value: Request
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    prefix = (
        "Context:\n"
        + value.context
        + "\n\nTask type: choice\nQuestion:\n"
        + value.query
        + "\nOptions:"
    )
    pieces = [prefix]
    for candidate in value.candidates:
        pieces.append(
            "\n<option>\n"
            + json.dumps(
                {"key": candidate.id, "description": candidate.text},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n</option>"
        )
    pieces.append(
        "\n\nSelect the single option best supported by the context and instructions."
        "\nDecision:"
    )
    ids, endpoints = [], []
    for index, piece in enumerate(pieces):
        ids.extend(scorer.tokenizer.encode(piece, add_special_tokens=False))
        if 0 < index < len(pieces) - 1:
            endpoints.append(len(ids) - 1)
    return tuple(ids), tuple(endpoints), len(ids) - 1


def reference_head(
    head: CandidateHead, candidates: torch.Tensor, query: torch.Tensor
) -> torch.Tensor:
    """Independent functional expression of the pinned upstream formula."""
    state = head.state_dict()
    candidate = F.layer_norm(
        candidates.float(),
        (candidates.shape[-1],),
        state["candidate_norm.weight"],
        state["candidate_norm.bias"],
    )
    global_query = F.layer_norm(
        query.float(),
        (query.shape[-1],),
        state["query_norm.weight"],
        state["query_norm.bias"],
    )
    key = F.linear(candidate, state["key.weight"])
    projected_query = F.linear(global_query, state["query.weight"]).unsqueeze(1)
    interaction = torch.sum(key * projected_query, dim=-1) / math.sqrt(head.head_dim)
    candidate_term = F.linear(
        candidate, state["candidate_mlp.weight"], state["candidate_mlp.bias"]
    )
    global_term = F.linear(global_query, state["query_mlp.weight"]).unsqueeze(1)
    nonlinear = F.linear(
        F.gelu(candidate_term + global_term), state["scalar.weight"]
    ).squeeze(-1)
    return interaction + nonlinear


@given(rows=st.integers(1, 3), choices=st.integers(2, 8), seed=st.integers(0, 100))
@settings(max_examples=15, deadline=None)
def test_head_matches_independent_formula(rows: int, choices: int, seed: int) -> None:
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        head = CandidateHead(16, 8)
        candidates = torch.randn(rows, choices, 16)
        query = torch.randn(rows, 16)
    with torch.inference_mode():
        torch.testing.assert_close(
            head(candidates, query),
            reference_head(head, candidates, query),
            rtol=0,
            atol=0,
        )


def test_load_and_score_matches_independent_encoding_and_head(
    scorer: WyatScorer,
) -> None:
    values = [request(), request(query="世界?", choices=("write", "read", "é"))]
    encoded = [reference_encoding(scorer, value) for value in values]
    length = math.ceil(max(len(row[0]) for row in encoded) / 8) * 8
    pad_id = scorer.tokenizer.pad_token_id
    assert pad_id is not None
    ids = torch.full((2, length), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    endpoints = torch.zeros(2, 3, dtype=torch.long)
    for index, (tokens, positions, _) in enumerate(encoded):
        ids[index, : len(tokens)] = torch.tensor(tokens)
        mask[index, : len(tokens)] = 1
        endpoints[index, : len(positions)] = torch.tensor(positions)
    with torch.inference_mode():
        hidden = scorer.backbone(
            input_ids=ids, attention_mask=mask, use_cache=False
        ).last_hidden_state
        rows = torch.arange(2)
        logits = reference_head(
            scorer.head,
            hidden[rows[:, None], endpoints],
            hidden[rows, torch.tensor([row[2] for row in encoded])],
        )
    expected = [
        logits[index, : len(value.candidates)].tolist()
        for index, value in enumerate(values)
    ]
    for actual, reference in zip(scorer.score_many(values), expected, strict=True):
        torch.testing.assert_close(
            torch.tensor(actual), torch.tensor(reference), atol=1e-6, rtol=1e-5
        )
    assert scorer.score_many([]) == []


def test_token_cache_preserves_order_unicode_and_bounds(bundle: Path) -> None:
    cached = WyatScorer.from_directory(bundle, max_length=512, token_cache_size=2)
    uncached = WyatScorer.from_directory(bundle, max_length=512, token_cache_size=0)
    for texts in [("é", "世界", "<option>"), ("<option>", "世界", "é"), ("a", "b")]:
        value = request(choices=texts)
        assert cached.encode_request(value) == reference_encoding(cached, value)
        assert cached.encode_request(value) == uncached.encode_request(value)
        assert cached.score_many([value]) == uncached.score_many([value])
        assert cached._option_tokens.cache_info().currsize <= 2


def test_cache_does_not_hide_changed_prefix(scorer: WyatScorer) -> None:
    first = scorer.encode_request(request(context="read"))
    second = scorer.encode_request(request(context="write"))
    assert first.ids != second.ids
    assert scorer._option_tokens.cache_info().hits >= 2


def test_token_cache_is_opt_in(bundle: Path) -> None:
    scorer = WyatScorer.from_directory(bundle, max_length=512)
    scorer.encode_request(request())
    scorer.encode_request(request())
    assert scorer._option_tokens.cache_info().maxsize == 0
    assert scorer._option_tokens.cache_info().currsize == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("head_variant", "type-separated"),
        ("prompt_version", "other"),
        ("readout", "label_token"),
        ("checkpoint_format", "lora"),
        ("head_compute_dtype", "bfloat16"),
        ("max_options", True),
    ],
)
def test_rejects_unsupported_checkpoint_contract(
    bundle: Path, field: str, value: object
) -> None:
    path = bundle / "decision_config.json"
    metadata = json.loads(path.read_text())
    metadata[field] = value
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        WyatScorer.from_directory(bundle, max_length=512)


def test_rejects_missing_backbone_weight(bundle: Path) -> None:
    path = bundle / "backbone/model.safetensors"
    state = load_file(path)
    state.pop(next(iter(state)))
    save_file(state, path)
    with pytest.raises(ValueError, match="Backbone weights"):
        WyatScorer.from_directory(bundle, max_length=512)


@pytest.mark.parametrize(
    "name", ["backbone/model.safetensors", "decision_head.safetensors"]
)
def test_truncated_safetensors_body_uses_public_load_error(
    bundle: Path, name: str
) -> None:
    path = bundle / name
    # Keep the valid header so the tensor loader, not header validation, fails.
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="Malformed Wyat .* Safetensors weights"):
        WyatScorer.from_directory(bundle, max_length=512)


def test_model_identity_changes_with_weights(bundle: Path) -> None:
    first = WyatScorer.from_directory(bundle, max_length=512).model_id
    path = bundle / "decision_head.safetensors"
    state = load_file(path)
    state["scalar.weight"] += 0.01
    save_file(state, path)
    second = WyatScorer.from_directory(bundle, max_length=512).model_id
    assert first != second


@pytest.mark.parametrize("kind", ["int64", "bfloat16", "nan"])
def test_rejects_head_dtype_and_nonfinite(bundle: Path, kind: str) -> None:
    path = bundle / "decision_head.safetensors"
    state = load_file(path)
    if kind == "nan":
        state["scalar.weight"][0, 0] = float("nan")
    else:
        state["scalar.weight"] = state["scalar.weight"].to(getattr(torch, kind))
    save_file(state, path)
    with pytest.raises(ValueError, match="float32|Nonfinite"):
        WyatScorer.from_directory(bundle, max_length=512)


@pytest.mark.parametrize("kind", ["int64", "nan"])
def test_rejects_backbone_dtype_and_nonfinite(bundle: Path, kind: str) -> None:
    path = bundle / "backbone/model.safetensors"
    state = load_file(path)
    name = next(iter(state))
    if kind == "nan":
        state[name].view(-1)[0] = float("nan")
    else:
        state[name] = state[name].to(torch.int64)
    save_file(state, path)
    with pytest.raises(ValueError, match="dtype|Nonfinite"):
        WyatScorer.from_directory(bundle, max_length=512)


def test_rejects_ambiguous_metadata(bundle: Path) -> None:
    path = bundle / "decision_config.json"
    path.write_text('{"head_dim":8,"head_dim":9}')
    with pytest.raises(ValueError, match="duplicate"):
        WyatScorer.from_directory(bundle, max_length=512)


def test_manifest_changes_model_identity_and_cap(bundle: Path) -> None:
    before = WyatScorer.from_directory(bundle, max_length=512).model_id
    path = bundle / "MODEL_MANIFEST.json"
    path.write_text(json.dumps({"profile": "qwen-full", "max_input_tokens": 256}))
    after = WyatScorer.from_directory(bundle, max_length=256).model_id
    assert before != after
    with pytest.raises(ValueError, match="checkpoint limit"):
        WyatScorer.from_directory(bundle, max_length=512)


def test_declared_manifest_digest_detects_weight_drift(bundle: Path) -> None:
    path = bundle / "decision_head.safetensors"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "profile": "qwen-full",
        "max_input_tokens": 512,
        "files_sha256": {"decision_head.safetensors": digest},
    }
    (bundle / "MODEL_MANIFEST.json").write_text(json.dumps(manifest))
    WyatScorer.from_directory(bundle, max_length=512)
    state = load_file(path)
    state["scalar.weight"] += 0.01
    save_file(state, path)
    with pytest.raises(ValueError, match="declared digest"):
        WyatScorer.from_directory(bundle, max_length=512)


@pytest.mark.parametrize(
    "table",
    [
        None,
        [],
        {"decision_head.safetensors": "xyz"},
        {"decision_head.safetensors": "z" * 64},
    ],
)
def test_declared_manifest_hash_table_rejects_malformed_digest(
    bundle: Path, table: object
) -> None:
    manifest = {
        "profile": "qwen-full",
        "max_input_tokens": 512,
        "files_sha256": table,
    }
    (bundle / "MODEL_MANIFEST.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="64-hex"):
        WyatScorer.from_directory(bundle, max_length=512)


def test_no_truncation_or_nonchoice(scorer: WyatScorer, bundle: Path) -> None:
    with pytest.raises(ValueError, match="checkpoint limit"):
        WyatScorer.from_directory(bundle, max_length=513)
    with pytest.raises(ValueError, match="exceeds max_length"):
        scorer.encode_request(request(query="read " * 512))
    value = Request(
        "read?", (Candidate("a", "yes"), Candidate("b", "no")), task="score"
    )
    with pytest.raises(ValueError, match="choice"):
        scorer.score_many([value])


@given(
    descriptions=st.lists(
        st.text(alphabet='abc 世界é\n<>/"\\', min_size=1, max_size=25).filter(
            str.strip
        ),
        min_size=2,
        max_size=5,
    )
)
@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_segmented_token_contract_property(
    scorer: WyatScorer, descriptions: list[str]
) -> None:
    # The fixture has immutable weights; cache content cannot change token IDs.
    value = request(choices=descriptions)
    assert scorer.encode_request(value) == reference_encoding(scorer, value)


def test_release_rotary_config_preserves_theta() -> None:
    config = _config(
        {
            "model_type": "qwen3",
            "rope_parameters": {"rope_type": "default", "rope_theta": 1000000.0},
        }
    )
    theta = getattr(config, "rope_theta", None)
    if theta is None:
        rotary = config.rope_parameters
        assert isinstance(rotary, dict)
        theta = rotary["rope_theta"]
    assert theta == 1000000.0


def test_eager_and_sdpa_same_fixture_decisions(bundle: Path) -> None:
    sdpa = WyatScorer.from_directory(bundle, max_length=512, attention="sdpa")
    eager = WyatScorer.from_directory(bundle, max_length=512, attention="eager")
    values = [request(), request(query="write?", choices=("a", "b", "c"))]
    actual = torch.tensor(sdpa.score_many(values)[0])
    expected = torch.tensor(eager.score_many(values)[0])
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert actual.argmax().item() == expected.argmax().item()


@given(
    context_lengths=st.lists(st.integers(1, 12), min_size=2, max_size=4),
    padding_id=st.integers(0, 31),
)
@settings(
    max_examples=12,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_causal_right_padding_cannot_change_selected_endpoints(
    scorer: WyatScorer, context_lengths: list[int], padding_id: int
) -> None:
    """Independent challenge: arbitrary future pads cannot reach valid endpoints."""
    values = [
        request(context="read " * length, choices=("yes", "no", "maybe")[: 2 + i % 2])
        for i, length in enumerate(context_lengths)
    ]
    encoded = [reference_encoding(scorer, value) for value in values]
    length = max(len(row[0]) for row in encoded) + 5
    ids = torch.full((len(values), length), padding_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for index, (tokens, _, _) in enumerate(encoded):
        ids[index, : len(tokens)] = torch.tensor(tokens)
        mask[index, : len(tokens)] = 1
    with torch.inference_mode():
        causal = scorer.backbone(input_ids=ids, use_cache=False).last_hidden_state
        masked = scorer.backbone(
            input_ids=ids, attention_mask=mask, use_cache=False
        ).last_hidden_state
    actual = scorer.score_many(values)
    for index, (_, endpoints, query_position) in enumerate(encoded):
        positions = torch.tensor((*endpoints, query_position))
        torch.testing.assert_close(
            causal[index, positions], masked[index, positions], atol=1e-6, rtol=1e-5
        )
        singleton = scorer.score_many([values[index]])[0]
        torch.testing.assert_close(
            torch.tensor(actual[index]), torch.tensor(singleton), atol=1e-6, rtol=1e-5
        )
        assert max(range(len(singleton)), key=singleton.__getitem__) == max(
            range(len(actual[index])), key=actual[index].__getitem__
        )


def test_exact_candidate_ties_survive_ragged_batching(scorer: WyatScorer) -> None:
    with torch.no_grad():
        scorer.head.key.weight.zero_()
        scorer.head.scalar.weight.zero_()
    values = [request(), request(context="read " * 20, choices=("a", "b", "c"))]
    assert scorer.score_many(values) == [[0.0, 0.0], [0.0, 0.0, 0.0]]
    assert scorer.score_many(list(reversed(values))) == [
        [0.0, 0.0, 0.0],
        [0.0, 0.0],
    ]


# Exact CandidateHead parameter layout and logits for fixed weights and inputs.
# A failure means the readout changed. If intended, bump
# wyat.loading.native_artifact.PREDICTOR and record the new entry beside the
# old one; never rewrite an existing entry.
HEAD_GOLDEN = {
    # Recorded from CandidateHead source sha256 dca28885, the v7 training source.
    "wyat-bilinear-gelu-candidate-head-v1": (
        [
            ("candidate_mlp.bias", (4,)),
            ("candidate_mlp.weight", (4, 8)),
            ("candidate_norm.bias", (8,)),
            ("candidate_norm.weight", (8,)),
            ("key.weight", (4, 8)),
            ("query.weight", (4, 8)),
            ("query_mlp.weight", (4, 8)),
            ("query_norm.bias", (8,)),
            ("query_norm.weight", (8,)),
            ("scalar.weight", (1, 4)),
        ],
        [[-0.22214121, 1.3587637, 2.0296602], [15.898141, 2.3742051, 0.2675159]],
    ),
}


def test_head_matches_the_predictor_golden() -> None:
    from wyat.loading.native_artifact import PREDICTOR

    assert PREDICTOR in HEAD_GOLDEN, "record the golden values of a new predictor"
    layout, expected = HEAD_GOLDEN[PREDICTOR]
    head = CandidateHead(8, 4).eval()
    state = sorted(head.state_dict().items())
    assert [(name, tuple(value.shape)) for name, value in state] == layout
    with torch.no_grad():
        for index, (_, value) in enumerate(state):
            steps = torch.arange(value.numel()) * 0.37 + index
            value.copy_(torch.sin(steps).reshape(value.shape))
        candidates = torch.cos(torch.arange(48) * 0.11).reshape(2, 3, 8) * 3
        query = torch.sin(torch.arange(16) * 0.23).reshape(2, 8) * 2
        logits = head(candidates, query)
    assert torch.allclose(logits, torch.tensor(expected), rtol=1e-5, atol=1e-5), (
        "candidate head readout changed: restore it, or bump PREDICTOR"
    )
