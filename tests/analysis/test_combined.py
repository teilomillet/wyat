"""Combined cache/endpoint mechanics against maintained full Qwen3, CPU only."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

pytest.importorskip("mlx.core", exc_type=ImportError)
pytest.importorskip("mlx_lm", exc_type=ImportError)

import mlx.core as mx
from mlx_lm.models.qwen3 import ModelArgs, Qwen3Model

from benchmarks.analysis.mlx import combined
from wyat.backends.wyat import prefix as production_prefix
from wyat.backends.wyat.encoding import WyatEncoder
from wyat.backends.wyat.mlx import MLXCandidateHead, MLXWyatScorer
from wyat.backends.wyat.prefix import PrefixCache, PrefixScorer
from wyat.backends.wyat.prefix import PrefixScorer as PrefixEndpointScorer
from wyat.backends.wyat.readout import RotaryMap
from wyat.backends.wyat.readout import cached_endpoint_hidden as combined_hidden
from wyat.core import Candidate, Model, Request
from wyat.wire import decode_json


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
    with mx.stream(mx.cpu):
        mx.random.seed(43)
        backbone = Qwen3Model(
            ModelArgs("qwen3", 16, 3, 32, 2, 1e-6, 128, 1, 512, 1e6, 16, True)
        )
        head = MLXCandidateHead(16, 8)
        backbone.eval()
        head.eval()
    return MLXWyatScorer(
        backbone,
        head,
        WyatEncoder(CharacterTokenizer(), max_length=512, max_options=8),
        model_id="owned-tiny-asymmetric-combined-fixture",
        device=mx.Device(mx.cpu),
    )


@pytest.mark.parametrize("prefix_length", [1, 25])
def test_repeated_interleaved_scalar_batches_preserve_immutable_prefix(
    base: MLXWyatScorer, prefix_length: int
) -> None:
    values = [request(("yes", "no")), request(("a", "longer b", "c"))]
    scorer = PrefixEndpointScorer(
        base, base.encode_request(values[0]).ids[:prefix_length]
    )
    assert base.backbone.layers[0].self_attn.q_proj.weight.shape == (32, 16)
    before = [
        (cast(list[object], k.tolist()), cast(list[object], v.tolist()))
        for k, v in scorer._states
    ]
    for chunk in (values, values[:1], values[1:], values[::-1], values):
        for actual, expected in zip(
            scorer.score_many(chunk), base.score_many(chunk), strict=True
        ):
            assert actual == pytest.approx(expected, abs=3e-6, rel=1e-5)
    assert before == [
        (cast(list[object], k.tolist()), cast(list[object], v.tolist()))
        for k, v in scorer._states
    ]
    assert scorer.score_many([]) == []


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
def test_generated_candidate_counts_lengths_padding_and_caller_order(
    base: MLXWyatScorer, menus: list[list[str]]
) -> None:
    values = [request(menu) for menu in menus]
    scorer = PrefixEndpointScorer(base, base.encode_request(values[0]).ids[:25])
    for actual, expected in zip(
        scorer.score_many(values), base.score_many(values), strict=True
    ):
        assert len(actual) == len(expected)
        assert actual == pytest.approx(expected, abs=3e-6, rel=1e-5)


@given(
    positions=st.lists(st.integers(0, 10), min_size=1, max_size=6),
    prefix_length=st.integers(1, 5),
)
@settings(
    max_examples=12,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_absolute_cached_readouts_match_full_model_with_duplicate_positions(
    base: MLXWyatScorer, positions: list[int], prefix_length: int
) -> None:
    with mx.stream(mx.cpu):
        ids = mx.array([list(range(18)), list(range(18))], dtype=mx.int32)
        scorer = PrefixEndpointScorer(base, tuple(range(prefix_length)))
        suffix_positions = mx.array([positions, positions[::-1]], dtype=mx.int32)
        actual = combined_hidden(
            base.backbone, ids[:, prefix_length:], suffix_positions, scorer._fork(2)
        )
        whole = cast(mx.array, base.backbone(ids))
        expected = whole[mx.arange(2)[:, None], suffix_positions + prefix_length]
        mx.eval(actual, expected)
        assert bool(mx.all(mx.isfinite(actual)).item())
        assert bool(mx.allclose(actual, expected, atol=3e-6, rtol=1e-5).item())


@pytest.mark.parametrize("overlap", [False, True])
def test_mixed_mismatched_or_overlapping_batch_falls_back_intact(
    base: MLXWyatScorer, monkeypatch: pytest.MonkeyPatch, overlap: bool
) -> None:
    value = request(("a", "b"))
    encoded = base.encode_request(value)
    length = encoded.candidate_positions[0] + 1 if overlap else 25
    scorer = PrefixEndpointScorer(base, encoded.ids[:length])
    values = (
        [value]
        if overlap
        else [value, request(("a", "b"), context="different public state")]
    )

    def forbidden(
        ids: mx.array,
        endpoints: mx.array,
        positions: mx.array,
        caches: list[PrefixCache],
    ) -> mx.array:
        raise AssertionError("Fallback must not execute the optimized path")

    monkeypatch.setattr(scorer, "_logits", forbidden)
    assert scorer.score_many(values) == base.score_many(values)


def test_cache_growth_does_not_mutate_original_prefix(base: MLXWyatScorer) -> None:
    value = request(("long candidate " * 8, "short"))
    scorer = PrefixEndpointScorer(base, base.encode_request(value).ids[:25])
    assert len(base.encode_request(value).ids) - 25 > 256
    for _ in range(2):
        assert scorer.score_many([value])[0] == pytest.approx(
            base.score_many([value])[0], abs=3e-6, rel=1e-5
        )
        assert all(k.shape[2] == v.shape[2] == 25 for k, v in scorer._states)


def test_invalid_candidate_count_preserves_encoder_contract(base: MLXWyatScorer) -> None:
    valid = request(("a", "b"))
    scorer = PrefixEndpointScorer(base, base.encode_request(valid).ids[:25])
    with pytest.raises(ValueError, match="Wyat requires"):
        scorer.score_many([request(("a",))])


def test_original_prefix_rope_offset_negative_control(base: MLXWyatScorer) -> None:
    with mx.stream(mx.cpu):
        rotary = cast(RotaryMap, base.backbone.layers[-1].self_attn.rope)
        mx.random.seed(47)
        queries = mx.random.normal((1, 2, 20, 16))
        suffix_positions = mx.array([0, 9, 3, 3], dtype=mx.int32)
        original_positions = suffix_positions + 5
        selected = queries.transpose(0, 2, 1, 3)[:, original_positions]
        correct = rotary(selected.reshape(4, 2, 1, 16), offset=original_positions)
        wrong = rotary(selected.reshape(4, 2, 1, 16), offset=suffix_positions)
        whole = rotary(queries).transpose(0, 2, 1, 3)[:, original_positions]
        mx.eval(correct, wrong, whole)
        assert bool(mx.array_equal(correct.reshape(1, 4, 2, 16), whole).item())
        assert float(cast(float, mx.max(mx.abs(correct - wrong)).item())) > 0.1


def test_absolute_cached_mask_independent_uniform_gqa_oracle() -> None:
    with mx.stream(mx.cpu):
        positions = mx.array([0, 9, 3, 3], dtype=mx.int32)
        original_positions = positions + 5
        keys = mx.zeros((1, 1, 20, 16))
        values = mx.broadcast_to(
            mx.arange(20, dtype=mx.float32)[None, None, :, None], (1, 1, 20, 16)
        )
        queries = mx.zeros((1, 2, 4, 16))
        correct = mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=0.25,
            mask=mx.arange(20)[None, None, None, :]
            <= original_positions[None, None, :, None],
        )
        wrong = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=0.25, mask="causal"
        )
        relative = mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=0.25,
            mask=mx.arange(20)[None, None, None, :] <= positions[None, None, :, None],
        )
        expected = mx.broadcast_to(
            (original_positions.astype(mx.float32) / 2)[None, None, :, None],
            (1, 2, 4, 16),
        )
        mx.eval(correct, wrong, relative, expected)
        assert bool(mx.allclose(correct, expected, atol=1e-6, rtol=1e-6).item())
        assert float(cast(float, mx.max(mx.abs(wrong - expected)).item())) > 5
        assert float(cast(float, mx.max(mx.abs(relative - expected)).item())) > 2


def test_misaligned_cache_owners_rejected(base: MLXWyatScorer) -> None:
    scorer = PrefixEndpointScorer(base, (0, 1, 2))
    with mx.stream(mx.cpu):
        ids, positions = mx.array([[3, 4]]), mx.array([[1]])
        with pytest.raises(ValueError, match="per maintained layer"):
            combined_hidden(base.backbone, ids, positions, scorer._fork(1)[:-1])
        caches = scorer._fork(1)
        caches[0].offset += 1
        with pytest.raises(ValueError, match="same original prefix offset"):
            combined_hidden(base.backbone, ids, positions, caches)


@pytest.mark.parametrize(
    ("completed", "passed"),
    [(True, True), (True, False), (False, False), (False, True)],
)
def test_cli_uses_declared_strategy_and_rejects_failed_or_partial_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, completed: bool, passed: bool
) -> None:
    output = tmp_path / "combined.json"

    def fake_study(
        path: Path,
        *,
        batch_size: int,
        candidate_type: type[PrefixScorer],
        strategy: str,
    ) -> dict[str, object]:
        assert path == output and batch_size == 8
        assert candidate_type is PrefixEndpointScorer
        assert strategy == "prefix25-endpoints"
        return {
            "completed": completed,
            "passed": passed,
            "performance_gate_passed": False,
        }

    monkeypatch.setattr(combined, "study", fake_study)
    monkeypatch.setattr("sys.argv", ["combined", "--output", str(output)])
    if completed and passed:
        combined.main()  # A numerical pass with a negative speed result is complete.
    else:
        with pytest.raises(SystemExit) as error:
            combined.main()
        assert error.value.code == 1


@pytest.mark.parametrize(
    ("prefix", "error_type"),
    [
        ([1], TypeError),
        ("tokens", TypeError),
        (None, TypeError),
        ((), ValueError),
        ((-1,), ValueError),
        ((128,), ValueError),
        ((True,), ValueError),
        ((1.5,), ValueError),
        ((0,) * 512, ValueError),
    ],
)
def test_production_prefix_boundary_rejected_before_cache_construction(
    base: MLXWyatScorer,
    monkeypatch: pytest.MonkeyPatch,
    prefix: object,
    error_type: type[Exception],
) -> None:
    def forbidden() -> PrefixCache:
        raise AssertionError("Invalid prefix must fail before eager work")

    monkeypatch.setattr(production_prefix, "new_cache", forbidden)
    with pytest.raises(error_type):
        PrefixScorer(base, cast(tuple[int, ...], prefix))


def test_scaled_rope_and_compiled_base_rejected_before_cache_construction(
    base: MLXWyatScorer, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden() -> PrefixCache:
        raise AssertionError("Unsupported base must fail before eager work")

    monkeypatch.setattr(production_prefix, "new_cache", forbidden)
    compiled = MLXWyatScorer(
        base.backbone,
        base.head,
        base.encoder,
        model_id=base.model_id,
        device=base.device,
        compile=True,
    )
    assert compiled.compiled is True and base.compiled is False
    with pytest.raises(ValueError, match="eager, uncompiled"):
        PrefixScorer(compiled, (0, 1))
    with pytest.raises(AttributeError):
        setattr(compiled, "compiled", False)  # noqa: B010 - exercise a read-only boundary.
    base.backbone.args.rope_scaling = {"type": "linear", "factor": 2.0}
    with pytest.raises(ValueError, match="unscaled"):
        PrefixScorer(base, (0, 1))


def test_exact_prefix_identity_is_read_only(base: MLXWyatScorer) -> None:
    scorer = PrefixScorer(base, (0, 1))
    with pytest.raises(AttributeError):
        setattr(scorer, "prefix", (2, 3))  # noqa: B010 - exercise a read-only boundary.
    assert scorer.prefix == (0, 1)


def test_wrong_base_type_rejected() -> None:
    with pytest.raises(TypeError, match="native MLXWyatScorer"):
        PrefixScorer(cast(MLXWyatScorer, object()), (0, 1))


def test_constructor_rejects_non_boolean_compile(base: MLXWyatScorer) -> None:
    with pytest.raises(ValueError, match="boolean"):
        MLXWyatScorer(
            base.backbone,
            base.head,
            base.encoder,
            model_id=base.model_id,
            device=base.device,
            compile=cast(bool, 1),
        )


def test_production_prefix_supports_model_and_server_contracts(
    base: MLXWyatScorer,
) -> None:
    from fastapi.testclient import TestClient

    from wyat.server import create_app

    values = [request(("yes", "no")), request(("a", "b", "c"))]
    scorer = PrefixScorer(base, base.encode_request(values[0]).ids[:25])
    model = Model(scorer, scorer.model_id)
    reference = Model(base, base.model_id)
    actual, expected = model.decide_many(values), reference.decide_many(values)
    for left, right in zip(actual, expected, strict=True):
        assert left.selected == right.selected and left.ranking == right.ranking
        assert left.scores == pytest.approx(right.scores, abs=3e-6, rel=1e-5)
    with TestClient(create_app(model)) as client:
        for value in (values[0], request(("a", "b"), context="different state")):
            response = client.post(
                "/v1/decide",
                json={
                    "query": value.query,
                    "context": value.context,
                    "task": value.task,
                    "candidates": [
                        {"id": item.id, "text": item.text} for item in value.candidates
                    ],
                },
            )
            assert response.status_code == 200
            payload = cast(dict[str, object], decode_json(response.content))
            decision = reference.decide(value)
            assert payload["model_id"] == model.model_id
            assert payload["selected"] == decision.selected
            assert payload["ranking"] == list(decision.ranking)
            assert payload["scores"] == pytest.approx(
                decision.scores, abs=3e-6, rel=1e-5
            )
