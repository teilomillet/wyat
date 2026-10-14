"""Endpoint geometry against maintained whole-Qwen3 outputs, CPU-only mechanics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import NoReturn, cast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

import mlx.core as mx  # noqa: E402
from mlx.nn.layers.positional_encoding import RoPE  # noqa: E402
from mlx_lm.models.qwen3 import ModelArgs, Qwen3Model  # noqa: E402

from benchmarks.analysis.mlx import endpoints  # noqa: E402
from benchmarks.analysis.mlx.endpoints import (  # noqa: E402
    RotaryMap,
    bounded_cache,
    endpoint_hidden,
)
from wyat.backends.wyat.mlx import MLXCandidateHead  # noqa: E402
from wyat.wire import read_object  # noqa: E402


@given(
    first=st.lists(st.integers(0, 10), min_size=1, max_size=6),
    second=st.lists(st.integers(0, 10), min_size=1, max_size=6),
    padding=st.integers(1, 6),
)
@settings(max_examples=12, deadline=None)
def test_endpoint_full_model_oracle(
    first: list[int], second: list[int], padding: int
) -> None:
    with mx.stream(mx.cpu):
        mx.random.seed(19)
        model = Qwen3Model(
            ModelArgs("qwen3", 16, 2, 32, 2, 1e-6, 64, 1, 256, 1e6, 16, True)
        )
        head = MLXCandidateHead(16, 8)
        length, width = 12, max(len(first), len(second))
        ids = mx.array(
            [
                list(range(length)) + [31] * padding,
                list(range(length + padding)),
            ],
            dtype=mx.int32,
        )
        position_lists = [
            first + [0] * (width - len(first)) + [length - 1],
            second + [0] * (width - len(second)) + [length + padding - 1],
        ]
        positions = mx.array(position_lists, dtype=mx.int32)
        whole = cast(mx.array, model(ids))
        selected = endpoint_hidden(model, ids, positions)
        expected = whole[mx.arange(2)[:, None], positions]
        reference = head(expected[:, :-1], expected[:, -1])
        actual = head(selected[:, :-1], selected[:, -1])
        scalar = endpoint_hidden(model, ids[0:1, :length], positions[0:1])
        scalar_logits = head(scalar[:, :-1], scalar[:, -1])
        mx.eval(selected, expected, reference, actual, scalar_logits)
        assert bool(mx.all(mx.isfinite(selected)).item())
        assert bool(mx.allclose(selected, expected, atol=3e-6, rtol=1e-5).item())
        for index, count in enumerate((len(first), len(second))):
            assert cast(list[list[float]], actual.tolist())[index][
                :count
            ] == pytest.approx(
                cast(list[list[float]], reference.tolist())[index][:count],
                abs=3e-6,
                rel=1e-5,
            )
        assert cast(list[list[float]], actual.tolist())[0][
            : len(first)
        ] == pytest.approx(
            cast(list[list[float]], scalar_logits.tolist())[0][: len(first)],
            abs=3e-6,
            rel=1e-5,
        )


def test_native_array_rope_offsets_preserve_nonconsecutive_duplicate_positions() -> (
    None
):
    with mx.stream(mx.cpu):
        mx.random.seed(23)
        rotary = cast(RotaryMap, RoPE(16, base=1e6))
        queries = mx.random.normal((2, 2, 15, 16))
        positions = mx.array([[0, 9, 3, 3], [14, 4, 4, 1]], dtype=mx.int32)
        whole = rotary(queries)
        selected = queries.transpose(0, 2, 1, 3)[mx.arange(2)[:, None], positions]
        rotated = rotary(selected.reshape(8, 2, 1, 16), offset=positions.reshape(-1))
        rotated = rotated.reshape(2, 4, 2, 16)
        expected = whole.transpose(0, 2, 1, 3)[mx.arange(2)[:, None], positions]
        wrong_zero_offsets = rotary(selected.reshape(8, 2, 1, 16)).reshape(2, 4, 2, 16)
        mx.eval(rotated, expected, wrong_zero_offsets)
        assert bool(mx.array_equal(rotated, expected).item())
        assert (
            float(cast(float, mx.max(mx.abs(rotated - wrong_zero_offsets)).item()))
            > 0.1
        )
        assert bool(mx.array_equal(rotated[0, 2], rotated[0, 3]).item())


def test_lower_right_causal_mask_negative_control() -> None:
    with mx.stream(mx.cpu):
        # Uniform attention makes the independent oracle the mean key index.
        # Two query heads share one KV head, matching real GQA semantics.
        positions = mx.array([0, 9, 3, 3], dtype=mx.int32)
        keys = mx.zeros((1, 1, 15, 16))
        values = mx.broadcast_to(
            mx.arange(15, dtype=mx.float32)[None, None, :, None], (1, 1, 15, 16)
        )
        whole = mx.fast.scaled_dot_product_attention(
            mx.zeros((1, 2, 15, 16)), keys, values, scale=0.25, mask="causal"
        )
        explicit_mask = (
            mx.arange(15)[None, None, None, :] <= positions[None, None, :, None]
        )
        selected = mx.fast.scaled_dot_product_attention(
            mx.zeros((1, 2, 4, 16)), keys, values, scale=0.25, mask=explicit_mask
        )
        wrong = mx.fast.scaled_dot_product_attention(
            mx.zeros((1, 2, 4, 16)), keys, values, scale=0.25, mask="causal"
        )
        expected = mx.broadcast_to(
            (positions.astype(mx.float32) / 2)[None, None, :, None], (1, 2, 4, 16)
        )
        mx.eval(whole, selected, wrong, expected)
        assert bool(mx.allclose(selected, expected, atol=1e-6, rtol=1e-6).item())
        assert bool(
            mx.allclose(selected, whole[:, :, positions], atol=1e-6, rtol=1e-6).item()
        )
        assert float(cast(float, mx.max(mx.abs(wrong - expected)).item())) > 5


@pytest.mark.parametrize("failure", ["initial_clear", "synchronize", "body"])
def test_cache_cap_restored_when_experiment_fails(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    cap = [123]
    clears = [0]

    def set_limit(value: int) -> int:
        previous = cap[0]
        cap[0] = value
        return previous

    def clear() -> None:
        clears[0] += 1
        if failure == "initial_clear" and clears[0] == 1:
            raise RuntimeError("initial clear failed")

    def synchronize() -> None:
        if failure == "synchronize":
            raise RuntimeError("synchronize failed")

    monkeypatch.setattr(mx, "set_cache_limit", set_limit)
    monkeypatch.setattr(mx, "clear_cache", clear)
    monkeypatch.setattr(mx, "synchronize", synchronize)
    with pytest.raises(RuntimeError):
        with bounded_cache(456):
            assert cap[0] == 456
            if failure == "body":
                raise RuntimeError("scoring failed")
    assert cap[0] == 123


@pytest.mark.parametrize("completed,passed", [(True, False), (False, False)])
def test_failed_or_partial_quality_receipt_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, completed: bool, passed: bool
) -> None:
    output = tmp_path / "failed.json"

    def failed(path: Path, *, batch_size: int = 8) -> dict[str, object]:
        receipt: dict[str, object] = {
            "completed": completed,
            "passed": passed,
            "batch_size": batch_size,
        }
        path.write_text(json.dumps(receipt))
        return receipt

    monkeypatch.setattr(endpoints, "study", failed)
    monkeypatch.setattr("sys.argv", ["endpoint-study", "--output", str(output)])
    with pytest.raises(SystemExit) as error:
        endpoints.main()
    assert error.value.code == 1
    receipt = read_object(output)
    assert receipt["passed"] is False
    assert receipt["completed"] is completed


def test_model_load_failure_preserves_incomplete_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "load-failure.json"

    def set_limit(limit: int) -> int:
        return 123

    def clear() -> None:
        return None

    class FailedLoad:
        @classmethod
        def from_directory(cls, *args: object, **kwargs: object) -> FailedLoad:
            raise RuntimeError("model load failed")

    monkeypatch.setattr(endpoints, "MLXWyatScorer", FailedLoad)
    monkeypatch.setattr(mx, "set_cache_limit", set_limit)
    monkeypatch.setattr(mx, "clear_cache", clear)
    monkeypatch.setattr(mx, "synchronize", clear)
    with pytest.raises(RuntimeError, match="model load failed"):
        endpoints.study(output)
    receipt = read_object(output)
    assert receipt["passed"] is False and receipt["completed"] is False
    assert receipt["error"] == "RuntimeError: model load failed"
    assert isinstance(receipt["reference_sha256"], str)
    assert isinstance(receipt["sources_archive"], dict)


def test_setup_failure_invalidates_an_older_passed_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "old-success.json"
    output.write_text('{"completed": true, "passed": true}')

    def fail_setup(*args: object, **kwargs: object) -> NoReturn:
        raise RuntimeError("source freeze failed")

    monkeypatch.setattr(endpoints, "freeze_sources", fail_setup)
    with pytest.raises(RuntimeError, match="source freeze failed"):
        endpoints.study(output)
    receipt = read_object(output)
    assert receipt["passed"] is False and receipt["completed"] is False
    assert receipt["error"] == "RuntimeError: source freeze failed"
