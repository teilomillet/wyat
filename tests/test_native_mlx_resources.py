# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Responsive scheduling preserves math and bounds same-instance GPU work."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from os import environ
from threading import Event
from typing import cast

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
import mlx.core as mx  # noqa: E402
from mlx_lm.models.qwen3 import ModelArgs, Qwen3Model  # noqa: E402

from wyat import tasks  # noqa: E402
from wyat.backends.wyat.mlx import MLXCandidateHead  # noqa: E402
from wyat.loading import mlx as native  # noqa: E402


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return [1]


def _scorer(
    backbone: object,
    *,
    head: object | None = None,
    gpu: bool = True,
    responsive: bool = True,
) -> native.NativeMLXScorer:
    return native.NativeMLXScorer(
        cast(Qwen3Model, backbone),
        cast(MLXCandidateHead, head),
        _Tokenizer(),
        {"model_id": "resource-test", "max_batch_tokens": 64},
        device=mx.Device(mx.gpu if gpu else mx.cpu),
        max_length=64,
        responsive=responsive,
    )


class _Layer:
    def __init__(self, index: int, events: list[tuple[str, object]]) -> None:
        self.index = index
        self.events = events

    def __call__(self, hidden: mx.array, mask: mx.array, cache: object) -> mx.array:
        assert hidden == self.index - 1
        assert mask == "mask"
        assert cache is None
        self.events.append(("layer", self.index))
        return cast(mx.array, self.index)


class _Backbone:
    def __init__(self, count: int, events: list[tuple[str, object]]) -> None:
        self.events = events
        self.layers = [_Layer(index, events) for index in range(1, count + 1)]

    def embed_tokens(self, ids: mx.array) -> mx.array:
        assert ids == "ids"
        self.events.append(("embed", None))
        return cast(mx.array, 0)

    def norm(self, hidden: mx.array) -> mx.array:
        self.events.append(("norm", hidden))
        return hidden

    def __call__(self, ids: mx.array) -> mx.array:
        assert ids == "ids"
        self.events.append(("backbone", None))
        return cast(mx.array, "maintained-result")


@pytest.mark.parametrize("count", [3, 4, 7, 8, 9])
def test_responsive_forward_preserves_order_and_yields_only_between_chunks(
    count: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[tuple[str, object]] = []
    scorer = _scorer(_Backbone(count, events))

    def mask(hidden: mx.array, cache: object) -> mx.array:
        assert hidden == 0
        assert cache is None
        events.append(("mask", None))
        return cast(mx.array, "mask")

    def evaluate(hidden: mx.array) -> None:
        events.append(("eval", hidden))

    def pause(seconds: float) -> None:
        events.append(("sleep", seconds))

    monkeypatch.setattr(native, "create_attention_mask", mask)
    monkeypatch.setattr(mx, "eval", evaluate)
    monkeypatch.setattr(native, "sleep", pause)
    assert scorer._hidden(cast(mx.array, "ids")) == count
    expected: list[tuple[str, object]] = [("embed", None), ("mask", None)]
    for index in range(1, count + 1):
        expected.append(("layer", index))
        if index % 4 == 0 and index < count:
            expected.extend([("eval", index), ("sleep", 0.001)])
    expected.append(("norm", count))
    assert events == expected


@pytest.mark.parametrize(("gpu", "responsive"), [(False, True), (True, False)])
def test_cpu_and_unpaced_forward_use_maintained_backbone_without_sleep(
    gpu: bool, responsive: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[tuple[str, object]] = []
    scorer = _scorer(_Backbone(9, events), gpu=gpu, responsive=responsive)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("CPU/unpaced forward must not schedule chunks")

    monkeypatch.setattr(native, "sleep", forbidden)
    monkeypatch.setattr(native, "create_attention_mask", forbidden)
    monkeypatch.setattr(mx, "eval", forbidden)
    assert scorer._hidden(cast(mx.array, "ids")) == "maintained-result"
    assert events == [("backbone", None)]


def _row() -> list[tasks.EncodedQuestion]:
    question = tasks.Question("q", "choice", "read", (("a", "a"), ("b", "b")))
    return [tasks.EncodedQuestion(question, (1, 2, 3), (0, 1), 2)]


def _prepare_logits(
    scorer: native.NativeMLXScorer, monkeypatch: pytest.MonkeyPatch
) -> None:
    with mx.stream(mx.cpu):
        hidden = mx.zeros((1, 3, 2))
        mx.eval(hidden)
    monkeypatch.setattr(scorer, "_hidden", lambda ids: hidden)
    original_stream = mx.stream
    monkeypatch.setattr(mx, "stream", lambda device: original_stream(mx.cpu))


def test_inference_lock_covers_eval_and_host_result_and_releases_on_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scorer = _scorer(object(), gpu=False)
    _prepare_logits(scorer, monkeypatch)
    events: list[str] = []

    class Output:
        def tolist(self) -> list[list[float]]:
            assert scorer._inference_lock.locked()
            events.append("tolist")
            return [[1.0, 2.0]]

    def head(candidates: mx.array, query: mx.array) -> mx.array:
        assert scorer._inference_lock.locked()
        events.append("head")
        return cast(mx.array, Output())

    def evaluate(value: mx.array) -> None:
        assert scorer._inference_lock.locked()
        events.append("eval")

    monkeypatch.setattr(scorer, "head", head)
    monkeypatch.setattr(mx, "eval", evaluate)
    assert scorer._logits(_row()) == [[1.0, 2.0]]
    assert events == ["head", "eval", "tolist"]
    assert not scorer._inference_lock.locked()

    def failed(value: mx.array) -> None:
        raise RuntimeError("evaluation failed")

    monkeypatch.setattr(mx, "eval", failed)
    with pytest.raises(RuntimeError, match="evaluation failed"):
        scorer._logits(_row())
    assert not scorer._inference_lock.locked()
    monkeypatch.setattr(mx, "eval", evaluate)
    assert scorer._logits(_row()) == [[1.0, 2.0]]


def test_concurrent_same_instance_calls_are_serialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scorer = _scorer(object(), gpu=False)
    _prepare_logits(scorer, monkeypatch)
    entered, release, second_started, second_entered = (Event() for _ in range(4))
    with mx.stream(mx.cpu):
        result = mx.array([[1.0, 2.0]])
        mx.eval(result)
    calls = 0

    def head(candidates: mx.array, query: mx.array) -> mx.array:
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            assert release.wait(timeout=3)
        else:
            second_entered.set()
        return result

    def second() -> list[list[float]]:
        second_started.set()
        return scorer._logits(_row())

    monkeypatch.setattr(scorer, "head", head)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(scorer._logits, _row())
        try:
            assert entered.wait(timeout=2)
            second_future = pool.submit(second)
            assert second_started.wait(timeout=2)
            assert not second_entered.wait(timeout=0.05)
        finally:
            release.set()
        assert first_future.result(timeout=2) == [[1.0, 2.0]]
        assert second_future.result(timeout=2) == [[1.0, 2.0]]
    assert calls == 2
    assert second_entered.is_set()


def test_distinct_scorers_have_independent_locks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, second = (_scorer(object(), gpu=False) for _ in range(2))
    _prepare_logits(first, monkeypatch)
    _prepare_logits(second, monkeypatch)
    entered, release = Event(), Event()
    with mx.stream(mx.cpu):
        result = mx.array([[1.0, 2.0]])
        mx.eval(result)

    def blocked(candidates: mx.array, query: mx.array) -> mx.array:
        entered.set()
        assert release.wait(timeout=3)
        return result

    monkeypatch.setattr(first, "head", blocked)
    monkeypatch.setattr(second, "head", lambda candidates, query: result)
    assert first._inference_lock is not second._inference_lock
    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(first._logits, _row())
        try:
            assert entered.wait(timeout=2)
            assert pool.submit(second._logits, _row()).result(timeout=2) == [[1.0, 2.0]]
        finally:
            release.set()
        assert first_future.result(timeout=2) == [[1.0, 2.0]]


@pytest.mark.parametrize("gpu", [False, True])
def test_seven_layer_real_backbone_paced_logits_equal_unpaced(
    gpu: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    if gpu and not mx.metal.is_available():
        pytest.skip("Metal GPU is unavailable")
    device = mx.Device(mx.gpu if gpu else mx.cpu)
    with mx.stream(device):
        args = ModelArgs(
            model_type="qwen3",
            hidden_size=16,
            num_hidden_layers=7,
            intermediate_size=32,
            num_attention_heads=2,
            rms_norm_eps=1e-6,
            vocab_size=32,
            num_key_value_heads=1,
            max_position_embeddings=64,
            rope_theta=1000000.0,
            head_dim=8,
            tie_word_embeddings=True,
        )
        backbone = Qwen3Model(args)  # type: ignore[no-untyped-call, unused-ignore]
        head = MLXCandidateHead(16, 8)
        backbone.eval()
        head.eval()
        mx.eval(backbone.parameters(), head.parameters())  # type: ignore[no-untyped-call, unused-ignore]
    scorer = _scorer(backbone, head=head, gpu=gpu)
    thread_settings = {
        name: environ.get(name)
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "MLX_NUM_THREADS",
        )
    }

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("inference changed global resource configuration")

    for name in (
        "set_default_device",
        "set_default_stream",
        "set_memory_limit",
        "set_cache_limit",
        "set_wired_limit",
        "clear_cache",
        "reset_peak_memory",
    ):
        monkeypatch.setattr(mx, name, forbidden)
    unpaced = _scorer(backbone, head=head, gpu=gpu, responsive=False)
    assert scorer._logits(_row()) == unpaced._logits(_row())
    assert {name: environ.get(name) for name in thread_settings} == thread_settings


@pytest.mark.parametrize("responsive", [0, 1, "true", None])
def test_responsive_option_requires_boolean(responsive: object) -> None:
    with pytest.raises(ValueError, match="responsive must be a boolean"):
        _scorer(object(), responsive=cast(bool, responsive))
    with pytest.raises(ValueError, match="responsive must be a boolean"):
        native.load_native_mlx(
            "unused-artifact", device="cpu", responsive=cast(bool, responsive)
        )
