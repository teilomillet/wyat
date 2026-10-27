# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Resource admission is exercised without allocating a model or using a GPU."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from wyat.core import Model, Request
from wyat.loading import local
from wyat.resources import ResourceGuard, ResourceLimitError, current_guard


class _Scorer:
    model_id = "resource-fixture"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def score_many(self, requests: Sequence[Request]) -> Sequence[Sequence[float]]:
        self.calls.append("score_many")
        return [(1.0, 2.0) for _ in requests]

    def logits(self, record: object) -> object:
        self.calls.append("logits")
        return record

    def predict(self, record: object) -> object:
        self.calls.append("predict")
        return record

    def _records_logits(self, record: object) -> object:
        self.calls.append("_records_logits")
        return record


@pytest.fixture
def policy(monkeypatch: pytest.MonkeyPatch) -> list[ResourceGuard]:
    guards: list[ResourceGuard] = []
    real_guard = ResourceGuard

    def make(max_fraction: float) -> ResourceGuard:
        guard = real_guard(max_fraction)
        guard.enabled = True
        guards.append(guard)
        return guard

    monkeypatch.setattr(local, "ResourceGuard", make)
    monkeypatch.setattr(ResourceGuard, "check", lambda *args, **kwargs: None)
    return guards


def _fake_load(monkeypatch: pytest.MonkeyPatch, scorer: _Scorer) -> None:
    def load(path: str | Path, **kwargs: object) -> Model:
        assert current_guard() is not None
        return Model(scorer, scorer.model_id)

    monkeypatch.setattr(local, "_load", load)


@pytest.mark.parametrize("fraction", [0.90, 0.7, 1.0])
def test_load_retains_default_or_custom_policy_across_inference(
    fraction: float,
    policy: list[ResourceGuard],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scorer = _Scorer()
    _fake_load(monkeypatch, scorer)
    kwargs: dict[str, Any] = (
        {} if fraction == 0.90 else {"max_resource_fraction": fraction}
    )
    model = local.load(tmp_path, **kwargs)
    assert policy[0].max_fraction == fraction
    assert isinstance(model._scorer, local._GuardedScorer)
    assert model._scorer.model_id == scorer.model_id
    assert current_guard() is None

    seen: list[ResourceGuard | None] = []

    def check(*args: object, **kwargs: object) -> None:
        seen.append(current_guard())

    monkeypatch.setattr(policy[0], "check", check)
    for method in ("score_many", "logits", "predict", "_records_logits"):
        getattr(model._scorer, method)([])
    assert seen == [policy[0]] * 4
    assert scorer.calls == ["score_many", "logits", "predict", "_records_logits"]
    assert current_guard() is None


@pytest.mark.parametrize("directory", ["", "base", "backbone"])
def test_projected_loading_budget_is_checked_before_loader(
    directory: str,
    policy: list[ResourceGuard],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    weights = tmp_path / directory
    weights.mkdir(exist_ok=True)
    (weights / "weights.safetensors").write_bytes(b"x" * 40)
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "weights.bin").write_bytes(b"x" * 900)
    calls: list[int] = []

    def check(self: ResourceGuard, additional_bytes: int = 0, **kwargs: object) -> None:
        calls.append(additional_bytes)
        if additional_bytes:
            raise ResourceLimitError("insufficient headroom")

    monkeypatch.setattr(ResourceGuard, "check", check)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("loader ran after admission refusal")

    monkeypatch.setattr(local, "_load", forbidden)
    with pytest.raises(ResourceLimitError, match="headroom"):
        local.load(tmp_path)
    assert calls == [0, 120]
    assert current_guard() is None


def test_inference_refusal_does_not_call_scorer(
    policy: list[ResourceGuard],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scorer = _Scorer()
    _fake_load(monkeypatch, scorer)
    model = local.load(tmp_path)

    def refused(*args: object, **kwargs: object) -> None:
        raise ResourceLimitError("other applications need memory")

    monkeypatch.setattr(policy[0], "check", refused)
    request = Request.choice("Choose", {"a": "A", "b": "B"})
    with pytest.raises(ResourceLimitError):
        model.decide_many((request,), errors="return")
    assert scorer.calls == []
    assert current_guard() is None


def test_non_macos_returns_original_scorer_without_probing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    scorer = _Scorer()
    _fake_load(monkeypatch, scorer)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("non-macOS telemetry was used")

    monkeypatch.setattr(ResourceGuard, "_snapshot", forbidden)
    model = local.load(tmp_path)
    assert model._scorer is scorer


def test_runtime_configuration_uses_active_guard_before_allocation(
    policy: list[ResourceGuard], monkeypatch: pytest.MonkeyPatch
) -> None:
    guard = ResourceGuard(0.85)
    guard.enabled = True
    runtime = SimpleNamespace()
    calls: list[tuple[object, object]] = []
    monkeypatch.setitem(sys.modules, "torch", runtime)
    monkeypatch.setattr(
        guard, "configure_torch", lambda module, device: calls.append((module, device))
    )
    with guard.activate():
        local._configure_current_runtime("torch", None)
        local._configure_current_runtime("torch", "mps")
    assert calls == [(runtime, "cpu"), (runtime, "mps")]


def test_missing_optional_methods_stay_absent(policy: list[ResourceGuard]) -> None:
    class Minimal:
        def score_many(self, requests: Sequence[Request]) -> Sequence[Sequence[float]]:
            return []

    scorer = local._GuardedScorer(Minimal(), ResourceGuard(0.85))
    assert not hasattr(scorer, "predict")
