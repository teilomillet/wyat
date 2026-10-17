# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import builtins
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Literal

import pytest

import wyat


def fail_backend_import(
    monkeypatch: pytest.MonkeyPatch,
    backend: Literal["torch", "mlx"],
    error: ImportError,
) -> None:
    original_import = builtins.__import__

    def import_with_failure(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> ModuleType:
        if name == f"wyat.backends.wyat.{backend}":
            raise error
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_with_failure)


@pytest.fixture
def loaded(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    class LocalScorer:
        model_id = "local-fixture"

        @classmethod
        def from_directory(cls, path: str | Path, **kwargs: object) -> LocalScorer:
            calls.append({"path": path, **kwargs})
            return cls()

        def score_many(
            self, requests: Sequence[wyat.Request]
        ) -> Sequence[Sequence[float]]:
            return [(1.0, 2.0) for _ in requests]

    for module_name, scorer_name in (
        ("wyat.backends.wyat.torch", "WyatScorer"),
        ("wyat.backends.wyat.mlx", "MLXWyatScorer"),
    ):
        module = ModuleType(module_name)
        setattr(module, scorer_name, LocalScorer)
        monkeypatch.setitem(sys.modules, module_name, module)
    return calls


def test_torch_default_preserves_cpu(loaded: list[dict[str, object]]) -> None:
    model = wyat.load("local")
    assert model.model_id == "local-fixture"
    assert loaded[0]["device"] == "cpu"
    assert loaded[0]["attention"] == "sdpa"


def test_mlx_explicit_selection_and_compile(loaded: list[dict[str, object]]) -> None:
    wyat.load("local", backend="mlx", compile=True, token_cache_size=8)
    assert loaded[0]["device"] == "gpu"
    assert loaded[0]["compile"] is True
    assert loaded[0]["token_cache_size"] == 8


def test_device_is_preserved(loaded: list[dict[str, object]]) -> None:
    wyat.load("local", backend="mlx", device="cpu")
    assert loaded[0]["device"] == "cpu"


def test_unsupported_compile_fails_before_load(loaded: list[dict[str, object]]) -> None:
    with pytest.raises(ValueError, match="only by the MLX"):
        wyat.load("local", compile=True)
    assert loaded == []


def test_unsupported_attention_fails_before_load(
    loaded: list[dict[str, object]],
) -> None:
    with pytest.raises(ValueError, match="native attention"):
        wyat.load("local", backend="mlx", attention="eager")
    assert loaded == []


@pytest.mark.parametrize("backend", ["torch", "mlx"])
def test_unpaced_inference_fails_before_load_without_native_mlx(
    backend: Literal["torch", "mlx"], loaded: list[dict[str, object]]
) -> None:
    with pytest.raises(ValueError, match="only to native MLX"):
        wyat.load("local", backend=backend, responsive=False)
    for bad in (0, 1, "false", None):
        with pytest.raises(ValueError, match="responsive must be a boolean"):
            wyat.load("local", backend=backend, responsive=bad)  # type: ignore[arg-type]
    assert loaded == []


@pytest.mark.parametrize("backend", ["torch", "mlx"])
def test_missing_primary_runtime_explains_optional_extra(
    monkeypatch: pytest.MonkeyPatch, backend: Literal["torch", "mlx"]
) -> None:
    original = ModuleNotFoundError(f"No module named '{backend}'", name=backend)
    fail_backend_import(monkeypatch, backend, original)
    with pytest.raises(ModuleNotFoundError) as error:
        wyat.load("local", backend=backend)
    extra = "inference" if backend == "torch" else "mlx"
    message = str(error.value)
    application_command = f"uv add /absolute/path/to/wyat --extra {extra}"
    checkout_command = f"uv sync --extra {extra}"
    assert application_command in message
    assert checkout_command in message
    assert message.index(application_command) < message.index(checkout_command)
    assert "your application's project" in message
    assert "From the Wyat checkout" in message
    assert error.value.name == backend
    assert error.value.__cause__ is original
    if backend == "mlx":
        assert "Apple Silicon macOS" in message


@pytest.mark.parametrize(
    "backend,missing",
    [
        ("torch", "transformers"),
        ("torch", "torch._C"),
        ("mlx", "mlx_lm"),
        ("mlx", "mlx.core"),
        ("mlx", "torch"),
    ],
)
def test_nested_missing_dependency_preserves_original_error(
    monkeypatch: pytest.MonkeyPatch,
    backend: Literal["torch", "mlx"],
    missing: str,
) -> None:
    original = ModuleNotFoundError(f"No module named '{missing}'", name=missing)
    fail_backend_import(monkeypatch, backend, original)
    with pytest.raises(ModuleNotFoundError) as error:
        wyat.load("local", backend=backend)
    assert error.value is original


@pytest.mark.parametrize("backend", ["torch", "mlx"])
def test_runtime_abi_import_error_is_not_rewritten(
    monkeypatch: pytest.MonkeyPatch, backend: Literal["torch", "mlx"]
) -> None:
    original = ImportError("incompatible native symbol")
    fail_backend_import(monkeypatch, backend, original)
    with pytest.raises(ImportError) as error:
        wyat.load("local", backend=backend)
    assert error.value is original


def test_load_sets_model_temperature(loaded: list[dict[str, object]]) -> None:
    model_default = wyat.load("local")
    assert model_default.temperature == 1.0

    model_custom = wyat.load("local", temperature=1.75)
    assert model_custom.temperature == 1.75

    with pytest.raises(
        ValueError, match="temperature must be a positive finite number"
    ):
        wyat.load("local", temperature=-0.5)


def test_load_autodetects_native_artifact_for_torch_and_mlx(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    native_dir = tmp_path / "native-bundle"
    native_dir.mkdir()
    (native_dir / "native_manifest.json").write_text("{}")
    calls: list[dict[str, object]] = []

    class FakeNativeScorer:
        model_id = "native-fixture"

        def score_many(
            self, requests: Sequence[wyat.Request]
        ) -> Sequence[Sequence[float]]:
            return [(1.0, 2.0) for _ in requests]

    def fake_load_native(path: str | Path, **kwargs: object) -> FakeNativeScorer:
        calls.append({"loader": "torch", "path": path, **kwargs})
        return FakeNativeScorer()

    def fake_load_native_mlx(path: str | Path, **kwargs: object) -> FakeNativeScorer:
        calls.append({"loader": "mlx", "path": path, **kwargs})
        return FakeNativeScorer()

    native_mod = ModuleType("wyat.loading.native")
    monkeypatch.setattr(native_mod, "load_native", fake_load_native, raising=False)
    mlx_mod = ModuleType("wyat.loading.mlx")
    monkeypatch.setattr(mlx_mod, "load_native_mlx", fake_load_native_mlx, raising=False)
    monkeypatch.setitem(sys.modules, "wyat.loading.native", native_mod)
    monkeypatch.setitem(sys.modules, "wyat.loading.mlx", mlx_mod)

    model_torch = wyat.load(native_dir, prefix_cache_size=32, max_length=256)
    assert model_torch.model_id == "native-fixture"
    assert calls[0] == {
        "loader": "torch",
        "path": native_dir,
        "device": "cpu",
        "max_length": 256,
        "prefix_cache_size": 32,
    }

    model_mlx = wyat.load(
        native_dir, backend="mlx", device="cpu", prefix_cache_size=4, max_length=128
    )
    assert model_mlx.model_id == "native-fixture"
    assert calls[1] == {
        "loader": "mlx",
        "path": native_dir,
        "device": "cpu",
        "max_length": 128,
        "responsive": True,
        "prefix_cache_size": 4,
    }

    wyat.load(native_dir, backend="mlx", responsive=False)
    assert calls[2]["responsive"] is False
    with pytest.raises(ValueError, match="only to native MLX"):
        wyat.load(native_dir, backend="torch", responsive=False)
    assert len(calls) == 3

    with pytest.raises(ValueError, match="sdpa attention"):
        wyat.load(native_dir, backend="torch", attention="eager")

    for bad_size in (-1, True, "16"):
        with pytest.raises(ValueError, match="prefix_cache_size"):
            wyat.load(native_dir, prefix_cache_size=bad_size)  # type: ignore[arg-type]


def test_budget_groups_keep_order_and_fit_each_padded_batch() -> None:
    from wyat import tasks
    from wyat.loading.native_artifact import budget_groups

    question = tasks.Question("q", "noul", "q", (("false", "no"), ("true", "yes")))

    def row(length: int) -> tasks.EncodedQuestion:
        return tasks.EncodedQuestion(
            question, tuple(range(length)), (length - 3, length - 2), length - 1
        )

    rows = tuple(row(length) for length in (4, 9, 3, 3, 8, 3))
    assert budget_groups(rows, 100) == [rows]
    groups = budget_groups(rows, 18)
    assert tuple(item for group in groups for item in group) == rows
    assert all(
        max(len(item.ids) for item in group) * len(group) <= 18 for group in groups
    )
    assert groups == [rows[:2], rows[2:4], rows[4:6]]
    # A row larger than the budget stays whole for the executor to refuse.
    assert budget_groups(rows[:2], 5) == [rows[:1], rows[1:2]]
    assert budget_groups((), 5) == []
