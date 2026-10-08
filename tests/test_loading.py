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
        if name == f"wyat.backends.kai.{backend}":
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
        ("wyat.backends.kai.torch", "KaiScorer"),
        ("wyat.backends.kai.mlx", "MLXKaiScorer"),
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
