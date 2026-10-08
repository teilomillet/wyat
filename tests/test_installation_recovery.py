"""Recovery guidance works for installed applications and preserves import failures."""

from __future__ import annotations

import builtins
import shlex
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Literal

import pytest

import wyat


@pytest.mark.parametrize("backend,extra", [("torch", "inference"), ("mlx", "mlx")])
def test_application_missing_runtime_offers_original_wheel_first(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backend: Literal["torch", "mlx"],
    extra: str,
) -> None:
    # A consuming application's working directory has no Wyat source checkout.
    monkeypatch.chdir(tmp_path)
    original = ModuleNotFoundError(f"No module named '{backend}'", name=backend)
    _fail_backend_import(monkeypatch, backend, original)
    with pytest.raises(ModuleNotFoundError) as caught:
        wyat.load("application-model", backend=backend)

    error = caught.value
    message = str(error)
    wheel_command = message.split("run: ", 1)[1].split(" (use the wheel", 1)[0]
    arguments = shlex.split(wheel_command)
    assert arguments == [
        "uv",
        "add",
        "/absolute/path/to/wyat-0.0.1-py3-none-any.whl",
        "--extra",
        extra,
    ]
    assert "use the wheel you installed" in message
    assert f"uv add /absolute/path/to/wyat --extra {extra}" in message
    assert f"uv sync --extra {extra}" in message
    assert message.index(".whl") < message.index("For a source checkout")
    assert error.name == backend
    assert error.__cause__ is original
    if backend == "mlx":
        assert "requires Apple Silicon macOS" in message


@pytest.mark.parametrize(
    "backend,missing",
    [("torch", "torch._C"), ("mlx", "mlx.core"), ("mlx", "torch")],
)
def test_application_nested_runtime_failure_keeps_original_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backend: Literal["torch", "mlx"],
    missing: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    original = ModuleNotFoundError(f"No module named '{missing}'", name=missing)
    _fail_backend_import(monkeypatch, backend, original)
    with pytest.raises(ModuleNotFoundError) as caught:
        wyat.load("application-model", backend=backend)
    assert caught.value is original
    assert caught.value.__cause__ is None


def _fail_backend_import(
    monkeypatch: pytest.MonkeyPatch,
    backend: Literal["torch", "mlx"],
    error: ModuleNotFoundError,
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
