# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Option-fork dispatch preserves public loader options and dependency errors."""

from __future__ import annotations

import builtins
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import pytest

import wyat


@pytest.fixture
def fork_directory(tmp_path: Path) -> Path:
    directory = tmp_path / "fork"
    directory.mkdir()
    (directory / "fork_manifest.json").write_text("{}")
    return directory


@pytest.fixture
def loaded_fork(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    class ForkScorer:
        model_id = "fork-fixture"

        def score_many(
            self, requests: Sequence[wyat.Request]
        ) -> Sequence[Sequence[float]]:
            return [(1.0, 2.0) for _ in requests]

    def load_fork(path: str | Path, **kwargs: object) -> ForkScorer:
        calls.append({"path": path, **kwargs})
        return ForkScorer()

    module = ModuleType("wyat.loading.fork")
    monkeypatch.setattr(module, "load_fork", load_fork, raising=False)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return calls


def test_fork_dispatch_defaults_to_cpu_and_preserves_model_options(
    fork_directory: Path, loaded_fork: list[dict[str, object]]
) -> None:
    model = wyat.load(fork_directory, temperature=1.5)
    assert model.model_id == "fork-fixture"
    assert model.temperature == 1.5
    assert loaded_fork == [
        {"path": fork_directory, "device": "cpu", "max_length": None}
    ]

    wyat.load(
        fork_directory,
        device="cuda:1",
        max_length=256,
        token_cache_size=8,
        prefix_cache_size=32,
    )
    assert loaded_fork[1] == {
        "path": fork_directory,
        "device": "cuda:1",
        "max_length": 256,
    }


def test_fork_mlx_dispatch_defaults_to_gpu_and_passes_responsiveness(
    fork_directory: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []

    class ForkMLXScorer:
        model_id = "fork-mlx-fixture"

    def load_fork_mlx(path: str | Path, **kwargs: object) -> ForkMLXScorer:
        calls.append({"path": path, **kwargs})
        return ForkMLXScorer()

    module = ModuleType("wyat.loading.fork_mlx")
    monkeypatch.setattr(module, "load_fork_mlx", load_fork_mlx, raising=False)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    model = wyat.load(fork_directory, backend="mlx", temperature=1.5)
    assert model.model_id == "fork-mlx-fixture"
    assert model.temperature == 1.5
    wyat.load(
        fork_directory, backend="mlx", device="cpu", max_length=64, responsive=False
    )
    assert calls == [
        {
            "path": fork_directory,
            "device": "gpu",
            "max_length": None,
            "responsive": True,
        },
        {
            "path": fork_directory,
            "device": "cpu",
            "max_length": 64,
            "responsive": False,
        },
    ]
    with pytest.raises(ValueError, match="sdpa attention, uncompiled"):
        wyat.load(fork_directory, backend="mlx", compile=True)
    assert len(calls) == 2


def test_fork_mlx_missing_runtime_names_the_mlx_extra(
    monkeypatch: pytest.MonkeyPatch, fork_directory: Path
) -> None:
    original = ModuleNotFoundError("No module named 'mlx'", name="mlx")
    original_import = builtins.__import__

    def failing_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> ModuleType:
        if name == "wyat.loading.fork_mlx":
            raise original
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    with pytest.raises(ModuleNotFoundError) as raised:
        wyat.load(fork_directory, backend="mlx")
    assert "wyat[mlx]" in str(raised.value)
    assert raised.value.__cause__ is original


@pytest.mark.parametrize(
    "options,message",
    [
        ({"attention": "eager"}, "sdpa attention, uncompiled"),
        ({"compile": True}, "sdpa attention, uncompiled"),
        ({"responsive": False}, "only to native MLX and option-fork MLX"),
        ({"prefix_cache_size": -1}, "prefix_cache_size"),
        ({"prefix_cache_size": True}, "prefix_cache_size"),
    ],
)
def test_unsupported_fork_options_fail_before_loading(
    fork_directory: Path,
    loaded_fork: list[dict[str, object]],
    options: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        wyat.load(fork_directory, **options)  # type: ignore[arg-type]
    assert loaded_fork == []


@pytest.mark.parametrize("missing", ["torch", "transformers", "torch._C"])
def test_fork_missing_dependency_keeps_the_correct_error(
    monkeypatch: pytest.MonkeyPatch, fork_directory: Path, missing: str
) -> None:
    original = ModuleNotFoundError(f"No module named '{missing}'", name=missing)
    original_import = builtins.__import__

    def failing_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> ModuleType:
        if name == "wyat.loading.fork":
            raise original
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    with pytest.raises(ModuleNotFoundError) as raised:
        wyat.load(fork_directory)
    if missing == "torch":
        assert "wyat[inference]" in str(raised.value)
        assert raised.value.__cause__ is original
    else:
        assert raised.value is original


def test_fork_contract_import_remains_framework_free() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import wyat.loading.fork_artifact; "
            "assert not {'torch', 'transformers', 'safetensors'} & sys.modules.keys()",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
