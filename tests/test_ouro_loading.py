# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Ouro decision dispatch preserves public loader options and dependency errors."""

from __future__ import annotations

import ast
import builtins
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import pytest

import wyat
from wyat.loading import ouro_artifact

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_OURO_IDENTIFIERS = frozenset(
    {
        "OuroForkModel",
        "OptionForkModel",
        "SegmentedQuestion",
        "PrefixState",
        "LayerKV",
        "ScalarHead",
        "BranchAdapterController",
        "BranchAdapterLinear",
        "attach_branch_adapters",
        "branch_hidden",
        "branch_logits",
        "cached_branch_logits",
        "prefix_ids",
        "option_ids",
        "PREFIX_LOOPS",
        "PREFIX_BANK",
        "BRANCH_LOOPS",
        "BRANCH_POLICY",
        "TOTAL_LOOPS",
        "reuse_prefix",
        "_branch_adapters",
    }
)


@pytest.fixture
def ouro_directory(tmp_path: Path) -> Path:
    directory = tmp_path / "ouro"
    directory.mkdir()
    (directory / "ouro_manifest.json").write_text("{}")
    return directory


@pytest.fixture
def loaded_ouro(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    class OuroScorer:
        model_id = "ouro-fixture"

        def score_many(
            self, requests: Sequence[wyat.Request]
        ) -> Sequence[Sequence[float]]:
            return [(1.0, 2.0) for _ in requests]

    def load_ouro(path: str | Path, **kwargs: object) -> OuroScorer:
        calls.append({"path": path, **kwargs})
        return OuroScorer()

    module = ModuleType("wyat.loading.ouro")
    monkeypatch.setattr(module, "load_ouro", load_ouro, raising=False)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return calls


def test_ouro_dispatch_defaults_to_cpu_and_preserves_model_options(
    ouro_directory: Path, loaded_ouro: list[dict[str, object]]
) -> None:
    model = wyat.load(ouro_directory, temperature=1.25)
    assert model.model_id == "ouro-fixture"
    assert model.temperature == 1.25
    assert loaded_ouro == [
        {
            "path": ouro_directory,
            "device": "cpu",
            "max_length": None,
            "prefix_cache_size": 0,
            "attention": "sdpa",
        }
    ]

    wyat.load(
        ouro_directory,
        device="cuda:0",
        max_length=896,
        prefix_cache_size=8,
        attention="eager",
    )
    assert loaded_ouro[1] == {
        "path": ouro_directory,
        "device": "cuda:0",
        "max_length": 896,
        "prefix_cache_size": 8,
        "attention": "eager",
    }


@pytest.mark.parametrize(
    "options,message",
    [
        ({"backend": "mlx"}, "supports the torch backend only"),
        ({"compile": True}, "does not support compile"),
        ({"responsive": False}, "only to native MLX and option-fork MLX"),
        ({"prefix_cache_size": -1}, "prefix_cache_size"),
        ({"prefix_cache_size": True}, "prefix_cache_size"),
    ],
)
def test_unsupported_ouro_options_fail_before_loading(
    ouro_directory: Path,
    loaded_ouro: list[dict[str, object]],
    options: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        wyat.load(ouro_directory, **options)  # type: ignore[arg-type]
    assert loaded_ouro == []


@pytest.mark.parametrize("missing", ["torch", "transformers", "torch._C"])
def test_ouro_missing_dependency_keeps_the_correct_error(
    monkeypatch: pytest.MonkeyPatch, ouro_directory: Path, missing: str
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
        if name == "wyat.loading.ouro":
            raise original
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    with pytest.raises(ModuleNotFoundError) as raised:
        wyat.load(ouro_directory)
    if missing == "torch":
        assert "wyat[inference]" in str(raised.value)
        assert raised.value.__cause__ is original
    else:
        assert raised.value is original


def test_ouro_contract_import_remains_framework_free() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import wyat.loading.ouro_artifact; "
            "assert not {'torch', 'transformers', 'safetensors'} & sys.modules.keys()",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_ouro_modules_forbid_legacy_nomenclature() -> None:
    for relative in ("wyat/loading/ouro_artifact.py", "wyat/loading/ouro.py"):
        tree = ast.parse(
            (ROOT / relative).read_text(encoding="utf-8"), filename=relative
        )
        found: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in FORBIDDEN_OURO_IDENTIFIERS:
                found.add(node.id)
            elif (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and node.name in FORBIDDEN_OURO_IDENTIFIERS
            ):
                found.add(node.name)
            elif (
                isinstance(node, ast.Attribute)
                and node.attr in FORBIDDEN_OURO_IDENTIFIERS
            ):
                found.add(node.attr)
        assert not found, (
            f"{relative} uses forbidden legacy identifiers: {sorted(found)}"
        )
    for forbidden in FORBIDDEN_OURO_IDENTIFIERS:
        assert not hasattr(ouro_artifact, forbidden), forbidden
