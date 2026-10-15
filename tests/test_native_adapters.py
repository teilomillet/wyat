# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""The all-linear adapter policy: complete coverage, exact math, legacy loads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
from fixtures.wyat import create_native_bundle  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402
from torch import nn  # noqa: E402
from transformers import Qwen3Config  # noqa: E402
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer  # noqa: E402

from wyat.loading.native import load_native  # noqa: E402
from wyat.loading.native_artifact import (  # noqa: E402
    ADAPTER,
    ADAPTER_TARGETS,
    MANIFEST,
    file_sha256,
    model_identity,
)

RECORD = {
    "state": {"text": "Bonjour 世界 é", "count": 7},
    "questions": {
        "choice": {
            "type": "choice",
            "instructions": "read?",
            "criteria": {"é": "yes", "世界": "no", "a": "write " * 9},
        },
        "score": {"type": "score", "criteria": ["no", "read " * 4, "yes"]},
        "binary": {"type": "noul", "criteria": {"false": "no", "true": "yes"}},
    },
}


def _refresh_manifest(root: Path, **changes: object) -> None:
    path = root / MANIFEST
    manifest: dict[str, Any] = json.loads(path.read_text())
    files = {name: file_sha256(root / name) for name in manifest["files_sha256"]}
    manifest.update(files_sha256=files, model_id=model_identity(files), **changes)
    path.write_text(json.dumps(manifest))


def test_policy_names_every_linear_module_of_a_qwen3_block() -> None:
    config = Qwen3Config(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
    )
    block = Qwen3DecoderLayer(config, layer_idx=0)
    linear = {
        tuple(name.split("."))
        for name, module in block.named_modules()
        if isinstance(module, nn.Linear)
    }
    assert linear == set(ADAPTER_TARGETS)
    assert ADAPTER["targets"] == [f"{owner}.{name}" for owner, name in ADAPTER_TARGETS]


def test_all_linear_adapters_equal_their_merged_weights(tmp_path: Path) -> None:
    """Every declared adapter applies W + 2 * up @ down, checked against merging."""
    adapted = create_native_bundle(tmp_path / "adapted")
    merged = create_native_bundle(tmp_path / "merged")
    base = load_file(str(merged / "backbone/model.safetensors"))
    factors = load_file(str(merged / "native-adapters.safetensors"))
    for owner, name in ADAPTER_TARGETS:
        prefix = f"layers.0.{owner}.{name}"
        delta = 2 * factors[prefix + ".up"] @ factors[prefix + ".down"]
        base[prefix + ".weight"] = base[prefix + ".weight"] + delta
        factors[prefix + ".up"] = torch.zeros_like(factors[prefix + ".up"])
    save_file(base, merged / "backbone/model.safetensors")
    save_file(factors, merged / "native-adapters.safetensors")
    _refresh_manifest(merged)

    expected = load_native(merged).logits(RECORD)
    actual = load_native(adapted).logits(RECORD)
    for actual_row, expected_row in zip(actual, expected, strict=True):
        assert actual_row == pytest.approx(expected_row, abs=1e-5, rel=1e-5)


@pytest.mark.parametrize(("owner", "name"), ADAPTER_TARGETS)
def test_each_adapter_target_changes_the_logits(
    tmp_path: Path, owner: str, name: str
) -> None:
    bundle = create_native_bundle(tmp_path / "native")
    reference = load_native(bundle).logits(RECORD)
    path = bundle / "native-adapters.safetensors"
    factors = load_file(str(path))
    up = f"layers.0.{owner}.{name}.up"
    factors[up] = torch.zeros_like(factors[up])
    save_file(factors, path)
    _refresh_manifest(bundle)
    assert load_native(bundle).logits(RECORD) != reference


def test_released_qv_adapters_still_load(tmp_path: Path) -> None:
    bundle = create_native_bundle(tmp_path / "native", adapter="legacy-qv")
    assert len(load_native(bundle).logits(RECORD)) == 3


@pytest.mark.parametrize(
    "adapter",
    [
        ADAPTER | {"targets": [f"{o}.{n}" for o, n in ADAPTER_TARGETS[:-1]]},
        ADAPTER | {"rank": 8},
        ADAPTER | {"merged": True},
        {"rank": 16, "alpha": 32, "targets": ["q_proj", "v_proj", "o_proj"]},
    ],
    ids=["missing-down-proj", "rank-8", "merged", "partial-attention"],
)
def test_other_adapter_contracts_are_refused(
    tmp_path: Path, adapter: dict[str, object]
) -> None:
    bundle = create_native_bundle(tmp_path / "native")
    _refresh_manifest(bundle, adapter=adapter)
    with pytest.raises(ValueError, match="unsupported native adapter contract"):
        load_native(bundle)


def test_declared_qv_with_all_linear_tensors_is_refused(tmp_path: Path) -> None:
    bundle = create_native_bundle(tmp_path / "native")
    legacy = {"rank": 16, "alpha": 32, "targets": ["q_proj", "v_proj"], "merged": False}
    _refresh_manifest(bundle, adapter=legacy)
    with pytest.raises(ValueError, match="adapter tensor inventory differs"):
        load_native(bundle)
