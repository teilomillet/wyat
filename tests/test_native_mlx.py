"""Seeded native runtime oracles establish mechanics, not learned quality."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
torch = pytest.importorskip("torch")
from fixtures.kai import create_kai_bundle  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from wyat import tasks  # noqa: E402
from wyat.loading.mlx import NativeMLXScorer, load_native_mlx  # noqa: E402
from wyat.loading.native import (  # noqa: E402
    FORMAT,
    MANIFEST,
    artifact_contract,
    file_sha256,
    load_native,
)


def _refresh_manifest(root: Path, **changes: object) -> None:
    path = root / MANIFEST
    manifest: dict[str, Any] = json.loads(path.read_text())
    files = {name: file_sha256(root / name) for name in manifest["files_sha256"]}
    manifest["files_sha256"] = files
    manifest["model_id"] = (
        "native-"
        + hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    )
    manifest.update(changes)
    path.write_text(json.dumps(manifest))


def _create_native_bundle(root: Path, *, bf16: bool = False) -> Path:
    create_kai_bundle(root)
    (root / "decision_config.json").unlink()
    if bf16:
        path = root / "backbone/model.safetensors"
        state = load_file(str(path))
        state = {
            name: value.bfloat16() if name.endswith("_proj.weight") else value
            for name, value in state.items()
        }
        save_file(state, path)
    with torch.random.fork_rng():
        torch.manual_seed(91)
        adapters = {
            f"layers.0.self_attn.{projection}.{factor}": torch.randn(shape) * 0.06
            for projection, output in (("q_proj", 16), ("v_proj", 8))
            for factor, shape in (("down", (16, 16)), ("up", (output, 16)))
        }
    save_file(adapters, root / "native-adapters.safetensors")
    files = {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in root.rglob("*")
        if path.is_file()
    }
    manifest = {
        "format": FORMAT,
        "model_id": "native-"
        + hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        "contract": artifact_contract(),
        "adapter": {
            "rank": 16,
            "alpha": 32,
            "targets": ["q_proj", "v_proj"],
            "merged": False,
        },
        "max_length": 512,
        "max_batch_tokens": 2048,
        "head_dim": 8,
        "files_sha256": files,
        "provenance": {"fixture": "seeded-mechanics-only"},
        "production_kai_compatible": False,
        "resume_supported": False,
    }
    (root / MANIFEST).write_text(json.dumps(manifest))
    return root


@pytest.fixture(params=[False, True], ids=["fp32", "mixed-bf16"])
def bundle(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    return _create_native_bundle(tmp_path / "native", bf16=bool(request.param))


@pytest.fixture(params=["cpu", "gpu"])
def scorer(bundle: Path, request: pytest.FixtureRequest) -> NativeMLXScorer:
    device = str(request.param)
    if device == "gpu" and not mx.metal.is_available():
        pytest.skip("Metal GPU is unavailable")
    return load_native_mlx(bundle, device=device, metal=False)


def _record(*, reverse_noul: bool = False) -> dict[str, Any]:
    binary = (
        {"true": "yes", "false": "no"}
        if reverse_noul
        else {"false": "no", "true": "yes"}
    )
    return {
        "state": {"text": "Bonjour 世界 é", "count": 7},
        "questions": {
            "choice": {
                "type": "choice",
                "instructions": "read?",
                "criteria": {"é": "yes", "世界": "no", "a": "write " * 9},
            },
            "score": {"type": "score", "criteria": ["no", "read " * 4, "yes"]},
            "binary": {"type": "noul", "criteria": binary},
        },
    }


@pytest.mark.parametrize("reverse_noul", [False, True])
def test_native_encoding_logits_and_all_typed_readouts_match_torch(
    bundle: Path, scorer: NativeMLXScorer, reverse_noul: bool
) -> None:
    reference = load_native(bundle)
    record = _record(reverse_noul=reverse_noul)
    assert scorer.model_id == reference.model_id
    assert scorer.encode_record(record) == reference.encode_record(record)
    actual_logits = scorer.logits(record)
    expected_logits = reference.logits(record)
    for actual_row, expected_row in zip(actual_logits, expected_logits, strict=True):
        assert actual_row == pytest.approx(expected_row, abs=2e-5, rel=2e-5)
    actual_answers, expected_answers = scorer.predict(record), reference.predict(record)
    assert tuple(actual_answers) == ("choice", "score", "binary")
    for key, answer in actual_answers.items():
        expected = expected_answers[key]
        assert type(answer) is type(expected)
        assert answer.option_ids == expected.option_ids
        assert answer.probabilities == pytest.approx(
            expected.probabilities, abs=2e-5, rel=2e-5
        )
        if isinstance(answer, tasks.ChoiceAnswer):
            assert isinstance(expected, tasks.ChoiceAnswer)
            assert answer.choice == expected.choice
            assert answer.ranking == expected.ranking
        elif isinstance(answer, tasks.ScoreAnswer):
            assert isinstance(expected, tasks.ScoreAnswer)
            assert answer.expected_ordinal_value == pytest.approx(
                expected.expected_ordinal_value, abs=2e-5, rel=2e-5
            )
        else:
            assert isinstance(expected, tasks.NoulAnswer)
            assert answer.probability_true == pytest.approx(
                expected.probability_true, abs=2e-5, rel=2e-5
            )


def test_native_independent_rows_and_right_padding(scorer: NativeMLXScorer) -> None:
    record = _record()
    encoded = scorer.encode_record(record)
    batched = scorer._logits(encoded)
    assert len({len(row.ids) for row in encoded}) > 1
    for row, actual in zip(encoded, batched, strict=True):
        assert scorer._logits([row])[0] == pytest.approx(actual, abs=2e-5, rel=2e-5)
    modified = copy.deepcopy(record)
    modified["questions"]["binary"]["instructions"] = "write " * 75
    assert scorer.encode_record(modified)[0] == encoded[0]
    assert scorer.logits(modified)[0] == pytest.approx(batched[0], abs=2e-5, rel=2e-5)
    assert scorer._logits([]) == []


def test_native_padding_budget_rejected_before_backbone(
    scorer: NativeMLXScorer, monkeypatch: pytest.MonkeyPatch
) -> None:
    encoded = scorer.encode_record(_record())
    scorer.max_batch_tokens = max(len(row.ids) for row in encoded) * len(encoded) - 1

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("backbone executed before budget validation")

    monkeypatch.setattr(type(scorer.backbone), "__call__", forbidden)
    with pytest.raises(ValueError, match="budget"):
        scorer._logits(encoded)


def test_native_context_limit_never_truncates(bundle: Path) -> None:
    scorer = load_native_mlx(bundle, device="cpu", max_length=16, metal=False)
    with pytest.raises(ValueError):
        scorer.encode_record(_record())
    for maximum in (0, -1, 513, True):
        with pytest.raises(ValueError):
            load_native_mlx(bundle, device="cpu", max_length=maximum, metal=False)


@pytest.mark.parametrize(
    "change",
    [
        {"attention_bias": True},
        {"attention_dropout": 0.1},
        {"hidden_act": "gelu"},
        {"use_sliding_window": True, "sliding_window": 32},
        {
            "rope_parameters": {
                "rope_type": "linear",
                "rope_theta": 1000000.0,
                "factor": 2,
            }
        },
        {"rope_theta": 10000.0},
    ],
)
def test_unsupported_native_configs_fail_closed(
    bundle: Path, change: dict[str, object]
) -> None:
    path = bundle / "backbone/config.json"
    config = json.loads(path.read_text())
    config.update(change)
    path.write_text(json.dumps(config))
    _refresh_manifest(bundle)
    with pytest.raises(ValueError):
        load_native_mlx(bundle, device="cpu", metal=False)


@pytest.mark.parametrize(
    ("filename", "change"),
    [
        ("native-adapters.safetensors", "missing"),
        ("native-adapters.safetensors", "shape"),
        ("native-adapters.safetensors", "dtype"),
        ("native-adapters.safetensors", "nonfinite"),
        ("backbone/model.safetensors", "missing"),
        ("backbone/model.safetensors", "dtype"),
        ("backbone/model.safetensors", "nonfinite"),
        ("decision_head.safetensors", "missing"),
        ("decision_head.safetensors", "dtype"),
        ("decision_head.safetensors", "nonfinite"),
    ],
)
def test_malformed_native_tensors_fail_closed(
    bundle: Path, filename: str, change: str
) -> None:
    path = bundle / filename
    state = load_file(str(path))
    name = next(iter(state))
    transforms: dict[str, Callable[[Any], Any]] = {
        "shape": lambda value: value.reshape(-1),
        "dtype": lambda value: value.half(),
        "nonfinite": lambda value: torch.full_like(value, float("nan")),
    }
    if change == "missing":
        del state[name]
    else:
        state[name] = transforms[change](state[name])
    save_file(state, path)
    _refresh_manifest(bundle)
    with pytest.raises(ValueError):
        load_native_mlx(bundle, device="cpu", metal=False)


def test_native_contract_mismatch_is_not_bypassed(bundle: Path) -> None:
    _refresh_manifest(bundle, contract={"recipe": "incompatible"})
    with pytest.raises(ValueError, match="contract differs"):
        load_native_mlx(bundle, device="cpu", metal=False)


def test_native_changed_weight_identity_is_not_bypassed(bundle: Path) -> None:
    path = bundle / "native-adapters.safetensors"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="identity differs"):
        load_native_mlx(bundle, device="cpu", metal=False)
