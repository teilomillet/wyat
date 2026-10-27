# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""End-to-end parity and ownership of bounded Ouro inference storage."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

torch = pytest.importorskip("torch")
from test_ouro_artifact import _build_ouro_artifact  # noqa: E402

import wyat  # noqa: E402
from wyat.loading import ouro, ouro_artifact  # noqa: E402
from wyat.loading.native_artifact import file_sha256  # noqa: E402
from wyat.loading.ouro import OuroScorer  # noqa: E402
from wyat.loading.ouro_cache import DiskKVBank  # noqa: E402


def _artifact(path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    artifact = _build_ouro_artifact(
        path,
        cache_entries=0,
        context_prefill={
            "chunk": 64,
            "single_chunk_max_tokens": 128,
            "chunked_attention": "flash",
            "host_cache_above_tokens": 1024,
        },
        numerics="deterministic-no-tf32-v1",
    )
    # The integration fixture is a tiny, locally pinned model, not the public
    # 1.4B release. Exercise the same size/hash verification against its pins.
    monkeypatch.setattr(
        ouro_artifact,
        "PINNED_BASE_FILES",
        {
            name: {
                "size": (artifact / name).stat().st_size,
                "sha256": file_sha256(artifact / name),
            }
            for name in ouro_artifact.PINNED_BASE_FILES
        },
    )
    return artifact


def _record(long: bool) -> dict[str, Any]:
    return {
        "state": " note" * (60 if long else 1),
        "questions": {
            "q": {
                "type": "choice",
                "instructions": "Choose.",
                "criteria": {"a": "yes", "b": "no"},
            }
        },
    }


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("long", [False, True])
def test_streaming_matches_resident_and_restores_placeholders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device: str, long: bool
) -> None:
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Apple MPS unavailable")
    path = _artifact(tmp_path, monkeypatch)
    resident = wyat.load(path, device=device)
    stream = wyat.load(path, device=device, memory_mode="streaming")
    scorer = cast(OuroScorer, stream._scorer)
    before = {name: id(value) for name, value in scorer.backbone.named_parameters()}
    record = _record(long)
    expected, actual = resident.logits(record), stream.logits(record)
    if device == "cpu":
        assert actual == expected
    else:
        torch.testing.assert_close(
            torch.tensor(actual), torch.tensor(expected), atol=1e-4, rtol=1e-4
        )
    assert {
        name: id(value) for name, value in scorer.backbone.named_parameters()
    } == before
    assert all(
        value.is_meta
        for name, value in scorer.backbone.named_parameters()
        if ".lora_" not in name
    )
    assert (
        stream.predict(record)["q"].to_dict()["choice"]
        == resident.predict(record)["q"].to_dict()["choice"]
    )


def test_spill_failure_restores_weights_and_closes_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stream = wyat.load(
        _artifact(tmp_path, monkeypatch), device="cpu", memory_mode="streaming"
    )
    banks = []

    def fail(bank: DiskKVBank, key: Any, value: Any) -> None:
        banks.append(bank)
        raise OSError("injected write failure")

    monkeypatch.setattr(DiskKVBank, "append", fail)
    with pytest.raises(OSError, match="injected"):
        stream.logits(_record(True))
    assert len(banks) == 1 and banks[0].closed
    assert all(
        value.is_meta
        for name, value in cast(OuroScorer, stream._scorer).backbone.named_parameters()
        if ".lora_" not in name
    )


def test_streaming_rejects_unsupported_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _artifact(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="streaming requires"):
        ouro.load_ouro(path, device="cpu", memory_mode="streaming", prefix_cache_size=1)
    with pytest.raises(ValueError, match="streaming requires"):
        ouro.load_ouro(
            path,
            device="cpu",
            memory_mode="streaming",
            candidate_batching="per_candidate",
        )
