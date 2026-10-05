"""Historical execution is admitted from its archive, independently of live code."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

from benchmarks.analysis.torch.prefix import pairs
from benchmarks.analysis.torch.prefix.latency import study as latency
from benchmarks.analysis.torch.prefix.memory import study as memory


@pytest.mark.parametrize("stage", ["first8", "confirmation", "resource"])
def test_history_never_reads_live_execution_source(
    monkeypatch: pytest.MonkeyPatch,
    stage: Literal["first8", "confirmation", "resource"],
) -> None:
    digest = pairs.digest

    def historical_digest(path: Path) -> str:
        if path.suffix in {".py", ".h"}:
            raise AssertionError("Historical identity must come from executed archives")
        return digest(path)

    for module in (pairs, memory, latency):
        monkeypatch.setattr(module, "digest", historical_digest)
    history = pairs.admit_history()
    assert history["raw_outputs"] == 64
    if stage == "confirmation":
        assert memory.admit_confirmation(history)["raw_outputs"] == 2156
    elif stage == "resource":
        admitted, paths = latency.admit_resource()
        assert admitted["model_id"] == history["model_id"]
        assert paths and all(isinstance(path, Path) for path in paths.values())
