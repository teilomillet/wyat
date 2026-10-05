from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import pytest

from benchmarks.analysis.scaling_sensitivity import sensitivity


def _receipt(path: Path, *, passed: bool = True) -> None:
    sizes: dict[str, object] = {}
    for size in (1, 2, 4, 8, 16, 32):
        cycles = [
            {
                "latency_ms": milliseconds,
                "failed": 0,
                "correct": 1,
                "peak_rss_before_bytes": 1234,
                "peak_rss_after_bytes": 1234,
                "rows": [
                    {"id": "one", "latency_ms": milliseconds / 2},
                    {"id": "two", "latency_ms": milliseconds / 2},
                ],
            }
            for milliseconds in (100 / size, 100 / size, 1000 / size)
        ]
        sizes[str(size)] = {
            "passed": passed,
            "failed": 0,
            "attempted": 6,
            "measured_cycles": cycles,
        }
    path.write_text(json.dumps({"sizes": sizes, "source_sha256": {"runner": "old"}}))


def test_sensitivity_retains_delay_and_separates_total_from_typical_capacity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "measurement.json"
    _receipt(path)
    original = path.read_bytes()
    report = sensitivity(path)
    assert path.read_bytes() == original
    assert report["original_receipt_sha256"] == hashlib.sha256(original).hexdigest()
    mean_fit = cast(dict[str, object], report["reciprocal_mean_cycle_fit"])
    median_fit = cast(dict[str, object], report["reciprocal_median_cycle_fit"])
    # Six completed requests in 1.2 seconds is 5/s; typical 2-request calls
    # take .1 seconds, so reciprocal-median throughput is 20/s.
    assert mean_fit["baseline_throughput"] == 5
    assert median_fit["baseline_throughput"] == 20
    sizes = cast(dict[str, dict[str, object]], report["sizes"])
    assert sizes["1"]["cycle_ms_in_original_order"] == [100, 100, 1000]
    largest = cast(dict[str, object], sizes["1"]["largest_cycle"])
    assert largest["zero_based_repeat"] == 2
    assert largest["slowest_batch_case_ids"] == ["one", "two"]
    assert largest["slowest_batch_residence_ms"] == 500


def test_failed_quality_cannot_be_reframed_as_scaling_capacity(tmp_path: Path) -> None:
    path = tmp_path / "measurement.json"
    _receipt(path, passed=False)
    with pytest.raises(ValueError, match="all retained shapes"):
        sensitivity(path)
