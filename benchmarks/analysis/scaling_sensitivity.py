"""Posthoc descriptive sensitivity; never replaces the retained measurement."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from argparse import ArgumentParser, Namespace
from dataclasses import asdict
from pathlib import Path

from wyat.wire import JsonObject, JsonValue, read_object

from .scaling import ScalingPoint, fit_usl


def _object(value: JsonValue) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError("Expected a receipt object")
    return value


def _objects(value: JsonValue) -> tuple[JsonObject, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("Expected nonempty measured cycles")
    return tuple(_object(item) for item in value)


def _number(value: JsonValue) -> float:
    if type(value) not in (int, float) or not isinstance(value, (int, float)):
        raise ValueError("Expected a numeric measurement")
    return float(value)


def sensitivity(path: Path) -> dict[str, object]:
    """Compare reciprocal mean/median cycle times without deleting observations."""
    original = read_object(path)
    sizes = _object(original["sizes"])
    points_mean: list[ScalingPoint] = []
    points_median: list[ScalingPoint] = []
    summaries: dict[str, object] = {}
    for name, value in sizes.items():
        arm = _object(value)
        if arm["passed"] is not True or arm["failed"] != 0:
            raise ValueError("Sensitivity requires all retained shapes to pass")
        cycles = _objects(arm["measured_cycles"])
        times = [_number(cycle["latency_ms"]) for cycle in cycles]
        cases = _number(arm["attempted"]) / len(cycles)
        mean = math.fsum(times) / len(times)
        median = statistics.median(times)
        points_mean.append(ScalingPoint(int(name), cases * 1000 / mean))
        points_median.append(ScalingPoint(int(name), cases * 1000 / median))
        worst = max(range(len(times)), key=times.__getitem__)
        cycle = cycles[worst]
        rows = _objects(cycle["rows"])
        slowest_residence = max(_number(row["latency_ms"]) for row in rows)
        summaries[name] = {
            "cycle_ms_in_original_order": times,
            "mean_cycle_ms": mean,
            "median_cycle_ms": median,
            "largest_cycle": {
                "zero_based_repeat": worst,
                "cycle_ms": times[worst],
                "failed": cycle["failed"],
                "correct": cycle["correct"],
                "peak_rss_before_bytes": cycle["peak_rss_before_bytes"],
                "peak_rss_after_bytes": cycle["peak_rss_after_bytes"],
                "slowest_batch_residence_ms": slowest_residence,
                "slowest_batch_case_ids": [
                    row["id"]
                    for row in rows
                    if _number(row["latency_ms"]) == slowest_residence
                ],
            },
        }
    return {
        "scope": "posthoc descriptive sensitivity; no timings removed or repeated",
        "original_receipt_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "original_source_sha256": original["source_sha256"],
        "analysis_source_sha256": {
            "calculator": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scaling": hashlib.sha256(
                Path(__file__).with_name("scaling.py").read_bytes()
            ).hexdigest(),
        },
        "sizes": summaries,
        "reciprocal_mean_cycle_fit": asdict(fit_usl(points_mean)),
        "reciprocal_median_cycle_fit": asdict(fit_usl(points_median)),
        "limits": [
            "Reciprocal mean cycle time equals overall successful throughput here; "
            "reciprocal median describes a typical cycle, not overall capacity.",
            "No physical fit accepted by changing the summary statistic.",
            "Historical normalized_design_condition denotes normalized Gram "
            "condition; current field correctly names that quantity.",
            "No OS pressure, queue telemetry or cycle wall timestamps were "
            "captured; cause of large delays is unknown.",
            "Cumulative RSS is shared process high-water, not current Metal memory.",
            "Fresh predeclared paired timing confirmation remains necessary "
            "before a batching recommendation.",
        ],
    }


class Arguments(Namespace):
    receipt: Path
    output: Path


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("output", type=Path)
    options = parser.parse_args(namespace=Arguments())
    result = sensitivity(options.receipt)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
