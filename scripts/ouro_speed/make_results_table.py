#!/usr/bin/env python3
"""Render the final Ouro-on-Mac benchmark table from receipts — profiling only.

Reads a manifest (written by the final-suite runner) that lists each arm's
bench receipt, optional saved logits and optional parity report, plus a CPU
default baseline per workload, then emits ``FINAL-RESULTS.md`` with:

* arm x workload x median end-to-end time
* speedup vs the CPU default arm for the same workload
* max |Δlogit| and rank/top-1 flips vs the paired default arm (from parity.py
  reports or saved logits compared on the fly)
* memory high-water (RSS for CPU, MPS current/driver for MPS)
* integrity columns: golden check, self-consistency, loader stability

It never loads the 1.4B model; everything comes from receipt JSON/logits files.

Manifest schema (all paths relative to ``--artifacts-dir``)::

  {
    "schema": "ouro-speed-final-manifest/v1",
    "loader_sha256_expected": "...",
    "arms": [
      {"arm": "cpu-default", "workload": "p400-k6-s16",
       "device": "cpu", "candidate_batching": false,
       "receipt": "baseline-cpu-p400-k6-s16.json",
       "logits": "logits-cpu-p400-k6-s16.pt",
       "paired_default_logits": null},
      {"arm": "cpu-batching", "workload": "p400-k6-s16",
       "device": "cpu", "candidate_batching": true,
       "receipt": "...json", "logits": "...pt",
       "paired_default_logits": "logits-cpu-p400-k6-s16.pt"}
    ],
    "parity_reports": [
      {"label": "cpu-batching drift", "workload": "p400-k6-s16",
       "report": "parity-cpu-batching-vs-default-p400-k6-s16.json"}
    ]
  }

Usage::

  .venv/bin/python scripts/ouro_speed/make_results_table.py \
      --manifest artifacts/ouro-speed-20261021/final-suite-manifest.json \
      --out artifacts/ouro-speed-20261021/FINAL-RESULTS.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from _common import (  # noqa: E402
    DEFAULT_ATOL,
    PROFILING_NOTICE,
    compare_logits,
    read_json,
)

DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "ouro-speed-20261021"


def load_logits_list(path: Path) -> list[float] | None:
    if not path.exists():
        return None
    suffix = path.suffix.lower()
    if suffix == ".pt" or suffix == ".pth":
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("logits", payload)
        if isinstance(payload, dict):
            payload = payload.get("values", payload)
    if hasattr(payload, "tolist"):
        payload = payload.tolist()
    if not isinstance(payload, (list, tuple)):
        return None
    return [float(value) for value in payload]


def stage_median(receipt: dict[str, Any], stage: str) -> float | None:
    summary = receipt.get("stages", {}).get(stage)
    if isinstance(summary, dict):
        if "median_s" in summary:
            return float(summary["median_s"])
        if "seconds" in summary:
            return float(summary["seconds"])
    return None


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def fmt_sci(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.2e}"
    return str(value)


def memory_text(receipt: dict[str, Any]) -> str:
    memory = receipt.get("memory", {})
    mps = memory.get("mps") or {}
    at_end = mps.get("at_end") or {}
    driver = at_end.get("driver_allocated_gib")
    if driver is not None:
        return f"MPS {fmt(driver, 2)} GiB drv"
    peak = memory.get("peak_rss_gib")
    return f"RSS {fmt(peak, 2)} GiB" if peak is not None else "—"


def integrity_text(receipt: dict[str, Any]) -> str:
    parts = []
    provenance = receipt.get("provenance", {})
    if provenance.get("unstable_loader"):
        parts.append("UNSTABLE")
    else:
        parts.append("stable")
    golden = receipt.get("golden_check")
    if golden is not None:
        parts.append("golden " + ("PASS" if golden.get("passed") else "FAIL"))
    self_consistency = receipt.get("self_consistency")
    if self_consistency is not None:
        parts.append("self " + ("PASS" if self_consistency.get("passed") else "FAIL"))
    return ", ".join(parts)


def drift_cell(
    entry: dict[str, Any],
    receipts: dict[str, dict[str, Any]],
    logits_cache: dict[str, list[float] | None],
    artifacts_dir: Path,
) -> str:
    report_path = entry.get("parity_report")
    if report_path:
        target = artifacts_dir / report_path
        if target.exists():
            report = read_json(target)
            comparison = report.get("comparison", report)
            max_abs = comparison.get("max_abs_diff")
            rank = comparison.get("rank_order_changed")
            top1 = comparison.get("top1_changed")
            return f"{fmt_sci(max_abs)} (rank {rank}, top1 {top1})"
    paired = entry.get("paired_default_logits")
    logits_name = entry.get("logits")
    if paired and logits_name:
        left = logits_cache.setdefault(
            str(artifacts_dir / logits_name),
            load_logits_list(artifacts_dir / logits_name),
        )
        right = logits_cache.setdefault(
            str(artifacts_dir / paired), load_logits_list(artifacts_dir / paired)
        )
        if left and right:
            comparison = compare_logits(left, right, atol=DEFAULT_ATOL)
            return (
                f"{fmt_sci(comparison.get('max_abs_diff'))} "
                f"(rank {comparison.get('rank_order_changed')}, "
                f"top1 {comparison.get('top1_changed')})"
            )
    return "—"


def build_table(
    manifest: dict[str, Any], artifacts_dir: Path
) -> tuple[str, dict[str, Any]]:
    arms = list(manifest.get("arms", []))
    receipts: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for entry in arms:
        receipt_path = artifacts_dir / entry.get("receipt", "")
        receipt = read_json(receipt_path) if receipt_path.exists() else {}
        receipts[entry["receipt"]] = receipt
        rows.append(
            {
                "entry": entry,
                "receipt": receipt,
                "device": entry.get("device")
                or receipt.get("environment", {}).get("device")
                or receipt.get("scorer", {}).get("device"),
                "e2e": stage_median(receipt, "end_to_end"),
                "prefill": stage_median(receipt, "prefill_stage1"),
                "candidate": stage_median(receipt, "candidate_scoring_from_cache"),
                "load": stage_median(receipt, "load"),
                "status": receipt.get("status"),
            }
        )

    # CPU default baseline per workload for speedups.
    baseline: dict[str, float] = {}
    for row in rows:
        entry = row["entry"]
        if (
            str(row["device"]).startswith("cpu")
            and not entry.get("candidate_batching", False)
            and row["e2e"] is not None
        ):
            baseline.setdefault(entry["workload"], row["e2e"])

    logits_cache: dict[str, list[float] | None] = {}
    lines: list[str] = []
    lines.append("# Ouro on Mac — final controlled benchmark results")
    lines.append("")
    lines.append(f"> {PROFILING_NOTICE}")
    lines.append("")
    loader_expected = manifest.get("loader_sha256_expected")
    lines.append(
        f"- frozen loader: `wyat/loading/ouro.py` sha256 "
        f"`{loader_expected or 'see receipts'}`"
    )
    lines.append(
        f"- declared logit tolerance: abs {DEFAULT_ATOL} "
        f"(rank order/top-1 must be identical for decision parity)"
    )
    lines.append(
        "- protocol: exclusive BENCH_LOCK, nice 10, threads 6, RSS watchdog "
        "(hard abort), MPS fraction 0.6 + LOW/HIGH watermark 1.0/1.2, "
        "sequential only"
    )
    lines.append(
        "- arms are single-process; medians over the repetition counts in "
        "each receipt (`protocol.warmups`/`repeats`)"
    )
    lines.append("")
    lines.append("## Table")
    lines.append("")
    lines.append(
        "| Arm | Workload | Device | Batching | median e2e (s) | Speedup vs CPU "
        "default | Max abs Δlogit (rank/top1) | Memory | Integrity |"
    )
    lines.append("| :--- | :--- | :---: | :---: | ---: | ---: | :--- | :--- | :--- |")
    for row in rows:
        entry = row["entry"]
        workload = entry["workload"]
        speedup = "—"
        base = baseline.get(workload)
        if base and row["e2e"]:
            speedup = f"{base / row['e2e']:.2f}x"
        batching = "yes" if entry.get("candidate_batching") else "no"
        drift = drift_cell(entry, receipts, logits_cache, artifacts_dir)
        lines.append(
            f"| {entry['arm']} | {workload} | {row['device']} | {batching} | "
            f"{fmt(row['e2e'])} | {speedup} | {drift} | "
            f"{memory_text(row['receipt'])} | {integrity_text(row['receipt'])} |"
        )
    lines.append("")

    # Stage breakdown table.
    lines.append("## Stage medians")
    lines.append("")
    lines.append(
        "| Arm | Workload | load s | load peak RSS GiB | prefill s | candidate s | "
        "candidate speedup vs CPU default | e2e s |"
    )
    lines.append("| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |")
    candidate_baseline: dict[str, float] = {}
    for row in rows:
        entry = row["entry"]
        if (
            str(row["device"]).startswith("cpu")
            and not entry.get("candidate_batching", False)
            and row["candidate"] is not None
        ):
            candidate_baseline.setdefault(entry["workload"], row["candidate"])
    for row in rows:
        entry = row["entry"]
        load_summary = row["receipt"].get("stages", {}).get("load", {})
        load_peak = (
            load_summary.get("peak_rss_gib") if isinstance(load_summary, dict) else None
        )
        candidate_speedup = "—"
        base_candidate = candidate_baseline.get(entry["workload"])
        if base_candidate and row["candidate"]:
            candidate_speedup = f"{base_candidate / row['candidate']:.2f}x"
        lines.append(
            f"| {entry['arm']} | {entry['workload']} | {fmt(row['load'])} | "
            f"{fmt(load_peak)} | {fmt(row['prefill'])} | {fmt(row['candidate'])} | "
            f"{candidate_speedup} | {fmt(row['e2e'])} |"
        )
    lines.append("")

    parity_reports = list(manifest.get("parity_reports", []))
    if parity_reports:
        lines.append("## Parity / drift reports")
        lines.append("")
        lines.append(
            "| Comparison | Workload | max abs Δ | bitwise | rank changed | "
            "top1 changed | passed |"
        )
        lines.append("| :--- | :--- | ---: | :---: | :---: | :---: | :---: |")
        for item in parity_reports:
            target = artifacts_dir / item.get("report", "")
            if not target.exists():
                continue
            report = read_json(target)
            if report.get("kind") == "parity-collection":
                collection = report.get("collection", {})
                worst = collection.get("worst", {})
                failures = collection.get("failures", [])
                rank_flips = sum(
                    1 for entry in failures if entry.get("rank_order_changed")
                )
                top1_flips = sum(1 for entry in failures if entry.get("top1_changed"))
                lines.append(
                    f"| {item.get('label')} | collections "
                    f"({collection.get('entries_total')} pairs) | "
                    f"{fmt_sci(worst.get('max_abs_diff'))} | "
                    f"{worst.get('bitwise_equal')} | "
                    f"{rank_flips} | {top1_flips} | "
                    f"{collection.get('passed')} |"
                )
                continue
            comparison = report.get("comparison", {})
            lines.append(
                f"| {item.get('label')} | {item.get('workload')} | "
                f"{fmt_sci(comparison.get('max_abs_diff'))} | "
                f"{comparison.get('bitwise_equal')} | "
                f"{comparison.get('rank_order_changed')} | "
                f"{comparison.get('top1_changed')} | "
                f"{report.get('passed')} |"
            )
        lines.append("")

    lines.append("## Receipts")
    lines.append("")
    for entry in arms:
        lines.append(
            f"- `{entry['arm']}` / `{entry['workload']}`: "
            f"[receipt]({entry.get('receipt')})"
            + (f", [logits]({entry.get('logits')})" if entry.get("logits") else "")
        )
    for item in parity_reports:
        lines.append(f"- parity `{item.get('label')}`: [report]({item.get('report')})")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    for note in manifest.get("notes", []):
        lines.append(f"- {note}")
    lines.append("")
    summary = {
        "arms": len(rows),
        "passed_arms": sum(
            1
            for row in rows
            if row["status"] == "ok"
            and not row["receipt"].get("provenance", {}).get("unstable_loader")
            and row["receipt"].get("golden_check", {}).get("passed", True)
            and row["receipt"].get("self_consistency", {}).get("passed", True)
        ),
        "baselines": baseline,
    }
    return "\n".join(lines), summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render FINAL-RESULTS.md from Ouro speed receipts (profiling only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest", default=str(DEFAULT_ARTIFACTS_DIR / "final-suite-manifest.json")
    )
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument(
        "--out", default=str(DEFAULT_ARTIFACTS_DIR / "FINAL-RESULTS.md")
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    manifest = read_json(manifest_path)
    artifacts_dir = Path(args.artifacts_dir)
    markdown, summary = build_table(manifest, artifacts_dir)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    print(
        f"[results] wrote {output} ({summary['arms']} arms, "
        f"{summary['passed_arms']} integrity-clean)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
