"""Replay a saved MPS logits receipt against the CPU FP32 reference.

Pure comparison: no model load, no MPS allocator, no bench lock. Reads the frozen
question set, the CPU FP32 reference and one logits file produced by
``probe_fp32.py`` / ``probe_batching.py`` / ``probe_precision.py``, then prints
(and exits nonzero on) any drift or rank/top-1 change outside the declared
tolerance.

Usage:
    .venv/bin/python scripts/ouro_speed/mps/replay_parity.py
    .venv/bin/python scripts/ouro_speed/mps/replay_parity.py \
        --logits artifacts/ouro-speed-20261021/mps/logits/mps-batching.json \
        --atol 2e-5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mps_common as common  # noqa: E402


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_correlations(payload: dict) -> list[dict]:
    """Normalize any of the probe logits layouts into comparable rows."""
    rows: list[dict] = []
    if "headline" in payload:
        headline = payload["headline"]
        if headline and isinstance(next(iter(headline.values())), dict):
            # batching layout: {name: {strategy: logits}}
            for name, strategies in headline.items():
                for strategy, logits in strategies.items():
                    rows.append(
                        {
                            "scope": "headline",
                            "id": name,
                            "arm": strategy,
                            "logits": logits,
                        }
                    )
        else:
            for name, logits in headline.items():
                rows.append(
                    {
                        "scope": "headline",
                        "id": name,
                        "arm": payload.get("arm", ""),
                        "logits": logits,
                    }
                )
    for scope in ("corpus", "real"):
        if scope not in payload:
            continue
        items = payload[scope]
        if isinstance(items, dict):
            # precision layout: {strategy: [{id, logits}, ...]}
            for arm, records in items.items():
                for record in records:
                    rows.append(
                        {
                            "scope": scope,
                            "id": record["id"],
                            "arm": arm,
                            "logits": record["logits"],
                        }
                    )
        else:
            for record in items:
                rows.append(
                    {
                        "scope": scope,
                        "id": record["id"],
                        "arm": payload.get("arm", ""),
                        "logits": record["logits"],
                    }
                )
    return rows


def reference_rows(reference: dict) -> dict[tuple[str, str], list[float]]:
    out: dict[tuple[str, str], list[float]] = {}
    for name, logits in reference.get("headline_logits", {}).items():
        out[("headline", name)] = logits
    for record in reference.get("corpus", []):
        out[("corpus", record["id"])] = record["logits"]
    for record in reference.get("real", []):
        out[("real", record["id"])] = record["logits"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logits",
        default=str(common.LOGIT_DIR / "mps-fp32-serial.json"),
    )
    parser.add_argument("--atol", type=float, default=common.DECLARED_ATOL)
    parser.add_argument("--reference", default=str(common.reference_path()))
    args = parser.parse_args()

    payload = load_json(Path(args.logits))
    reference = load_json(Path(args.reference))
    if payload.get("input_question_set_sha256") != reference.get("question_set_sha256"):
        raise SystemExit(
            "logits file and CPU reference were produced for different question sets"
        )
    expected = reference_rows(reference)
    rows = collect_correlations(payload)
    failures: list[str] = []
    by_scope: dict[str, list[dict]] = {}
    for row in rows:
        key = (row["scope"], row["id"])
        if key not in expected:
            failures.append(f"no CPU reference for {key}")
            continue
        comparison = common.compare_logits(row["logits"], expected[key])
        comparison["arm"] = row["arm"]
        comparison["id"] = row["id"]
        by_scope.setdefault(row["scope"], []).append(comparison)
        if comparison["max_abs_diff"] > args.atol:
            failures.append(
                f"{row['scope']}/{row['arm']}/{row['id']}: "
                f"max|Δ|={comparison['max_abs_diff']:.3g} > atol={args.atol}"
            )
        if not comparison["rank_equal"]:
            failures.append(
                f"{row['scope']}/{row['arm']}/{row['id']}: rank order changed"
            )
        if not comparison["top1_equal"]:
            failures.append(f"{row['scope']}/{row['arm']}/{row['id']}: top-1 changed")
    summary = {
        "logits_file": args.logits,
        "reference": args.reference,
        "declared_atol": args.atol,
        "scopes": {
            scope: {
                "rows": len(items),
                "global_max_abs_diff": max(item["max_abs_diff"] for item in items),
                "rank_equal_rows": sum(1 for item in items if item["rank_equal"]),
                "top1_equal_rows": sum(1 for item in items if item["top1_equal"]),
            }
            for scope, items in sorted(by_scope.items())
        },
        "failures": failures,
        "within_atol_and_rank_stable": not failures,
    }
    print(json.dumps(summary, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
