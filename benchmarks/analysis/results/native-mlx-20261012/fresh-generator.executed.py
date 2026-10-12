"""Freeze additional DEV parity challenges before observing candidate outputs.

Preserve the original frozen records, cases and gates; append disjoint records.
This is a new implementation challenge on reused DEV, not held-out accuracy.
Only tokenizer encoding runs here; no backbone or decision head is loaded.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

import native_mlx_benchmark_20261012 as bench

SELECTION = (
    "Exclude every original frozen record ID. Group remaining valid single-row "
    "public records by (family, task type), sorted lexicographically. Within each "
    "stratum sort by (encoded length, record ID). Round robin strata, choosing "
    "lower median remaining on cycle 0, shortest remaining on cycle 1, longest "
    "remaining on cycle 2, and repeating that three-cycle order until 16 records. "
    "Targets and labels do not enter selection or inference."
)


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    from wyat import tasks
    from wyat.backends.kai.encoding import load_tokenizer

    original = bench.read(args.original)
    bench.check_frozen(original)
    if bench.digest(args.cohort) != original["cohort_sha256"]:
        raise RuntimeError("source cohort differs from original frozen plan")
    if bench.digest(Path(tasks.__file__)) != original["tasks_sha256"]:
        raise RuntimeError("task encoder differs from original frozen plan")
    manifest_path = args.artifact / "native_manifest.json"
    if bench.digest(manifest_path) != original["artifact_manifest_sha256"]:
        raise RuntimeError("artifact manifest differs from original frozen plan")
    manifest = bench.read(manifest_path)
    if manifest["model_id"] != original["model_id"]:
        raise RuntimeError("artifact model identity differs")
    old_ids = {record["id"] for record in original["records"]}
    old_case_ids = {case["id"] for case in original["cases"]}
    if len(old_ids) != len(original["records"]):
        raise RuntimeError("original record IDs are not unique")
    if len(old_case_ids) != len(original["cases"]):
        raise RuntimeError("original case IDs are not unique")
    tokenizer = load_tokenizer(args.artifact)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    seen_source: set[str] = set()
    excluded = []
    for line in args.cohort.read_text().splitlines():
        source = json.loads(line)
        record_id = source["id"]
        if record_id in seen_source:
            raise RuntimeError("duplicate source record ID: " + record_id)
        seen_source.add(record_id)
        if record_id in old_ids:
            continue
        public = source["public_record"]
        try:
            rows = tasks.encode_record(
                tokenizer, public, max_length=manifest["max_length"]
            )
        except ValueError as error:
            excluded.append({"id": record_id, "error": str(error)})
            continue
        if len(rows) != 1:
            raise RuntimeError("source record is not one independent question")
        groups[(source["family"], rows[0].question.type)].append(
            {
                "id": record_id,
                "family": source["family"],
                "public_record": public,
                "encoded": list(map(bench.encoded_value, rows)),
            }
        )
    ranked = {
        key: sorted(
            values, key=lambda item: (len(item["encoded"][0]["ids"]), item["id"])
        )
        for key, values in groups.items()
    }
    selected = []
    cycle = 0
    while len(selected) < 16:
        progress = False
        for key in sorted(ranked):
            values = ranked[key]
            if not values:
                continue
            index = ((len(values) - 1) // 2, 0, len(values) - 1)[cycle % 3]
            selected.append(values.pop(index))
            progress = True
            if len(selected) == 16:
                break
        if not progress:
            raise RuntimeError("fewer than 16 disjoint valid records available")
        cycle += 1
    selected_ids = [record["id"] for record in selected]
    if len(set(selected_ids)) != 16 or old_ids.intersection(selected_ids):
        raise RuntimeError("fresh selection is duplicated or overlaps original")
    new_cases = [
        {"id": f"fresh-{index:02d}", "records": [record["id"]]}
        for index, record in enumerate(selected)
    ]
    if old_case_ids.intersection(case["id"] for case in new_cases):
        raise RuntimeError("new case IDs collide with original plan")
    result = dict(original)
    result.update(
        records=original["records"] + selected,
        cases=original["cases"] + new_cases,
        classification=(
            "implementation parity challenge: original 32 DEV plus 16 disjoint "
            "additional DEV records and original synthetic contracts; "
            "no held-out quality claim"
        ),
        challenge_provenance={
            "parent_plan_path": str(args.original.resolve()),
            "parent_plan_sha256": bench.digest(args.original),
            "selection_script_sha256": bench.digest(Path(__file__)),
            "cohort_sha256": bench.digest(args.cohort),
            "selection_rule": SELECTION,
            "added_record_ids": selected_ids,
            "added_records": 16,
            "old_record_count": len(original["records"]),
            "old_case_count": len(original["cases"]),
            "original_record_overlap": 0,
            "eligible_strata": [list(key) for key in sorted(groups)],
            "excluded_before_selection": excluded,
            "classification": "selection frozen before additional model outputs",
        },
    )
    if (
        result["records"][: len(original["records"])] != original["records"]
        or result["cases"][: len(original["cases"])] != original["cases"]
        or result["quality_gates"] != original["quality_gates"]
        or result["performance_gates"] != original["performance_gates"]
        or result["harness_sha256"] != original["harness_sha256"]
    ):
        raise RuntimeError("original inputs, cases or gates changed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original", "cohort", "artifact", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; choose a fresh path")
    bench.save(args.output, freeze(args))


if __name__ == "__main__":
    main()
