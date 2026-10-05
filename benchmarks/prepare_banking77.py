"""Build a pinned labeled development assay from BANKING77 train, never test."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"
SOURCE_SHA256 = "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b"
SOURCE_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
    f"{REVISION}/banking_data/train.csv"
)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prepare(source: Path, output: Path, *, menu_size: int = 6) -> Path:
    """One unique real utterance per intent; deterministic selection and negatives."""
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("BANKING77 training source does not match pinned SHA-256")
    with source.open(encoding="utf-8", newline="") as stream:
        source_rows = list(csv.DictReader(stream))
    labels = sorted({row["category"] for row in source_rows})
    if len(source_rows) != 10003 or len(labels) != 77:
        raise ValueError("Unexpected pinned BANKING77 train inventory")
    if not 2 <= menu_size <= len(labels):
        raise ValueError("menu_size must be between 2 and 77")
    selected: list[tuple[int, str, str]] = []
    seen_groups: set[str] = set()
    for label in labels:
        options = sorted(
            (
                (_hash(row["text"]), index, row["text"])
                for index, row in enumerate(source_rows)
                if row["category"] == label
            )
        )
        for _, index, text in options:
            group = _hash(" ".join(text.casefold().split()))
            if group not in seen_groups:
                selected.append((index, label, text))
                seen_groups.add(group)
                break
        else:
            raise ValueError(f"No unique development utterance for {label}")
    cases: list[dict[str, object]] = []
    metadata: list[dict[str, object]] = []
    for index, label, text in selected:
        negatives = sorted(
            (other for other in labels if other != label),
            key=lambda other: _hash(f"wyat-banking77-menu-v1:{index}:{other}"),
        )[: menu_size - 1]
        slot = int(_hash(f"wyat-banking77-slot-v1:{index}"), 16) % menu_size
        menu = negatives[:slot] + [label] + negatives[slot:]
        identifier = f"banking77-train-{index:05d}"
        cases.append(
            {
                "id": identifier,
                "request": {
                    "query": text,
                    "context": "Classify the customer's banking support request into "
                    "exactly one of the supplied intents.",
                    "task": "choice",
                    "candidates": [
                        {"id": item, "text": item.replace("_", " ")} for item in menu
                    ],
                },
                "expected": [label],
            }
        )
        metadata.append(
            {
                "id": identifier,
                "source_row_zero_based": index,
                "source_text_sha256": _hash(text),
                "group_id": _hash(" ".join(text.casefold().split())),
                "label": label,
                "positive_slot_zero_based": slot,
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    dataset = output / f"banking77-train-dev-menu{menu_size}.jsonl"
    rendered = "".join(
        json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n" for case in cases
    )
    dataset.write_text(rendered, encoding="utf-8")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "purpose": "runtime development assay; not an official BANKING77 score",
        "source_url": SOURCE_URL,
        "source_revision": REVISION,
        "source_sha256": SOURCE_SHA256,
        "source_split": "train",
        "source_rows": len(source_rows),
        "license": "CC-BY-4.0",
        "attribution": "PolyAI; Casanueva et al. (2020), Efficient Intent Detection "
        "with Dual Sentence Encoders, https://arxiv.org/abs/2003.04807",
        "selection": "one lowest text-SHA256 unique normalized utterance per "
        "intent, tie-break by source row; labels sorted lexicographically",
        "candidate_recipe": "wyat-banking77-menu-v1: hashed source-row/negative "
        "intent; positive inserted at hashed slot; underscores changed to spaces",
        "menu_size": menu_size,
        "total": len(cases),
        "dataset_sha256": _hash(rendered),
        "groups": metadata,
        "limits": [
            "Training data exposure in the released model is unknown.",
            "Menus contain the labeled intent plus sampled negatives.",
            "Source annotations may contain label errors; no relabeling performed.",
            "No official test or reserved Kayak confirmation examples accessed.",
        ],
    }
    dataset.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="pinned official train.csv")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/real"))
    parser.add_argument("--menu-size", type=int, default=6)
    args = parser.parse_args()
    print(prepare(args.source, args.output, menu_size=args.menu_size))


if __name__ == "__main__":
    main()
