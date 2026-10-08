"""Freeze the human study's files without installing or executing a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

PACKET = Path(__file__).resolve().parent
PERSONAS = {"beginner": "B", "integrator": "I", "scorer-author": "S"}


def pin(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "bytes": path.stat().st_size,
    }


def check_config(config: dict[str, object]) -> None:
    def has_null(value: object) -> bool:
        if value is None:
            return True
        if isinstance(value, dict):
            return any(has_null(item) for item in value.values())
        if isinstance(value, list):
            return any(has_null(item) for item in value)
        return False

    if has_null(config):
        raise ValueError("resolve every null configuration value before freezing")
    if config.get("schema") != "wyat-prospective-human-study-v1":
        raise ValueError("unsupported study schema")
    if config.get("study_phase") not in {"pilot", "confirmation"}:
        raise ValueError("study phase must be pilot or confirmation")
    if config.get("caps_seconds") != {"U2": 1800, "other": 600}:
        raise ValueError(
            "changing caps requires a revised protocol before confirmation"
        )
    if config.get("observed_targets") != {
        "equally_weighted_cell_Q": 0.9,
        "every_cell": 0.8,
    }:
        raise ValueError("preserve the existing prospective ergonomic targets")
    participants = config.get("participants")
    if not isinstance(participants, list):
        raise ValueError("participants must be an anonymous assignment roster")
    minimum = 6 if config["study_phase"] == "pilot" else 30
    counts: Counter[str] = Counter()
    identifiers: set[str] = set()
    for participant in participants:
        if not isinstance(participant, dict) or set(participant) != {"id", "persona"}:
            raise ValueError("participant fields must be only id and persona")
        identifier, persona = participant["id"], participant["persona"]
        if persona not in PERSONAS or not isinstance(identifier, str):
            raise ValueError("unknown persona or invalid anonymous ID")
        if not re.fullmatch(
            PERSONAS[persona] + r"[0-9]{3}", identifier
        ) or identifier.endswith("000"):
            raise ValueError("anonymous IDs must be positive B001/I001/S001-style IDs")
        if identifier in identifiers:
            raise ValueError("duplicate participant ID")
        identifiers.add(identifier)
        counts[persona] += 1
    if any(counts[persona] < minimum for persona in PERSONAS):
        raise ValueError(
            f"this proposal requires at least {minimum} assignments per persona"
        )
    for name in ("artifact_files", "files_to_pin"):
        paths = config.get(name)
        if not isinstance(paths, list) or not paths:
            raise ValueError(f"{name} must list the actual prepared files")
        if name == "artifact_files" and len(paths) != 9:
            raise ValueError("pin all nine supported Kai artifact files")
    if config.get("human_observations") != "NONE":
        raise ValueError("this is a prospective freeze, not a result record")


def freeze(config_path: Path, destination: Path) -> dict[str, object]:
    config = json.loads(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError("configuration must be a JSON object")
    check_config(config)
    wheel = pin(Path(config["wheel_path"]))
    if wheel["sha256"] != config.get("expected_wheel_sha256"):
        raise ValueError("wheel does not match the declared independent handoff")
    artifacts = [pin(Path(path)) for path in config["artifact_files"]]
    if len({item["path"] for item in artifacts}) != len(artifacts):
        raise ValueError("artifact file pins must be distinct")
    files = [pin(Path(path)) for path in config["files_to_pin"]]
    packet_names = (
        "PROTOCOL.md",
        "PARTICIPANT.md",
        "OBSERVER-KEY.md",
        "fixtures.py",
        "freeze.py",
        "observations-template.csv",
        "concept-ledger-template.csv",
        "enrollment-template.csv",
    )
    packet = [pin(PACKET / name) for name in packet_names]
    text = (PACKET / "PARTICIPANT.md").read_text()
    sections = re.split(r"(?m)^## (?=U[0-9]+ )", text)
    cards = {section.split(" ", 1)[0]: "## " + section for section in sections[1:]}
    if set(cards) != {f"U{index}" for index in range(1, 11)}:
        raise ValueError("public task cards are incomplete")
    receipt = {
        "schema": "wyat-prospective-human-study-freeze-v1",
        "frozen_utc": datetime.now(UTC).isoformat(),
        "config": config,
        "wheel": wheel,
        "artifact_files": artifacts,
        "files": files,
        "packet": packet,
        "human_observations": "NONE",
        "real_inference_checked_by_this_tool": False,
    }
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "public").mkdir()
    (destination / "researcher-cards").mkdir()
    (destination / "public" / "INTRO.md").write_text(sections[0])
    shutil.copyfile(PACKET / "fixtures.py", destination / "public" / "fixtures.py")
    for task, card in cards.items():
        (destination / "researcher-cards" / f"{task}.md").write_text(card)
    (destination / "FREEZE.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    receipt = freeze(args.config, args.out)
    config = cast(dict[str, object], receipt["config"])
    print(
        json.dumps(
            {
                "study_id": config["study_id"],
                "freeze_path": str(args.out / "FREEZE.json"),
                "human_observations": "NONE",
            }
        )
    )


if __name__ == "__main__":
    main()
