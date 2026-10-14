"""Export the explicit public source inventory without Git metadata or history."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.package.check import ROOT, required_sources
from scripts.package.privacy import check_files, check_text


def _reject_links(path: Path) -> None:
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError(f"Export paths must not contain symbolic links: {path}")


def verify_export(destination: Path, contents: dict[str, bytes], root: Path) -> None:
    """Require the exact inventory and bytes, and a source snapshot still current."""
    actual = set()
    for path in destination.rglob("*"):
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise ValueError("Export contains symbolic links or special files")
        if path.is_file():
            actual.add(path.relative_to(destination).as_posix())
    if actual != set(contents) or required_sources(root) != set(contents):
        raise ValueError("Export inventory differs from the public source selection")
    for name, raw in contents.items():
        if (destination / name).read_bytes() != raw or (
            root / name
        ).read_bytes() != raw:
            raise ValueError(f"Export or source changed while copying: {name}")


def export_source(destination: Path, root: Path = ROOT) -> Path:
    """Create a fresh tree and sibling checksums, cleaning owned output on failure."""
    destination = destination.absolute()
    root = root.resolve()
    _reject_links(destination)
    destination = destination.resolve()
    names = required_sources(root)
    if destination == root or any(
        destination.is_relative_to(root / name.split("/")[0])
        for name in names
        if "/" in name
    ):
        raise ValueError("Export destination must be outside public source directories")
    if ".git" in destination.parts:
        raise ValueError("Export destination must be outside Git metadata")
    manifest = destination.with_name(destination.name + ".SHA256SUMS")
    _reject_links(manifest)
    if destination.exists() or manifest.exists():
        raise FileExistsError("Export destination and checksum manifest must be new")
    contents = {name: (root / name).read_bytes() for name in sorted(names)}
    check_files({name: root / name for name in names})
    # Scan the snapshot that will actually be copied as well as its source paths.
    for name, raw in contents.items():
        check_text(name, raw)
    checksums = "".join(
        f"{hashlib.sha256(raw).hexdigest()}  {name}\n" for name, raw in contents.items()
    ).encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _reject_links(destination.parent)
    destination.mkdir()  # Exclusive ownership; never reuse or overwrite a tree.
    manifest_created = False
    try:
        for name, raw in contents.items():
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(raw)
            target.chmod((root / name).stat().st_mode & 0o777)
        verify_export(destination, contents, root)
        with manifest.open("xb") as stream:
            manifest_created = True
            stream.write(checksums)
    except BaseException:
        if manifest_created:
            manifest.unlink()
        shutil.rmtree(destination)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    manifest = export_source(arguments.destination)
    print(f"Verified public source: {manifest.with_suffix('')}")
    print(f"SHA256SUMS: {manifest}")


if __name__ == "__main__":
    main()
