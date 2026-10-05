"""Small immutable source snapshots and untimed process observations."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SourceFile:
    name: str
    path: str
    sha256: str
    text: str


@dataclass(frozen=True, slots=True)
class FrozenSources:
    files: tuple[SourceFile, ...]
    archive_path: str
    archive_sha256: str

    def hashes(self) -> dict[str, str]:
        return {source.name: source.sha256 for source in self.files}

    def archive(self) -> dict[str, str]:
        return {"path": self.archive_path, "sha256": self.archive_sha256}


def freeze_sources(paths: Mapping[str, Path], output: Path) -> FrozenSources:
    """Read declared small source files once and retain their exact UTF-8 bytes.

    A content-addressed exclusive-create archive is never overwritten. The
    snapshot describes on-disk source at setup; it is not checkpoint storage
    or proof that external dependencies match those bytes.
    """
    sources: list[SourceFile] = []
    for name, path in sorted(paths.items()):
        data = path.read_bytes()
        sources.append(
            SourceFile(
                name, str(path), hashlib.sha256(data).hexdigest(), data.decode("utf-8")
            )
        )
    payload = (
        json.dumps(
            {"schema_version": 1, "files": [asdict(source) for source in sources]},
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    archive_hash = hashlib.sha256(payload).hexdigest()
    archive = output.with_suffix(f".sources-{archive_hash[:16]}.json")
    archive.parent.mkdir(parents=True, exist_ok=True)
    try:
        with archive.open("xb") as stream:
            stream.write(payload)
    except FileExistsError:
        if archive.read_bytes() != payload:
            raise ValueError(
                "Existing source archive differs from its identity"
            ) from None
    return FrozenSources(tuple(sources), str(archive), archive_hash)


@dataclass(frozen=True, slots=True)
class ProcessDiagnostic:
    wall_time_ns: int
    perf_counter_ns: int
    user_seconds: float
    system_seconds: float
    minor_faults: int
    major_faults: int
    voluntary_context_switches: int
    involuntary_context_switches: int
    load_average: tuple[float, float, float] | None
    load_average_error: str | None
    collection_elapsed_ns: int
    device_memory_bytes: dict[str, int] | None = None


def process_diagnostic(
    *, device_memory_bytes: Mapping[str, int] | None = None
) -> ProcessDiagnostic:
    """Read cumulative own-process counters; no causal interpretation is implied."""
    start = time.perf_counter_ns()
    wall = time.time_ns()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    load: tuple[float, float, float] | None = None
    error: str | None = None
    try:
        load = os.getloadavg()
    except (AttributeError, OSError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    return ProcessDiagnostic(
        wall,
        start,
        usage.ru_utime,
        usage.ru_stime,
        usage.ru_minflt,
        usage.ru_majflt,
        usage.ru_nvcsw,
        usage.ru_nivcsw,
        load,
        error,
        time.perf_counter_ns() - start,
        None if device_memory_bytes is None else dict(device_memory_bytes),
    )
