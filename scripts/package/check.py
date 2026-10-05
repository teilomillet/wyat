"""Check local release artifacts, rebuild the sdist, and seal their hashes."""

from __future__ import annotations

import argparse
import base64
import configparser
import csv
import hashlib
import io
import os
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from collections import Counter
from dataclasses import dataclass
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import cast

from packaging.markers import Marker
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[2]
RequirementKey = tuple[str, frozenset[str], str, str | None, str]


def text(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("Project metadata must contain text")
    return value


def strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError("Project metadata must contain an array")
    return tuple(text(item) for item in value)


def requirement_key(value: str, extra: str | None = None) -> RequirementKey:
    requirement = Requirement(value)
    if extra is not None:
        prefix = (
            f"({requirement.marker}) and " if requirement.marker is not None else ""
        )
        requirement.marker = Marker(f'{prefix}extra == "{extra}"')
    marker = "" if requirement.marker is None else str(requirement.marker)
    return (
        requirement.name.lower().replace("_", "-"),
        frozenset(requirement.extras),
        str(requirement.specifier),
        requirement.url,
        marker,
    )


@dataclass(frozen=True)
class Project:
    name: str
    version: str
    python: str
    license: str
    extras: tuple[str, ...]
    requirements: tuple[RequirementKey, ...]

    @classmethod
    def read(cls, root: Path) -> Project:
        configuration = tomllib.loads((root / "pyproject.toml").read_text("utf-8"))
        project = cast(dict[str, object], configuration["project"])
        extras = project.get("optional-dependencies", {})
        if not isinstance(extras, dict):
            raise ValueError("Optional dependencies must be a table")
        extras = cast(dict[str, object], extras)
        requirements = [
            requirement_key(item) for item in strings(project.get("dependencies", []))
        ]
        for extra, dependencies in extras.items():
            requirements.extend(
                requirement_key(item, extra) for item in strings(dependencies)
            )
        return cls(
            text(project["name"]),
            text(project["version"]),
            text(project["requires-python"]),
            text(project["license"]),
            tuple(sorted(extras)),
            tuple(requirements),
        )


def metadata(raw: bytes) -> tuple[dict[str, tuple[str, ...]], str]:
    message = BytesParser().parsebytes(raw)
    fields: dict[str, list[str]] = {}
    for key, value in message.items():
        fields.setdefault(key.lower(), []).append(value)
    return (
        {key: tuple(sorted(values)) for key, values in fields.items()},
        cast(str, message.get_payload()),
    )


def check_metadata(raw: bytes, project: Project) -> None:
    fields, _ = metadata(raw)
    expected = {
        "name": (project.name,),
        "version": (project.version,),
        "requires-python": (project.python,),
        "license-expression": (project.license,),
        "provides-extra": project.extras,
        "license-file": ("LICENSE", "NOTICE"),
    }
    for key, value in expected.items():
        if fields.get(key, ()) != value:
            raise ValueError(f"Distribution metadata differs: {key}")
    requirements = tuple(
        requirement_key(item) for item in fields.get("requires-dist", ())
    )
    if Counter(requirements) != Counter(project.requirements):
        raise ValueError("Distribution dependencies differ from pyproject.toml")


def safe_name(name: str) -> str:
    value = name.removesuffix("/")
    if (
        not value
        or "\\" in value
        or "\x00" in value
        or PurePosixPath(value).is_absolute()
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or ":" in value.split("/")[0]
    ):
        raise ValueError(f"Unsafe archive path: {name!r}")
    return value


def forbidden(name: str) -> bool:
    parts = PurePosixPath(name).parts
    return any(
        part in {".git", ".venv", ".models", "models", "artifacts", "__pycache__"}
        or part.endswith((".trace", ".gputrace"))
        for part in parts
    ) or name.endswith(
        (".safetensors", ".pt", ".pth", ".gguf", ".onnx", ".raw.xml", ".log")
    )


def wheel_files(path: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    names: set[str] = set()
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            name = safe_name(member.filename)
            if name in names:
                raise ValueError("Duplicate wheel archive path")
            names.add(name)
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError("Wheel contains a symbolic link")
            if not member.is_dir():
                files[name] = archive.read(member)
    return files


def source_files(path: Path, project: Project) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    names: set[str] = set()
    prefix = f"{project.name}-{project.version}"
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            name = safe_name(member.name)
            if name in names:
                raise ValueError("Duplicate source archive path")
            names.add(name)
            if name != prefix and not name.startswith(prefix + "/"):
                raise ValueError("Source archive has an unexpected root")
            if not member.isdir() and not member.isfile():
                raise ValueError(
                    "Source archive must contain regular files and directories"
                )
            relative = name.removeprefix(prefix + "/")
            if forbidden(relative):
                raise ValueError(f"Forbidden source artifact: {relative}")
            if member.isfile():
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError("Unreadable source archive member")
                with stream:
                    files[relative] = stream.read()
    return files


def runtime_files(root: Path) -> dict[str, bytes]:
    paths = sorted((root / "wyat").rglob("*.py")) + [root / "wyat/py.typed"]
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in paths}


def check_record(files: dict[str, bytes], name: str) -> None:
    rows = list(csv.reader(io.StringIO(files[name].decode("utf-8"))))
    recorded: set[str] = set()
    for row in rows:
        if len(row) != 3 or row[0] in recorded or row[0] not in files:
            raise ValueError("Invalid wheel RECORD entry")
        path, digest, size = row
        recorded.add(path)
        if path == name:
            if digest or size:
                raise ValueError("Wheel RECORD must not hash itself")
            continue
        expected = base64.urlsafe_b64encode(hashlib.sha256(files[path]).digest())
        if digest != "sha256=" + expected.rstrip(b"=").decode("ascii"):
            raise ValueError("Wheel RECORD digest mismatch")
        if size != str(len(files[path])):
            raise ValueError("Wheel RECORD size mismatch")
    if recorded != set(files):
        raise ValueError("Wheel RECORD omits package files")


def check_wheel(path: Path, root: Path, project: Project) -> dict[str, bytes]:
    files = wheel_files(path)
    expected = runtime_files(root)
    runtime = {name: raw for name, raw in files.items() if name.startswith("wyat/")}
    if runtime != expected:
        raise ValueError("Wheel runtime resources differ from the source checkout")
    info = f"{project.name}-{project.version}.dist-info"
    allowed = {
        f"{info}/{name}"
        for name in (
            "METADATA",
            "WHEEL",
            "RECORD",
            "entry_points.txt",
            "licenses/LICENSE",
            "licenses/NOTICE",
        )
    }
    if set(files) != set(expected) | allowed:
        raise ValueError("Wheel contains unexpected or missing package files")
    check_metadata(files[f"{info}/METADATA"], project)
    for name in ("LICENSE", "NOTICE"):
        if files[f"{info}/licenses/{name}"] != (root / name).read_bytes():
            raise ValueError(f"Wheel attribution differs: {name}")
    entrypoints = configparser.ConfigParser()
    entrypoints.read_string(files[f"{info}/entry_points.txt"].decode("utf-8"))
    if dict(entrypoints["console_scripts"]) != {"wyat": "wyat.cli:main"}:
        raise ValueError("Wheel command entry point differs")
    check_record(files, f"{info}/RECORD")
    return files


def required_sources(root: Path) -> set[str]:
    names = {
        "pyproject.toml",
        "uv.lock",
        "README.md",
        "CONTRIBUTING.md",
        "LICENSE",
        "NOTICE",
    }
    for folder in (
        "wyat",
        "examples",
        "docs",
        "scripts",
        ".github",
        "tests",
        "benchmarks",
    ):
        for path in (root / folder).rglob("*"):
            name = path.relative_to(root).as_posix()
            if path.is_file() and not forbidden(name) and path.suffix != ".pyc":
                names.add(name)
    return names


def check_source(path: Path, root: Path, project: Project) -> dict[str, bytes]:
    files = source_files(path, project)
    if not required_sources(root) <= set(files):
        raise ValueError("Source distribution omits required project files")
    for name, raw in files.items():
        if name == "PKG-INFO":
            check_metadata(raw, project)
            continue
        source = root / name
        if source.is_symlink() or not source.is_file() or source.read_bytes() != raw:
            raise ValueError(f"Source distribution differs from the checkout: {name}")
    if "PKG-INFO" not in files:
        raise ValueError("Source distribution omits PKG-INFO")
    return files


def distributions(directory: Path) -> tuple[Path, Path]:
    wheels = sorted(directory.glob("*.whl"))
    sources = sorted(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        raise ValueError("Expected exactly one wheel and one source distribution")
    allowed = {
        wheels[0],
        sources[0],
        directory / "SHA256SUMS",
        directory / ".gitignore",
    }
    if any(path not in allowed or path.is_symlink() for path in directory.iterdir()):
        raise ValueError("Distribution directory contains unexpected artifacts")
    return wheels[0], sources[0]


def seal_hashes(directory: Path, paths: tuple[Path, Path]) -> None:
    content = "".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
        for path in sorted(paths)
    ).encode("ascii")
    manifest = directory / "SHA256SUMS"
    if manifest.exists():
        if manifest.is_symlink() or manifest.read_bytes() != content:
            raise ValueError("SHA256SUMS does not match the checked distributions")
    else:
        with manifest.open("xb") as stream:
            stream.write(content)


def run(command: tuple[str, ...], cwd: Path, timeout: int = 120) -> None:
    """Bound each owned process group, including build backend descendants."""
    child = subprocess.Popen(command, cwd=cwd, start_new_session=True)
    try:
        status = child.wait(timeout=timeout)
        if status:
            raise subprocess.CalledProcessError(status, command)
    finally:
        try:
            if os.name == "posix":
                os.killpg(child.pid, signal.SIGKILL)
            elif child.poll() is None:
                child.kill()
        except ProcessLookupError:
            pass
        child.wait()


SMOKE = """
import sys
from collections.abc import Sequence
sys.path.insert(0, sys.argv[1])
import wyat
assert wyat.__file__.startswith(sys.argv[1] + "/")
class Scorer:
    def score_many(
        self, requests: Sequence[wyat.Request]
    ) -> tuple[tuple[float, ...], ...]:
        return tuple((0.0, 2.0, 2.0) for _ in requests)
request = wyat.Request(query="Choose", candidates=(
    wyat.Candidate(id="a", text="First"),
    wyat.Candidate(id="b", text="Second"),
    wyat.Candidate(id="c", text="Third"),
))
model = wyat.Model(Scorer(), model_id="wheel-smoke")
decision = model.decide(request)
assert decision.scores == (0.0, 2.0, 2.0)
assert decision.ranking == ("b", "c", "a") and decision.selected == "b"
assert model.decide_many((request, request)) == (decision, decision)
assert not {"torch", "transformers", "mlx", "fastapi", "dmbench"} & sys.modules.keys()
from wyat.cli import main
main(["--help"])
"""


def smoke(wheel: Path, cwd: Path) -> None:
    run(
        (sys.executable, "-I", "-S", "-c", SMOKE, str(wheel.resolve())), cwd, timeout=30
    )


def check(directory: Path, root: Path = ROOT, tag: str | None = None) -> None:
    project = Project.read(root)
    if tag is not None and tag != "v" + project.version:
        raise ValueError("Release tag must match the project version exactly")
    wheel, source = distributions(directory)
    original = check_wheel(wheel, root, project)
    contents = check_source(source, root, project)
    run(
        (sys.executable, "-m", "twine", "check", "--strict", str(wheel), str(source)),
        root,
    )
    with tempfile.TemporaryDirectory(prefix="wyat-package-") as temporary:
        workspace = Path(temporary)
        smoke(wheel, workspace)
        extracted = workspace / "source"
        extracted.mkdir()
        for name, raw in contents.items():
            path = extracted / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        output = workspace / "rebuilt"
        run(
            (
                "uv",
                "--cache-dir",
                str(workspace / "cache"),
                "build",
                "--wheel",
                "--offline",
                "--no-build-isolation",
                "--python",
                sys.executable,
                "--out-dir",
                str(output),
            ),
            extracted,
        )
        rebuilt = tuple(output.glob("*.whl"))
        if len(rebuilt) != 1:
            raise ValueError("Source rebuild must produce exactly one wheel")
        actual = check_wheel(rebuilt[0], root, project)
        for name, raw in original.items():
            if name.endswith("/RECORD"):
                continue  # Each RECORD has already been checked against its own bytes.
            if name.endswith("/METADATA"):
                equal = metadata(raw) == metadata(actual[name])
            else:
                equal = raw == actual[name]
            if not equal:
                raise ValueError(f"Source rebuild differs: {name}")
        smoke(rebuilt[0], workspace)
    seal_hashes(directory, (wheel, source))
    print(f"Checked {wheel.name} and {source.name}; SHA256SUMS verified", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--tag")
    arguments = parser.parse_args()
    check(
        cast(Path, arguments.directory).resolve(), tag=cast(str | None, arguments.tag)
    )


if __name__ == "__main__":
    main()
