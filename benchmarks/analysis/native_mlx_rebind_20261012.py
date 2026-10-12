"""Create a separate, source-attested local admission copy of the pinned release.

No tensor or tokenizer file changes. The frozen parent is validated by its own
bundled runtime; the new manifest binds the current, semantically equal task
source. This research preparation does not promote or overwrite a release.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from wyat.loading import native_artifact as artifact  # noqa: E402

SOURCE_SHA256 = "de60ea0f7630bf447289d9524472e02c20d0eaadd371363d93aa4c28ab12cccf"


def _ast(source: str) -> str:
    return ast.dump(ast.parse(source), include_attributes=False)


def _definitions(source: str) -> dict[str, str]:
    return {
        node.name: ast.dump(node, include_attributes=False)
        for node in ast.parse(source).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }


def _verify_sources(release: Path) -> dict[str, str]:
    runtime = release / "runtime"
    source_path = runtime / "SOURCE.json"
    if artifact.file_sha256(source_path) != SOURCE_SHA256:
        raise ValueError("release runtime SOURCE identity differs from pinned source")
    source = json.loads(source_path.read_text())
    for name, expected in source["files_sha256"].items():
        if artifact.file_sha256(runtime / name) != expected:
            raise ValueError("bundled runtime source identity differs: " + name)
    originals = {}
    for name in ("wyat/tasks.py", "wyat/loading/native.py"):
        result = subprocess.run(
            ["git", "show", f"{source['commit']}:{name}"],
            cwd=ROOT,
            capture_output=True,
            check=True,
        )
        if (
            hashlib.sha256(result.stdout).hexdigest()
            != source["upstream_files_sha256"][name]
        ):
            raise ValueError("pinned upstream source identity differs: " + name)
        original = result.stdout.decode("utf-8")
        if _ast(original) != _ast((runtime / name).read_text()):
            raise ValueError("attributed release has semantic source changes: " + name)
        originals[name] = original
    if _ast(originals["wyat/tasks.py"]) != _ast((ROOT / "wyat/tasks.py").read_text()):
        raise ValueError("current task source differs semantically from parent")
    # The loader now delegates artifact validation to a framework-free owner.
    # Its tensor loading, adapters, execution and readouts remain exact ASTs.
    original_definitions = _definitions(originals["wyat/loading/native.py"])
    current_definitions = _definitions((ROOT / "wyat/loading/native.py").read_text())
    for name in ("_finite_state", "_Adapter", "NativeModel", "load_native"):
        if current_definitions.get(name) != original_definitions[name]:
            raise ValueError("native Torch execution changed: " + name)
    shared_definitions = _definitions(
        (ROOT / "wyat/loading/native_artifact.py").read_text()
    )
    for name in ("file_sha256", "_positive_integer", "validate_artifact"):
        if shared_definitions.get(name) != original_definitions[name]:
            raise ValueError("native validation changed: " + name)
    return {
        "release_source_sha256": SOURCE_SHA256,
        "parent_tasks_sha256": source["files_sha256"]["wyat/tasks.py"],
        "upstream_tasks_sha256": source["upstream_files_sha256"]["wyat/tasks.py"],
        "parent_loader_sha256": source["files_sha256"]["wyat/loading/native.py"],
        "current_tasks_sha256": artifact.file_sha256(ROOT / "wyat/tasks.py"),
        "current_loader_sha256": artifact.file_sha256(ROOT / "wyat/loading/native.py"),
        "current_artifact_owner_sha256": artifact.file_sha256(
            ROOT / "wyat/loading/native_artifact.py"
        ),
    }


def derive(release: Path, output: Path) -> dict[str, Any]:
    release = release.resolve(strict=True)
    output = output.resolve()
    if output.exists() or release == output or release in output.parents:
        raise ValueError("output must be a new directory outside the frozen release")
    pins = _verify_sources(release)
    parent = release / "model"
    parent_manifest_path = parent / artifact.MANIFEST
    parent_manifest_sha = artifact.file_sha256(parent_manifest_path)
    # Running outside the repository prevents accidental import of its runtime.
    environment = dict(os.environ, PYTHONPATH=str(release / "runtime"))
    code = (
        "import json,sys; from wyat.loading.native import validate_artifact; "
        "print(json.dumps(validate_artifact(sys.argv[1]),sort_keys=True))"
    )
    validated = subprocess.run(
        [sys.executable, "-c", code, str(parent)],
        cwd=release,
        env=environment,
        capture_output=True,
        check=True,
        text=True,
        timeout=120,
    )
    manifest = json.loads(validated.stdout)
    if manifest != json.loads(parent_manifest_path.read_text()):
        raise ValueError("bundled validator returned a different parent manifest")
    current_contract = artifact.artifact_contract()
    if (
        manifest["contract"] | {"tasks_sha256": current_contract["tasks_sha256"]}
        != current_contract
    ):
        raise ValueError(
            "parent/current contract differs beyond verified task attribution"
        )
    # Fresh regular copies avoid links that could mutate the original later.
    output.mkdir(parents=True)
    for name, expected in manifest["files_sha256"].items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parent / name, target)
        if artifact.file_sha256(target) != expected:
            raise ValueError("copied artifact identity differs: " + name)
    manifest["contract"] = current_contract
    manifest["provenance"] = {
        **manifest["provenance"],
        "local_runtime_admission": {
            "parent_manifest_sha256": parent_manifest_sha,
            "parent_model_id": manifest["model_id"],
            "files_unchanged": True,
            "verification": (
                "pinned bundled validator; attributed/upstream task and loader "
                "AST equality; current task AST equality; exact relocated "
                "validator and Torch execution AST equality; unchanged "
                "predictor source identity"
            ),
            "source_sha256": pins,
        },
    }
    (output / artifact.MANIFEST).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    artifact.validate_artifact(output)
    if artifact.file_sha256(parent_manifest_path) != parent_manifest_sha:
        raise ValueError("frozen parent manifest changed during derivation")
    for name, expected in manifest["files_sha256"].items():
        if artifact.file_sha256(parent / name) != expected:
            raise ValueError("frozen parent file changed during derivation: " + name)
    return {
        "model_id": manifest["model_id"],
        "parent_manifest_sha256": parent_manifest_sha,
        "derived_manifest_sha256": artifact.file_sha256(output / artifact.MANIFEST),
        "output": str(output),
        "source_sha256": pins,
        "script_sha256": artifact.file_sha256(Path(__file__)),
        "passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(derive(args.release, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
