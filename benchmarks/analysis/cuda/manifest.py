"""Freeze the exact transferable assets before remote consumption."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODEL = Path(".models/Decision-2.0-Kai-0.6B")
MODEL_ID = "kai:1f5525e01de43a724c71824579ef5195595fba916ce1e02be7bbbbea50f25cee"
DATA = Path("benchmarks/real/banking77-train-dev-menu6.jsonl")
REFERENCE = Path("benchmarks/real/torch-fp32-reference77.json")
DATA_SHA = "0020d9a3c72a21941ba551cf31f404187f69c6a5dc30299cc0a24d303b904a03"
REFERENCE_SHA = "b8a90b31941b60e793357a8cb360767520da70ec33d2161c1242606962b22664"
INPUT_SHA = "fd5dd942250c6416c4ab50c59a5164a20909d11ead12bf83c505efff6a5556e8"
MODEL_FILES = (
    "LICENSE",
    "tokenizer_config.json",
    "config.json",
    "tokenizer.json",
    "decision_head.safetensors",
    "decision_config.json",
    "MODEL_MANIFEST.json",
    "backbone/model.safetensors",
    "backbone/config.json",
)
PACKAGES = {
    "torch": "2.12.1",
    "transformers": "5.18.0",
    "safetensors": "0.8.0",
    "tokenizers": "0.23.2",
    "numpy": "2.5.3",
    "huggingface-hub": "1.33.0",
    "annotated-doc": "0.0.5",
    "anyio": "4.15.1",
    "certifi": "2026.7.22",
    "click": "8.5.0",
    "filelock": "4.0.11",
    "fsspec": "2026.9.0",
    "h11": "0.16.0",
    "hf-xet": "1.6.0",
    "httpcore": "1.0.9",
    "httpx": "0.28.1",
    "idna": "3.20",
    "markdown-it-py": "4.2.0",
    "mdurl": "0.1.2",
    "packaging": "26.3",
    "pygments": "2.21.0",
    "pyyaml": "6.0.3",
    "regex": "2026.9.29",
    "rich": "15.0.0",
    "shellingham": "1.5.4",
    "tqdm": "4.70.1",
    "typer": "0.27.2",
    "typing-extensions": "4.16.0",
}
RUNTIME = {"python": "3.12.13", "torch_runtime": "2.12.1+cu130", "cuda_build": "13.0"}


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def asset_paths(root: Path = ROOT) -> tuple[Path, ...]:
    helpers = (
        "benchmarks/__init__.py",
        "benchmarks/analysis/__init__.py",
        "benchmarks/analysis/evidence.py",
        "benchmarks/analysis/reference.py",
        "benchmarks/analysis/torch/process.py",
        "docs/analysis/cuda.md",
        "docs/analysis/cuda-run.md",
        "tests/analysis/test_cuda_protocol.py",
    )
    return tuple(
        sorted(
            set(
                [path.relative_to(root) for path in (root / "wyat").rglob("*.py")]
                + [
                    path.relative_to(root)
                    for path in (root / "benchmarks/analysis/cuda").glob("*.py")
                ]
                + [Path(path) for path in helpers]
                + [MODEL / path for path in MODEL_FILES]
                + [DATA, REFERENCE]
            )
        )
    )


def freeze(path: Path, root: Path = ROOT) -> None:
    if digest(root / DATA) != DATA_SHA or digest(root / REFERENCE) != REFERENCE_SHA:
        raise ValueError("Pinned data/reference changed")
    payload = {
        "model_id": MODEL_ID,
        "dataset_sha256": DATA_SHA,
        "reference_sha256": REFERENCE_SHA,
        "input_sha256": INPUT_SHA,
        "versions": PACKAGES,
        "runtime": RUNTIME,
        "files": [
            {
                "path": str(item),
                "sha256": digest(root / item),
                "bytes": (root / item).stat().st_size,
            }
            for item in asset_paths(root)
        ],
    }
    with path.open("x") as target:
        json.dump(payload, target, indent=2, allow_nan=False)
        target.write("\n")
