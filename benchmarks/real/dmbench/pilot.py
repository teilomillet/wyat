"""Exercise actual DMBench scalar and batch evaluation on four real dev queries."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import dmbench
from dmbench.api import Case, Choice
from dmbench.api import Request as DMBenchRequest
from dmbench.evaluation import evaluate

import wyat
from wyat.evaluation import case_from_dict
from wyat.interop.dmbench import as_dmbench, as_dmbench_batch

RECIPE = "dmbench-verbatim-choice-v1"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pilot(model_path: Path, dataset_path: Path) -> dict[str, object]:
    # Fix the prefix before any inference. Expected answers exist only in Cases.
    selected = tuple(
        case_from_dict(json.loads(line))
        for line in dataset_path.read_text(encoding="utf-8").splitlines()[:4]
    )
    if len(selected) != 4:
        raise ValueError("DMBench native pilot requires the declared first four cases")
    cases = tuple(
        Case(
            DMBenchRequest(
                sample.id,
                "banking77/choice",
                sample.request.query,
                tuple(
                    Choice(choice.id, choice.text)
                    for choice in sample.request.candidates
                ),
            ),
            sample.expected,
            hashlib.sha256(
                " ".join(sample.request.query.casefold().split()).encode()
            ).hexdigest(),
            "PolyAI/BANKING77/train",
            "development",
        )
        for sample in selected
    )
    model = wyat.load(model_path, backend="mlx", device="gpu", token_cache_size=0)
    scalar, batch = as_dmbench(model), as_dmbench_batch(model)
    scalar_report = evaluate(cases, scalar)
    batch_report = evaluate(cases, scalar, batch=batch, batch_size=2)
    source = Path(dmbench.__file__).resolve().parent
    return {
        "schema_version": 1,
        "scope": "bounded native-engine interoperability exercise; not a performance "
        "comparison or official quality score",
        "model_id": model.model_id,
        "model_artifact": str(model_path.resolve()),
        "input_dataset_sha256": digest(dataset_path),
        "case_ids": [sample.id for sample in selected],
        "recipe": RECIPE,
        "input_contract": {
            "text": "original selected query verbatim",
            "choices": "original IDs, text and order",
            "context": "empty; source assay's classification instruction omitted",
            "expected": "independent dataset annotation in DMBench Case only",
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "device": "mlx GPU",
            "dtype": "float32",
            "compile": False,
            "token_cache_size": 0,
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("mlx", "mlx-lm", "transformers", "safetensors")
            },
            "dmbench_version": dmbench.__version__,
        },
        "source_sha256": {
            "dmbench/api.py": digest(source / "api.py"),
            "dmbench/evaluation.py": digest(source / "evaluation.py"),
            "dmbench/__init__.py": digest(source / "__init__.py"),
            "wyat/interop/dmbench.py": digest(Path("wyat/interop/dmbench.py")),
            "reproducer": digest(Path(__file__)),
        },
        "scalar": scalar_report,
        "batch_size_2": batch_report,
        "limits": [
            "One execution per condition; no warmup or repeated timing protocol.",
            "Native model load excluded by DMBench inference timing.",
            "Both conditions share the same loaded model; scalar executes first.",
            "Four training-source queries; released model exposure is unknown.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("benchmarks/real/banking77-train-dev-menu6.jsonl"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/real/dmbench-native-pilot.json")
    )
    args = parser.parse_args()
    result = pilot(args.model, args.dataset)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "model_id": result["model_id"],
                "case_ids": result["case_ids"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
