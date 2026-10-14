"""Bounded, predeclared DMBench native interoperability assay; no score claim."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, cast

from benchmarks.analysis.evidence import freeze_sources
from wyat import Model
from wyat.interop.dmbench import DMBenchRecipe, as_dmbench, as_dmbench_batch

if TYPE_CHECKING:
    from dmbench.api import Case

RELEASE = "1e028905794498daba40a74cca8d826f965edb661879a725451bc222a3049dc0"
MODEL_ID = "kai:1f5525e01de43a724c71824579ef5195595fba916ce1e02be7bbbbea50f25cee"
RECIPE: DMBenchRecipe = "dmbench-native-choice-v1"
CACHE_BYTES = 512 << 20
UNSUPPORTED = tuple(f"clinc150/val/{index:06d}" for index in range(6))
LENGTH_ERROR = re.compile(
    r"^ValueError: ([0-9]+) tokens exceeds max_length=2048; no truncation$"
)


def object_value(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError("Expected an object")
    return cast(dict[str, object], value)


@dataclass(frozen=True, slots=True)
class Outcome:
    id: str
    answer: str | None
    error: str | None
    probabilities: dict[str, float] | None


def outcomes(report: Mapping[str, object], ids: tuple[str, ...]) -> tuple[Outcome, ...]:
    rows = report.get("predictions")
    if report.get("count") != len(ids) or not isinstance(rows, list):
        raise ValueError("Evaluator did not retain the complete denominator")
    values: list[Outcome] = []
    for raw in rows:
        row = object_value(raw)
        identifier, answer, error = row.get("id"), row.get("answer"), row.get("error")
        if not isinstance(identifier, str) or (
            answer is not None and not isinstance(answer, str)
        ):
            raise ValueError("Invalid evaluator identity or selected answer")
        if error is not None and not isinstance(error, str):
            raise ValueError("Invalid evaluator error")
        probabilities: dict[str, float] | None = None
        if row.get("probabilities") is not None:
            probabilities = {}
            for key, value in object_value(row["probabilities"]).items():
                if type(value) not in {int, float} or not math.isfinite(
                    cast(float, value)
                ):
                    raise ValueError("Invalid evaluator probability")
                probabilities[key] = float(cast(float, value))
        values.append(Outcome(identifier, answer, error, probabilities))
    if tuple(row.id for row in values) != ids:
        raise ValueError("Evaluator changed case IDs, order or count")
    return tuple(values)


def check_reports(
    scalar: Mapping[str, object],
    batch: Mapping[str, object],
    choices: Mapping[str, tuple[str, ...]],
) -> dict[str, object]:
    """Compare admission and public predictions, never source gold or accuracy."""
    ids = tuple(choices)
    first, second = outcomes(scalar, ids), outcomes(batch, ids)
    problems: list[str] = []
    maximum = 0.0
    admitted = unsupported = selected_equal = 0
    for left, right in zip(first, second, strict=True):
        if (left.error is None) != (right.error is None):
            problems.append(f"{left.id}: scalar/batch status mismatch")
        if left.id in UNSUPPORTED:
            valid = True
            for row in (left, right):
                match = LENGTH_ERROR.fullmatch(row.error or "")
                if (
                    match is None
                    or int(match[1]) <= 2048
                    or row.answer is not None
                    or row.probabilities is not None
                ):
                    valid = False
            if valid:
                unsupported += 1
            else:
                problems.append(
                    f"{left.id}: expected intact overlength failure missing"
                )
            continue
        if left.error is not None or right.error is not None:
            problems.append(f"{left.id}: unexpected runtime failure")
            continue
        admitted += 1
        menu = choices[left.id]
        if left.answer != right.answer or left.answer not in menu:
            problems.append(f"{left.id}: selected answer changed or is outside menu")
        else:
            selected_equal += 1
        if any(
            row.probabilities is None or set(row.probabilities) != set(menu)
            for row in (left, right)
        ):
            problems.append(
                f"{left.id}: probability coverage differs from original menu"
            )
            continue
        assert left.probabilities is not None and right.probabilities is not None
        difference = max(
            abs(left.probabilities[key] - right.probabilities[key]) for key in menu
        )
        maximum = max(maximum, difference)
        if difference > 1e-4:
            problems.append(f"{left.id}: probability drift exceeds 1e-4")
    if len(ids) != 30 or unsupported != 6 or admitted != 24:
        problems.append("Expected six unsupported and 24 admitted cases across all 30")
    return {
        "passed": not problems,
        "total": len(ids),
        "expected_unsupported": unsupported,
        "admitted": admitted,
        "selected_equal": selected_equal,
        "max_absolute_probability_drift": maximum,
        "problems": problems,
    }


def read_assay(peer: Path) -> tuple[dict[str, object], tuple[Case, ...]]:
    from dmbench.api import strict_json_loads
    from dmbench.io import load_benchmark

    contract = object_value(
        strict_json_loads(
            (peer / "reports/wyat-native-dev-assay-contract.json").read_text()
        )
    )
    manifest, cases = load_benchmark(peer / "benchmarks/wyat-native-dev-assay")
    ids = [case.request.id for case in cases]
    if manifest != contract.get("manifest") or manifest.get("release_id") != RELEASE:
        raise ValueError("Assay differs from the bound manifest/release")
    if (
        len(cases) != 30
        or ids != contract.get("request_ids")
        or tuple(ids[:6]) != UNSUPPORTED
    ):
        raise ValueError("Assay changed its exact 30-case selection/order")
    if any(
        case.source != "CLINC/CLINC150-full"
        or case.split != "val"
        or len(case.request.choices) != 151
        for case in cases[:6]
    ):
        raise ValueError("Expected six complete original CLINC val menus")
    if any(
        case.source != "authored-requirements-v1" or case.split != "dev"
        for case in cases[6:]
    ):
        raise ValueError("Expected the complete authored development family")
    if sorted({len(case.request.choices) for case in cases}) != contract.get(
        "menu_sizes"
    ):
        raise ValueError("Assay menu sizes changed")
    return contract, cases


def write_receipt(output: Path, result: Mapping[str, object]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)


def assay(model_path: Path, peer: Path, output: Path) -> dict[str, object]:
    """One scalar evaluation followed by explicit batch six on the same model."""
    result: dict[str, object] = {
        "completed": False,
        "passed": False,
        "stage": "preflight",
    }
    try:
        import dmbench
        from dmbench.evaluation import evaluate

        source = Path(dmbench.__file__).resolve().parent
        if source != (peer / "src/dmbench").resolve():
            raise ValueError("DMBench import does not match the declared peer checkout")
        root = Path(__file__).resolve().parents[3]
        paths = {
            "runner": Path(__file__),
            "contract": peer / "reports/wyat-native-dev-assay-contract.json",
            "manifest": peer / "benchmarks/wyat-native-dev-assay/manifest.json",
            "cases": peer / "benchmarks/wyat-native-dev-assay/cases.jsonl",
            "evidence": root / "benchmarks/analysis/evidence.py",
            "adapter": root / "wyat/interop/dmbench.py",
            "backend": root / "wyat/backends/wyat/mlx.py",
            "encoding": root / "wyat/backends/wyat/encoding.py",
            "artifact": root / "wyat/backends/wyat/artifact.py",
            "core": root / "wyat/core.py",
            "loading": root / "wyat/loading/local.py",
            "wire": root / "wyat/wire.py",
            **{
                f"dmbench/{name}": source / name
                for name in (
                    "__init__.py",
                    "api.py",
                    "io.py",
                    "evaluation.py",
                    "adapters/native.py",
                )
            },
        }
        frozen = freeze_sources(paths, output)
        result.update(source_sha256=frozen.hashes(), source_archive=frozen.archive())
        contract, cases = read_assay(peer)
        result.update(
            {
                "schema_version": 1,
                "scope": (
                    "predeclared native-recipe interoperability diagnostic; "
                    "no performance or model-quality claim"
                ),
                "source_sha256": frozen.hashes(),
                "source_archive": frozen.archive(),
                "contract": contract,
                "release_id": RELEASE,
                "recipe": RECIPE,
                "case_ids": [case.request.id for case in cases],
                "model_artifact": str(model_path.resolve()),
                "protocol": {
                    "device": "mlx GPU",
                    "dtype": "float32",
                    "max_length": 2048,
                    "token_cache_size": 0,
                    "compile": False,
                    "cache_limit_bytes": CACHE_BYTES,
                    "order": ["scalar", "explicit_batch6"],
                    "timeout_seconds": 480,
                    "probability_atol": 1e-4,
                    "MLX_ENABLE_TF32": os.environ.get("MLX_ENABLE_TF32"),
                },
                "environment": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "dmbench": dmbench.__version__,
                    "versions": {
                        name: version(name)
                        for name in ("mlx", "mlx-lm", "transformers", "safetensors")
                    },
                },
                "stage": "model_loading",
            }
        )
        write_receipt(output, result)
        if os.environ.get("MLX_ENABLE_TF32") != "0":
            raise ValueError(
                "FP32 assay requires MLX_ENABLE_TF32=0 before framework import"
            )
        import mlx.core as mx

        from wyat.backends.wyat.mlx import MLXWyatScorer

        previous_limit = mx.set_cache_limit(CACHE_BYTES)
        try:
            scorer = MLXWyatScorer.from_directory(
                model_path,
                max_length=2048,
                device="gpu",
                token_cache_size=0,
                compile=False,
            )
            if scorer.model_id != MODEL_ID:
                raise ValueError("Loaded model differs from the pinned FP32 artifact")
            model = Model(scorer, scorer.model_id)
            result["model_id"] = model.model_id
            scalar = as_dmbench(model, recipe=RECIPE)
            result["scalar"] = evaluate(cases, scalar)
            result["stage"] = "scalar_complete"
            write_receipt(output, result)
            result["batch_size_6"] = evaluate(
                cases,
                scalar,
                batch=as_dmbench_batch(model, recipe=RECIPE),
                batch_size=6,
            )
            result["checks"] = check_reports(
                object_value(result["scalar"]),
                object_value(result["batch_size_6"]),
                {
                    case.request.id: tuple(choice.id for choice in case.request.choices)
                    for case in cases
                },
            )
            result["passed"] = object_value(result["checks"])["passed"]
            result["completed"] = True
            result["stage"] = "complete"
        finally:
            mx.set_cache_limit(previous_limit)
    except Exception as error:
        result["completed"] = False
        result["passed"] = False
        result["error"] = f"{type(error).__name__}: {error}"
    write_receipt(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--peer", type=Path, default=Path("/Users/teilomillet/Code/dmbench")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/real/dmbench-native-dev-assay.json"),
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        result = assay(args.model, args.peer, args.output)
        sys.exit(
            0 if result.get("completed") is True and result.get("passed") is True else 1
        )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = (
        str(args.peer.resolve() / "src")
        + os.pathsep
        + environment.get("PYTHONPATH", "")
    )
    environment["MLX_ENABLE_TF32"] = "0"
    write_receipt(
        args.output, {"completed": False, "passed": False, "stage": "worker_start"}
    )
    start = time.perf_counter()
    timed_out = False
    exit_code: int | None = None
    error: str | None = None
    try:
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "benchmarks.real.dmbench.native",
                str(args.model),
                "--peer",
                str(args.peer),
                "--output",
                str(args.output),
                "--worker",
            ],
            env=environment,
            timeout=480,
            check=False,
        )
        exit_code = process.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        error = "Experiment exceeded its 480-second process cap"
    except OSError as exception:
        error = f"{type(exception).__name__}: {exception}"
    if error is not None:
        partial = (
            object_value(json.loads(args.output.read_text()))
            if args.output.exists()
            else {}
        )
        partial.update({"completed": False, "passed": False, "error": error})
        write_receipt(args.output, partial)
    wall = {
        "elapsed_seconds": time.perf_counter() - start,
        "timeout_seconds": 480,
        "timed_out": timed_out,
        "exit_code": exit_code,
        "completed": exit_code == 0 and not timed_out,
    }
    write_receipt(args.output.with_suffix(".wall.json"), wall)
    sys.exit(0 if wall["completed"] else 1)


if __name__ == "__main__":
    main()
