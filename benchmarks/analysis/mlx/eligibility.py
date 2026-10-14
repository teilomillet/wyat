"""CPU-only exact FP16 eligibility screen; no framework/model initialization."""

from __future__ import annotations

import hashlib
import json
import math
import struct
import subprocess
import sys
import time
from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from benchmarks.analysis.evidence import freeze_sources
from wyat.backends.wyat.artifact import load_artifact, positive_integer
from wyat.wire import decode_json

type Role = Literal["gate_proj", "up_proj", "down_proj"]
ROLES: tuple[Role, ...] = ("gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True, slots=True)
class Arm:
    name: str
    layers: tuple[int, ...]
    roles: tuple[Role, ...]
    diagnostic_only: bool = False

    def names(self) -> tuple[str, ...]:
        return tuple(
            f"layers.{layer}.mlp.{role}.weight"
            for layer in self.layers
            for role in self.roles
        )


def declared_arms(layers: int) -> tuple[Arm, ...]:
    count = positive_integer(layers, "layers")
    all_layers = tuple(range(count))
    return (
        Arm("all-down", all_layers, ("down_proj",)),
        Arm("all-gate-up", all_layers, ("gate_proj", "up_proj")),
        Arm("all-mlp", all_layers, ROLES),
        Arm("first4-mlp", all_layers[:4], ROLES),
        Arm("last4-mlp", all_layers[-4:], ROLES),
        Arm("last1-mlp", all_layers[-1:], ROLES, diagnostic_only=True),
    )


@dataclass(frozen=True, slots=True)
class TensorSpan:
    name: str
    shape: tuple[int, int]
    offset: int
    elements: int


def projection_spans(
    path: Path, *, layers: int, hidden: int, intermediate: int
) -> tuple[TensorSpan, ...]:
    """Validate each selected BF16 tensor's exact shape and disjoint byte span."""
    for name, dimension in (
        ("layers", layers),
        ("hidden", hidden),
        ("intermediate", intermediate),
    ):
        positive_integer(dimension, name)
    size = path.stat().st_size
    with path.open("rb") as stream:
        raw_size = stream.read(8)
        if len(raw_size) != 8:
            raise ValueError("Truncated Safetensors length")
        header_size = struct.unpack("<Q", raw_size)[0]
        if not 0 < header_size <= min(8 << 20, size - 8):
            raise ValueError("Invalid Safetensors header length")
        header = decode_json(stream.read(header_size))
    if not isinstance(header, dict):
        raise ValueError("Safetensors header must be an object")
    origin = 8 + header_size
    spans: list[TensorSpan] = []
    for layer in range(layers):
        for role in ROLES:
            name = f"layers.{layer}.mlp.{role}.weight"
            entry = header.get(name)
            shape = (
                (hidden, intermediate)
                if role == "down_proj"
                else (intermediate, hidden)
            )
            if not isinstance(entry, dict) or entry.get("dtype") != "BF16":
                raise ValueError(f"Expected selected BF16 projection: {name}")
            actual_shape, offsets = entry.get("shape"), entry.get("data_offsets")
            if (
                not isinstance(actual_shape, list)
                or any(type(value) is not int for value in actual_shape)
                or actual_shape != list(shape)
                or not isinstance(offsets, list)
                or len(offsets) != 2
                or any(type(value) is not int for value in offsets)
            ):
                raise ValueError(f"Invalid selected projection shape/span: {name}")
            start, end = offsets
            # Narrowed JSON integer offsets are checked again for static typing.
            if type(start) is not int or type(end) is not int:
                raise ValueError(f"Invalid offsets: {name}")
            elements = math.prod(shape)
            if start < 0 or end - start != elements * 2 or origin + end > size:
                raise ValueError(f"Selected projection span exceeds data: {name}")
            spans.append(TensorSpan(name, shape, origin + start, elements))
    ordered = sorted(spans, key=lambda item: item.offset)
    for left, right in zip(ordered, ordered[1:], strict=False):
        if left.offset + left.elements * 2 > right.offset:
            raise ValueError("Selected projection byte spans overlap")
    return tuple(spans)


@dataclass(frozen=True, slots=True)
class TensorEligibility:
    name: str
    shape: tuple[int, int]
    elements: int
    source_nonfinite: int
    mismatches: int
    overflow: int
    underflow_to_zero: int
    source_subnormal: int
    max_finite_absolute_error: float

    @property
    def eligible(self) -> bool:
        return self.source_nonfinite == 0 and self.mismatches == 0


def scan_tensor(
    path: Path, span: TensorSpan, *, chunk_elements: int = 1 << 18
) -> TensorEligibility:
    """Decode BF16 exactly, then compare FP16→FP32 values in bounded chunks.

    Overflow is reported separately: its absolute error is infinite, so the
    finite maximum alone is not an upper bound when overflow is nonzero.
    """
    positive_integer(chunk_elements, "chunk_elements")
    nonfinite = mismatches = overflow = underflow = subnormal = 0
    maximum = 0.0
    with path.open("rb") as stream:
        stream.seek(span.offset)
        for start in range(0, span.elements, chunk_elements):
            count = min(chunk_elements, span.elements - start)
            data = stream.read(count * 2)
            if len(data) != count * 2:
                raise ValueError(f"Truncated projection data: {span.name}")
            bits = np.frombuffer(data, dtype="<u2")
            original = (bits.astype(np.uint32) << 16).view(np.float32)
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                restored = original.astype(np.float16).astype(np.float32)
            finite = np.isfinite(original)
            restored_finite = np.isfinite(restored)
            nonfinite += int(np.count_nonzero(~finite))
            mismatches += int(np.count_nonzero(original != restored))
            overflow += int(np.count_nonzero(finite & ~restored_finite))
            underflow += int(
                np.count_nonzero(finite & (original != 0) & (restored == 0))
            )
            subnormal += int(
                np.count_nonzero(((bits & 0x7F80) == 0) & ((bits & 0x7F) != 0))
            )
            comparable = finite & restored_finite
            if np.any(comparable):
                maximum = max(
                    maximum,
                    float(np.max(np.abs(original[comparable] - restored[comparable]))),
                )
    return TensorEligibility(
        span.name,
        span.shape,
        span.elements,
        nonfinite,
        mismatches,
        overflow,
        underflow,
        subnormal,
        maximum,
    )


def arm_results(
    arms: tuple[Arm, ...], tensors: tuple[TensorEligibility, ...]
) -> list[dict[str, object]]:
    by_name = {tensor.name: tensor for tensor in tensors}
    if len(by_name) != len(tensors):
        raise ValueError("Duplicate tensor eligibility result")
    results: list[dict[str, object]] = []
    for arm in arms:
        selected = [by_name[name] for name in arm.names()]
        results.append(
            {
                **asdict(arm),
                "eligible": all(item.eligible for item in selected),
                "selected_elements": sum(item.elements for item in selected),
                "ineligible_tensors": [
                    item.name for item in selected if not item.eligible
                ],
            }
        )
    return results


def study(model: Path, output: Path) -> int:
    if output.exists():
        raise FileExistsError("Preserve existing eligibility receipt")
    output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[3]
    frozen = freeze_sources(
        {
            "runner": Path(__file__),
            "artifact": root / "wyat/backends/wyat/artifact.py",
            "wire": root / "wyat/wire.py",
            "numpy": Path(np.__file__),
            "tests": root / "tests/analysis/test_mlx_eligibility.py",
            "declaration": root / "docs/analysis/selective-precision.md",
        },
        output,
    )
    receipt: dict[str, object] = {
        "schema_version": 1,
        "complete": False,
        "any_eligible": False,
        "source_hashes": frozen.hashes(),
        "source_archive": frozen.archive(),
        "numpy_version": np.__version__,
        "policy": (
            "finite exact BF16→FP32→FP16→FP32; selected MLP tensors only; "
            "no framework/model initialized"
        ),
    }
    started = time.perf_counter()
    try:
        artifact = load_artifact(model, 512)
        path = artifact.root / "backbone/model.safetensors"
        dimensions = {
            name: positive_integer(artifact.config.get(name), name)
            for name in ("num_hidden_layers", "hidden_size", "intermediate_size")
        }
        arms = declared_arms(dimensions["num_hidden_layers"])
        with path.open("rb") as stream:
            checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        receipt.update(
            model_id=artifact.model_id,
            arms=[asdict(arm) for arm in arms],
            checkpoint_sha256=checkpoint_hash,
        )
        spans = projection_spans(
            path,
            layers=dimensions["num_hidden_layers"],
            hidden=dimensions["hidden_size"],
            intermediate=dimensions["intermediate_size"],
        )
        tensors: list[TensorEligibility] = []
        for span in spans:
            tensors.append(scan_tensor(path, span))
            receipt["tensors"] = [
                {**asdict(item), "eligible": item.eligible} for item in tensors
            ]
            output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        results = arm_results(arms, tuple(tensors))
        with path.open("rb") as stream:
            final_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        receipt["checkpoint_after_sha256"] = final_hash
        if final_hash != checkpoint_hash:
            raise ValueError("Checkpoint changed during parameter scan")
        receipt.update(
            arm_results=results,
            any_eligible=any(result["eligible"] is True for result in results),
            complete=True,
        )
    except Exception as exc:
        receipt.update(
            error=f"{type(exc).__name__}: {exc}", complete=False, any_eligible=False
        )
    finally:
        receipt["elapsed_seconds"] = time.perf_counter() - started
        output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return 0 if receipt["complete"] and receipt["any_eligible"] else 1


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    arguments = parser.parse_args()
    model, output = Path(arguments.model), Path(arguments.output)
    if arguments.worker:
        return study(model, output)
    return launch(model, output)


def launch(model: Path, output: Path) -> int:
    """Bound only our owned CPU scan and preserve timeout/error as non-passes."""
    wall = output.with_suffix(".wall.json")
    if output.exists() or wall.exists():
        raise FileExistsError("Preserve existing eligibility receipts")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "benchmarks.analysis.mlx.eligibility",
        "--worker",
        "--model",
        str(model.resolve()),
        "--output",
        str(output.resolve()),
    ]
    receipt: dict[str, object] = {
        "command": command,
        "timeout_seconds": 120,
        "completed": False,
        "timed_out": False,
    }
    wall.write_text(json.dumps(receipt, indent=2) + "\n")
    start = time.perf_counter()
    try:
        result = subprocess.run(command, check=False, timeout=120)
        code = result.returncode
        receipt["completed"] = True
    except subprocess.TimeoutExpired:
        receipt["timed_out"] = True
        code = 124
    except OSError as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        code = 1
    receipt.update(exit_code=code, elapsed_seconds=time.perf_counter() - start)
    wall.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
