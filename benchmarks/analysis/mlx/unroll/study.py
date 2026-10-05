"""Compare maintained, frozen64 and explicitly unrolled64 on exact prior inputs."""

from __future__ import annotations

import hashlib
import itertools
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

import mlx.core as mx
import numpy as np
from safetensors.numpy import load, save_file

from benchmarks.analysis.evidence import freeze_sources
from benchmarks.analysis.mlx.endpoints import memory
from benchmarks.analysis.mlx.mixed.capture import detached_float32, projection_quality
from benchmarks.analysis.mlx.mixed.kernel import ProjectionKernel as OriginalKernel
from benchmarks.analysis.mlx.mixed.study import (
    SHAPES,
    output_digest,
    restored_cache,
    save,
)
from benchmarks.analysis.mlx.tile.admit import ROOT, digest, object_value
from benchmarks.analysis.mlx.tile.kernel import ProjectionKernel as FrozenKernel
from benchmarks.analysis.mlx.tile.study import Array, Operation, read_arrays
from wyat.evaluation import percentile

from .run import PINNED, Context

ARMS = ("maintained", "tile64-bf16", "unroll64-bf16")
ORDERS = tuple(itertools.permutations(ARMS))
CONTROL = "unroll64-fp32"


def comparison(blocks: list[dict[str, object]], baseline: str) -> dict[str, object]:
    ratios: list[float] = []
    for block, expected in zip(blocks, ORDERS, strict=True):
        if tuple(cast(list[str], block["order"])) != expected:
            raise ValueError("Balanced execution order changed")
        timings = cast(dict[str, list[int]], block["dispatch_ns"])
        if set(timings) != set(ARMS) or any(
            len(values) != 10
            or any(type(value) is not int or value <= 0 for value in values)
            for values in timings.values()
        ):
            raise ValueError("Requires all30 positive timings per block")
        ratios.append(sum(timings[baseline]) / sum(timings[ARMS[2]]))
    median, p05 = percentile(ratios, 0.5), percentile(ratios, 0.05)
    report: dict[str, object] = {
        "paired_ratios": ratios,
        "paired_median": median,
        "paired_p05": p05,
    }
    if baseline == ARMS[0]:
        report["passed"] = median >= 1.05 and p05 > 1
    return report


def measure(
    calls: Mapping[str, Operation],
    cold: Mapping[str, str],
    result: dict[str, object],
    progress: Callable[[], None],
) -> None:
    """Fresh synchronized dispatches; output observations occur after each timer."""
    warm: dict[str, str] = {}
    for arm in ARMS:
        y = calls[arm]()
        mx.eval(y)
        mx.synchronize(mx.gpu)
        warm[arm] = output_digest(y)
        del y
    result["warm_output_sha256"] = warm
    progress()
    if warm != cold:
        raise ValueError("Warm outputs changed from cold")
    blocks: list[dict[str, object]] = []
    result["blocks"] = blocks
    for order in ORDERS:
        timings: dict[str, list[int]] = {}
        hashes: dict[str, list[str]] = {}
        block: dict[str, object] = {
            "order": list(order),
            "dispatch_ns": timings,
            "output_sha256": hashes,
            "completed": False,
            "outputs_reproduce_cold": False,
            "memory_before": memory(),
        }
        blocks.append(block)
        for arm in order:
            timings[arm], hashes[arm] = [], []
            for _ in range(10):
                mx.synchronize(mx.gpu)
                start = time.perf_counter_ns()
                y = calls[arm]()
                mx.eval(y)
                mx.synchronize(mx.gpu)
                timings[arm].append(time.perf_counter_ns() - start)
                hashes[arm].append(output_digest(y))
                del y
            progress()
        reproduced = all(
            value == cold[arm] for arm, values in hashes.items() for value in values
        )
        block.update(
            completed=True, outputs_reproduce_cold=reproduced, memory_after=memory()
        )
        progress()
        if not reproduced:
            raise ValueError("Timed outputs changed from cold")
    result["performance"] = comparison(blocks, ARMS[0])
    result["tile64_over_unroll64"] = comparison(blocks, ARMS[1])
    progress()


def previous_outputs(context: Context) -> dict[str, Array]:
    raw = context.cold.read_bytes()
    if (
        hashlib.sha256(raw).hexdigest()
        != object_value(context.prior["cold_outputs"])["sha256"]
    ):
        raise ValueError("Frozen64 output artifact changed before consumption")
    arrays = cast(dict[str, Array], load(raw))
    prior_arms = ("maintained", "tile32-bf16", "tile64-bf16", "tile64-fp32")
    if set(arrays) != {name + "-" + arm for name, _, _ in SHAPES for arm in prior_arms}:
        raise ValueError("Frozen64 output inventory changed")
    for name, _, rows in SHAPES:
        hashes = object_value(
            object_value(object_value(context.prior["shapes"])[name])[
                "cold_output_sha256"
            ]
        )
        for arm in prior_arms:
            value = arrays[name + "-" + arm]
            if (
                value.dtype != np.float32
                or value.shape != (rows, 3072)
                or not np.isfinite(value).all()
                or hashlib.sha256(value.tobytes()).hexdigest() != hashes[arm]
            ):
                raise ValueError("Frozen64 output tensor changed")
    return arrays


def screen(
    calls: Mapping[str, Operation], original: Mapping[str, Array], name: str
) -> tuple[dict[str, object], dict[str, Array]]:
    values: dict[str, Array] = {}
    cold_ms: dict[str, float] = {}
    for arm, call in calls.items():
        mx.synchronize(mx.gpu)
        start = time.perf_counter_ns()
        y = call()
        mx.eval(y)
        mx.synchronize(mx.gpu)
        cold_ms[arm] = (time.perf_counter_ns() - start) / 1e6
        values[arm] = detached_float32(y)
        del y
    reproduced = all(
        np.array_equal(
            values[arm].view(np.uint32), original[name + "-" + arm].view(np.uint32)
        )
        for arm in ARMS[:2]
    )
    storage_equal = np.array_equal(
        values[CONTROL].view(np.uint32), values[ARMS[2]].view(np.uint32)
    )
    quality = {
        arm: projection_quality(values[ARMS[0]], values[arm])
        for arm in (*ARMS[1:], CONTROL)
    }
    return {
        "cold_dispatch_ms": cold_ms,
        "cold_output_sha256": {
            arm: hashlib.sha256(value.tobytes()).hexdigest()
            for arm, value in values.items()
        },
        "original_cold_outputs_reproduced": reproduced,
        "custom_storage_bitwise_equal": storage_equal,
        "quality": quality,
        "passed": reproduced
        and storage_equal
        and all(gate["passed"] is True for gate in quality.values()),
        "blocks": [],
    }, values


def study(context: Context, output: Path) -> bool:
    if output.exists():
        raise FileExistsError("Preserve prior unroll receipts")
    if os.environ.get("MLX_ENABLE_TF32") != "0":
        raise ValueError("Disable TF32 before importing MLX")
    payload: dict[str, object] = {
        "completed": False,
        "accepted": False,
        "phase": "admission",
        "scope": (
            "Exact archived first-layer gate projection; no model recapture "
            "or complete inference speed claim"
        ),
        "protocol": {
            "shapes": SHAPES,
            "arms": ARMS,
            "cold_control": CONTROL,
            "orders": ORDERS,
            "dispatches_per_arm_block": 10,
            "warm_dispatches": 1,
            "cold_dispatches": 1,
            "timeout_seconds": 180,
            "cache_cap_bytes": 512 << 20,
            "atol": 1e-4,
            "rtol": 1e-4,
            "median_minimum": 1.05,
            "p05_exclusive_minimum": 1.0,
            "bitwise_storage_controls": True,
            "partial_receipt": (
                "Complete arms saved; interruption during ten dispatches "
                "leaves partial arm unknown"
            ),
            "observation": (
                "Every warm/timed output hashed after its timer; "
                "observations separate dispatches, not unobserved throughput"
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    save(output, payload)
    try:
        from .kernel import ProjectionKernel

        sources = {
            **context.sources,
            **{
                str(path.relative_to(ROOT)): path
                for path in Path(__file__).parent.glob("*.py")
            },
        }
        sources.update(
            {
                name: ROOT / name
                for name in (
                    "docs/analysis/unroll.md",
                    "pyproject.toml",
                    "tests/analysis/test_mlx_unroll_protocol.py",
                    "tests/analysis/test_mlx_unroll_kernel.py",
                )
            }
        )
        frozen = freeze_sources(sources, output)
        inputs, obsolete_cold = read_arrays(context.original)
        del obsolete_cold
        original = previous_outputs(context)
        payload.update(
            source_hashes=frozen.hashes(),
            source_archive=frozen.archive(),
            reused_inputs={
                "path": str(context.original.inputs),
                "sha256": digest(context.original.inputs),
            },
            prior_receipts={str(path): sha for path, sha in PINNED.items()},
            prior_cold_outputs=context.prior["cold_outputs"],
            **{
                field: context.prior[field]
                for field in (
                    "model_id",
                    "dataset_sha256",
                    "reference_sha256",
                    "input_sha256",
                )
            },
            device=mx.device_info(mx.gpu),
        )
        with restored_cache(payload), mx.stream(mx.gpu):
            frozen64 = FrozenKernel(OriginalKernel())
            unroll64 = ProjectionKernel(frozen64)
            w32 = mx.array(inputs["weight"], dtype=mx.float32)
            w16 = w32.astype(mx.bfloat16)
            mx.eval(w32, w16)
            if not np.array_equal(
                detached_float32(w16.astype(mx.float32)).view(np.uint32),
                inputs["weight"].view(np.uint32),
            ):
                raise ValueError("Device BF16 reconstruction changed weight bits")
            results: dict[str, dict[str, object]] = {}
            operations: dict[str, dict[str, Operation]] = {}
            cold_values: dict[str, Array] = {}
            payload.update(phase="screen", shapes=results)
            for name, _, _ in SHAPES:
                x = mx.array(inputs[name], dtype=mx.float32)
                mx.eval(x)
                ready_old = frozen64.prepare(x, w16)
                ready_new = unroll64.prepare(x, w16)
                control = unroll64.prepare(x, w32)

                def maintained(x: mx.array = x) -> mx.array:
                    return x @ w32.T

                calls = {
                    ARMS[0]: maintained,
                    ARMS[1]: ready_old.run,
                    ARMS[2]: ready_new.run,
                    CONTROL: control.run,
                }
                results[name], values = screen(calls, original, name)
                cold_values.update(
                    {name + "-" + arm: value for arm, value in values.items()}
                )
                operations[name] = {arm: calls[arm] for arm in ARMS}
                save(output, payload)
            cold_path = output.with_suffix(".outputs.tensors.safetensors")
            if cold_path.exists():
                raise FileExistsError("Preserve cold outputs")
            save_file(cold_values, str(cold_path))
            payload["cold_outputs"] = {
                "path": str(cold_path),
                "sha256": digest(cold_path),
            }
            numerical = all(result["passed"] is True for result in results.values())
            payload["projection_quality_passed"] = numerical
            if numerical:
                if any(digest(Path(item.path)) != item.sha256 for item in frozen.files):
                    raise ValueError("Frozen trial source changed")
                payload["phase"] = "timing"
                for name, _, _ in SHAPES:
                    hashes = cast(dict[str, str], results[name]["cold_output_sha256"])
                    measure(
                        operations[name],
                        {arm: hashes[arm] for arm in ARMS},
                        results[name],
                        lambda: save(output, payload),
                    )
            payload.update(
                completed=True,
                phase="complete",
                accepted=numerical
                and all(
                    cast(dict[str, object], results[name].get("performance", {})).get(
                        "passed"
                    )
                    is True
                    for name in ("prefix-b1", "prefix-b8")
                ),
            )
    except BaseException as error:
        payload.update(
            completed=False, accepted=False, error=f"{type(error).__name__}: {error}"
        )
    finally:
        try:
            save(output, payload)
        except Exception as error:
            payload.update(
                completed=False,
                accepted=False,
                final_write_error=f"{type(error).__name__}: {error}",
            )
            save(output, payload)
    return payload["completed"] is True and payload["accepted"] is True
