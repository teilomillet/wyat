"""Fixed64 reuse trial on admitted prior activations; no model recapture."""

from __future__ import annotations

import hashlib
import itertools
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import cast

import mlx.core as mx
import numpy as np
from safetensors.numpy import load, save_file

from benchmarks.analysis.evidence import freeze_sources
from benchmarks.analysis.mlx.endpoints import memory
from benchmarks.analysis.mlx.mixed.capture import (
    detached_float32,
    exact_bfloat_bits,
    projection_quality,
)
from benchmarks.analysis.mlx.mixed.kernel import ProjectionKernel as FrozenKernel
from benchmarks.analysis.mlx.mixed.study import ARMS as ORIGINAL_ARMS
from benchmarks.analysis.mlx.mixed.study import (
    SHAPES,
    output_digest,
    performance,
    restored_cache,
    save,
)

from .admit import IDENTITIES, ROOT, Admission, digest, object_value

type Array = np.ndarray[tuple[int, int], np.dtype[np.float32]]
type Operation = Callable[[], mx.array]
ARMS = ("maintained", "tile32-bf16", "tile64-bf16")
ORDERS = tuple(itertools.permutations(ARMS))
CONTROL = "tile64-fp32"


def read_arrays(admission: Admission) -> tuple[dict[str, Array], dict[str, Array]]:
    def consumed(path: Path, field: str) -> dict[str, Array]:
        raw = path.read_bytes()
        expected = object_value(admission.initial[field])["sha256"]
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("Captured tensor artifact changed before consumption")
        return cast(dict[str, Array], load(raw))

    inputs = consumed(admission.inputs, "captured_bytes")
    cold = consumed(admission.cold, "cold_outputs")
    if set(inputs) != {"weight", *(name for name, _, _ in SHAPES)} or set(cold) != {
        name + "-" + arm for name, _, _ in SHAPES for arm in ORIGINAL_ARMS
    }:
        raise ValueError("Frozen tensor inventory changed")
    checks = [("weight", (3072, 1024), object_value(admission.initial["weight"]))]
    captures = object_value(admission.initial["captures"])
    checks.extend(
        (name, (rows, 1024), object_value(captures[name])) for name, _, rows in SHAPES
    )
    for name, shape, record in checks:
        value = inputs[name]
        if (
            value.dtype != np.float32
            or value.shape != shape
            or not np.isfinite(value).all()
            or hashlib.sha256(value.tobytes()).hexdigest() != record["sha256"]
        ):
            raise ValueError("Frozen input geometry or bytes changed")
    if not exact_bfloat_bits(inputs["weight"]):
        raise ValueError("Captured weight is not bitwise BF16 decodable")
    shapes = object_value(admission.initial["shapes"])
    for name, _, rows in SHAPES:
        hashes = object_value(object_value(shapes[name])["cold_output_sha256"])
        for arm in ORIGINAL_ARMS:
            value = cold[name + "-" + arm]
            if (
                value.dtype != np.float32
                or value.shape != (rows, 3072)
                or not np.isfinite(value).all()
                or hashlib.sha256(value.tobytes()).hexdigest() != hashes[arm]
            ):
                raise ValueError("Frozen cold outputs changed")
    return inputs, cold


def comparison(blocks: list[dict[str, object]], baseline: str) -> dict[str, object]:
    """Reuse the original six-permutation gate; keep32 comparison descriptive."""
    translated: list[dict[str, object]] = []
    original_orders = tuple(itertools.permutations(ORIGINAL_ARMS))
    for block, expected, original_order in zip(
        blocks, ORDERS, original_orders, strict=True
    ):
        if tuple(cast(list[str], block["order"])) != expected:
            raise ValueError("Balanced execution order changed")
        timings = cast(dict[str, list[int]], block["dispatch_ns"])
        if set(timings) != set(ARMS):
            raise ValueError("Missing timed arm")
        translated.append(
            {
                "order": list(original_order),
                "dispatch_ns": {
                    "maintained": timings[baseline],
                    "custom-fp32": timings["tile32-bf16"],
                    "custom-bf16": timings["tile64-bf16"],
                },
            }
        )
    report = performance(translated)
    if baseline != "maintained":
        report.pop("passed")
    return report


def timed_blocks(
    calls: dict[str, Operation],
    cold: dict[str, str],
    result: dict[str, object],
    progress: Callable[[], None],
) -> None:
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
    result["performance"] = comparison(blocks, "maintained")
    result["tile32_over_tile64"] = comparison(blocks, "tile32-bf16")
    progress()


def screen(
    calls: dict[str, Operation], original: dict[str, Array], name: str
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
    hashes = {
        arm: hashlib.sha256(value.tobytes()).hexdigest()
        for arm, value in values.items()
    }
    history = all(
        np.array_equal(
            values[arm].view(np.uint32), original[name + "-" + old].view(np.uint32)
        )
        for arm, old in (("maintained", "maintained"), ("tile32-bf16", "custom-bf16"))
    )
    storage = np.array_equal(
        values[CONTROL].view(np.uint32), values["tile64-bf16"].view(np.uint32)
    )
    quality = {
        arm: projection_quality(values["maintained"], values[arm])
        for arm in (*ARMS[1:], CONTROL)
    }
    result: dict[str, object] = {
        "cold_dispatch_ms": cold_ms,
        "cold_output_sha256": hashes,
        "original_cold_outputs_reproduced": history,
        "custom_storage_bitwise_equal": storage,
        "quality": quality,
        "passed": history
        and storage
        and all(gate["passed"] is True for gate in quality.values()),
        "blocks": [],
    }
    return result, values


def study(admission: Admission, output: Path) -> bool:
    if output.exists():
        raise FileExistsError("Preserve prior tile receipts")
    if os.environ.get("MLX_ENABLE_TF32") != "0":
        raise ValueError("Disable TF32 before importing MLX")
    payload: dict[str, object] = {
        "completed": False,
        "accepted": False,
        "phase": "admission",
        "scope": (
            "Archived first-layer gate projection only;8 development cases; "
            "no model recapture or complete inference claim"
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
                "Every warm/timed output hashed outside its timer against cold; "
                "observations separate dispatches, not unobserved throughput"
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    save(output, payload)
    try:
        from .kernel import ProjectionKernel

        sources = {
            **admission.sources,
            **{
                str(path.relative_to(ROOT)): path
                for path in Path(__file__).parent.glob("*.py")
            },
        }
        sources["docs/analysis/tile.md"] = ROOT / "docs/analysis/tile.md"
        sources["tests/analysis/test_mlx_tile_protocol.py"] = (
            ROOT / "tests/analysis/test_mlx_tile_protocol.py"
        )
        sources["tests/analysis/test_mlx_tile_kernel.py"] = (
            ROOT / "tests/analysis/test_mlx_tile_kernel.py"
        )
        frozen = freeze_sources(sources, output)
        inputs, original = read_arrays(admission)
        payload.update(
            source_hashes=frozen.hashes(),
            source_archive=frozen.archive(),
            reused_inputs={
                "path": str(admission.inputs),
                "sha256": digest(admission.inputs),
            },
            prior_receipts={
                str(path): expected for path, expected in IDENTITIES.items()
            },
            initial_cold_outputs=admission.initial["cold_outputs"],
            model_id=admission.initial["model_id"],
            dataset_sha256=admission.initial["dataset_sha256"],
            reference_sha256=admission.initial["reference_sha256"],
            input_sha256=admission.initial["input_sha256"],
            device=mx.device_info(mx.gpu),
        )
        with restored_cache(payload), mx.stream(mx.gpu):
            fixed32 = FrozenKernel()
            fixed64 = ProjectionKernel(fixed32)
            w32 = mx.array(inputs["weight"], dtype=mx.float32)
            w16 = w32.astype(mx.bfloat16)
            mx.eval(w32, w16)
            if not np.array_equal(
                detached_float32(w16.astype(mx.float32)).view(np.uint32),
                inputs["weight"].view(np.uint32),
            ):
                raise ValueError("Device BF16 reconstruction changed weight bits")
            results: dict[str, dict[str, object]] = {}
            calls_by_shape: dict[str, dict[str, Operation]] = {}
            cold_values: dict[str, Array] = {}
            payload.update(phase="screen", shapes=results)
            for name, _, _ in SHAPES:
                x = mx.array(inputs[name], dtype=mx.float32)
                mx.eval(x)
                ready32 = fixed32.prepare(x, w16)
                ready64 = fixed64.prepare(x, w16)
                control = fixed64.prepare(x, w32)

                def maintained(x: mx.array = x) -> mx.array:
                    return x @ w32.T

                calls = {
                    "maintained": maintained,
                    "tile32-bf16": ready32.run,
                    "tile64-bf16": ready64.run,
                    CONTROL: control.run,
                }
                results[name], values = screen(calls, original, name)
                cold_values.update(
                    {name + "-" + arm: value for arm, value in values.items()}
                )
                calls_by_shape[name] = {arm: calls[arm] for arm in ARMS}
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
                    raise ValueError("Frozen trial sources changed")
                payload["phase"] = "timing"
                for name, _, _ in SHAPES:
                    hashes = cast(dict[str, str], results[name]["cold_output_sha256"])
                    timed_blocks(
                        calls_by_shape[name],
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
