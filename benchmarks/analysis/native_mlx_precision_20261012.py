"""Bounded, exploratory source-motivated native backbone precision screening.

One shared model, unchanged FP32 adapters/head and frozen gates. Each variant
first exercises the known failing case, then all frozen cases only if it passes.
This exploratory selection is not an independent confirmatory quality result.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, cast

import native_mlx_benchmark_20261012 as bench  # type: ignore[import-not-found, unused-ignore]


def precision_classes() -> tuple[type[Any], type[Any]]:
    import mlx.core as mx
    from mlx.nn.layers.base import Module

    class ExplicitRMSNorm(Module):
        """Torch's FP32 square/mean/rsqrt/multiply/weight operation ordering."""

        def __init__(self, original: Any) -> None:
            super().__init__()  # type: ignore[no-untyped-call]
            self.weight = original.weight  # Reference the same storage, never copy.
            self.eps = original.eps

        def __call__(self, value: mx.array) -> mx.array:
            dtype = value.dtype
            value = value.astype(mx.float32)
            square = mx.square(value)
            mx.eval(square)
            variance = mx.mean(square, axis=-1, keepdims=True)
            mx.eval(variance)
            reciprocal = mx.rsqrt(variance + self.eps)
            mx.eval(reciprocal)
            normalized = value * reciprocal
            mx.eval(normalized)
            return cast(mx.array, self.weight * normalized.astype(dtype))

    class ExplicitRoPE(Module):
        """Default half-stride rotation with separately materialized FP32 steps."""

        def __init__(self, original: Any) -> None:
            super().__init__()  # type: ignore[no-untyped-call]
            if original.traditional or original.scale != 1.0:
                raise ValueError("screening supports default unscaled half-stride RoPE")
            self.dims = original.dims
            self.base = original.base
            self._tables: dict[tuple[int, int], tuple[mx.array, mx.array]] = {}
            exponents = mx.arange(0, self.dims, 2, dtype=mx.float32) / self.dims
            mx.eval(exponents)
            powers = mx.power(mx.array(self.base, dtype=mx.float32), exponents)
            mx.eval(powers)
            self._inverse = mx.reciprocal(powers)
            mx.eval(self._inverse)

        def __call__(self, value: mx.array, offset: int = 0) -> mx.array:
            if type(offset) is not int or value.shape[-1] != self.dims:
                raise ValueError(
                    "screening requires scalar offset and complete rotary width"
                )
            width = value.shape[-2]
            key = (width, offset)
            if key not in self._tables:
                # Keep at most the current width, bounded independently of cohort size.
                self._tables.clear()
                positions = mx.arange(offset, offset + width, dtype=mx.float32)
                angles = positions[:, None] * self._inverse[None, :]
                mx.eval(angles)
                angles = mx.concatenate([angles, angles], axis=-1)
                cosine, sine = mx.cos(angles), mx.sin(angles)
                mx.eval(cosine, sine)
                self._tables[key] = (cosine, sine)
            cosine, sine = self._tables[key]
            left = value[..., : self.dims // 2]
            right = value[..., self.dims // 2 :]
            rotated = mx.concatenate([-right, left], axis=-1)
            original_product = value * cosine
            rotated_product = rotated * sine
            # A boundary before addition prevents multiply-add fusion/reassociation.
            mx.eval(original_product, rotated_product)
            return original_product + rotated_product

    return ExplicitRMSNorm, ExplicitRoPE


def owners(backbone: Any) -> list[tuple[Any, str, Any]]:
    result = [(backbone, "norm", backbone.norm)]
    for layer in backbone.layers:
        for owner, name in (
            (layer, "input_layernorm"),
            (layer, "post_attention_layernorm"),
            (layer.self_attn, "q_norm"),
            (layer.self_attn, "k_norm"),
            (layer.self_attn, "rope"),
        ):
            result.append((owner, name, getattr(owner, name)))
    return result


def restore(originals: list[tuple[Any, str, Any]]) -> None:
    for owner, name, original in originals:
        setattr(owner, name, original)


def install(
    originals: list[tuple[Any, str, Any]],
    variant: str,
    classes: tuple[type[Any], type[Any]],
) -> None:
    norm_type, rope_type = classes
    restore(originals)
    shared_rope = None
    for owner, name, original in originals:
        if name == "rope" and variant in {"rope", "both"}:
            if shared_rope is None:
                shared_rope = rope_type(original)
            elif (
                original.dims,
                original.base,
                original.scale,
                original.traditional,
            ) != (
                shared_rope.dims,
                shared_rope.base,
                1.0,
                False,
            ):
                raise ValueError("rotary configurations differ across layers")
            setattr(owner, name, shared_rope)
        elif name != "rope" and variant in {"rms", "both"}:
            setattr(owner, name, norm_type(original))


def observe(model: Any, rows: Any, ref: Any, gates: Any) -> dict[str, Any]:
    import mlx.core as mx

    result: dict[str, Any] = {"outputs": None, "error": None, "elapsed_ms": None}
    try:
        mx.synchronize()
        started = time.perf_counter()
        values = model._logits(rows)
        mx.synchronize()
        result["elapsed_ms"] = (time.perf_counter() - started) * 1000
        result["outputs"] = bench.snapshot(rows, values)
    except Exception as error:
        result["error"] = repr(error)
    result["comparison"] = bench.compare_rows(ref, result["outputs"], gates)
    result["memory"] = bench.memory("metal")
    time.sleep(0.250)
    return result


def run(args: argparse.Namespace, result: dict[str, Any]) -> None:
    import mlx.core as mx

    frozen = bench.read(args.frozen)
    bench.check_frozen(frozen)
    reference = bench.read(args.reference)
    if reference.get("load_error") or reference["frozen_sha256"] != bench.digest(
        args.frozen
    ):
        raise RuntimeError("CPU reference failed or cohort identity differs")
    if reference.get("arm") != "cpu" or reference.get("model_id") != frozen["model_id"]:
        raise RuntimeError("CPU reference arm or model identity differs")
    expected = {case["id"] for case in frozen["cases"]}
    refs = {case["id"]: case for case in reference["cases"]}
    if set(refs) != expected or any(case.get("error") for case in refs.values()):
        raise RuntimeError("CPU reference case denominator failed")
    if not all(
        bench.compare_rows(case["outputs"], case["outputs"], frozen["quality_gates"])[
            "pass"
        ]
        for case in refs.values()
    ):
        raise RuntimeError("CPU reference outputs failed finite numerical checks")
    model = bench.load_model(args, metal=args.arm == "metal")
    if model.model_id != frozen["model_id"]:
        raise RuntimeError("candidate model identity differs")
    batches = bench.prepare(model, frozen)
    if set(batches) != expected:
        raise RuntimeError("prepared case denominator differs")
    if not model.responsive:
        raise RuntimeError("responsive layer pacing must remain enabled")
    originals = owners(model.backbone)
    classes = precision_classes()
    result.update(
        metadata=bench.metadata(),
        model_id=model.model_id,
        metal_enabled=model.metal_enabled,
        manifest_sha256=bench.digest(args.artifact / "native_manifest.json"),
        resource_policy={
            "one_shared_backbone": True,
            "pause_ms": 250,
            "responsive": True,
            "memory_limit_mib": args.memory_limit_mib,
            "cache_limit_mib": args.cache_limit_mib,
        },
    )
    source_paths = {
        Path(__file__).resolve(),
        Path(bench.__file__).resolve(),
        args.frozen.resolve(),
        args.reference.resolve(),
        (args.artifact / "native_manifest.json").resolve(),
        *(Path(item["path"]) for item in result["metadata"]["sources"].values()),
    }
    result["source_sha256"] = {str(path): bench.digest(path) for path in source_paths}
    bench.checkpoint(args.output, result)
    try:
        for variant in args.variants:
            with mx.stream(model.device):
                install(originals, variant, classes)
            item: dict[str, Any] = {"variant": variant, "cases": []}
            result["variants"].append(item)
            screen = observe(
                model,
                batches[args.screen],
                refs[args.screen]["outputs"],
                frozen["quality_gates"],
            )
            item["screen"] = dict(screen, id=args.screen)
            item["screen_pass"] = screen["comparison"]["pass"]
            bench.checkpoint(args.output, result)
            if item["screen_pass"]:
                for case_id, rows in batches.items():
                    observed = observe(
                        model, rows, refs[case_id]["outputs"], frozen["quality_gates"]
                    )
                    item["cases"].append(dict(observed, id=case_id))
                    bench.checkpoint(args.output, result)
                item["full_frozen_pass"] = len(item["cases"]) == len(expected) and all(
                    case["comparison"]["pass"] for case in item["cases"]
                )
            else:
                item["full_frozen_pass"] = None
                item["full_frozen_status"] = "not run because screening failed"
            restore(originals)
            mx.synchronize()
            mx.clear_cache()
            bench.checkpoint(args.output, result)
    finally:
        restore(originals)
    result["sources_at_end_sha256"] = {
        path: bench.digest(Path(path)) for path in result["source_sha256"]
    }
    result["source_fence_pass"] = (
        result["source_sha256"] == result["sources_at_end_sha256"]
    )
    if not result["source_fence_pass"]:
        raise RuntimeError("diagnostic source or input changed during execution")
    result["error"] = None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("artifact", "frozen", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--arm", choices=("ordinary", "metal"), default="metal")
    parser.add_argument("--screen", default="single-34")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("baseline", "rms", "rope", "both"),
        default=["baseline", "rms", "rope", "both"],
    )
    parser.add_argument("--memory-limit-mib", type=int, default=6144)
    parser.add_argument("--cache-limit-mib", type=int, default=256)
    args = parser.parse_args()
    if not 1 <= args.memory_limit_mib <= 6144 or not 0 <= args.cache_limit_mib <= 256:
        parser.error("memory guideline must be1..6144MiB and cache0..256MiB")
    if len(set(args.variants)) != len(args.variants):
        parser.error("variants must not repeat")
    for suffix in ("", ".partial", ".partial.tmp"):
        if args.output.with_name(args.output.name + suffix).exists():
            parser.error("output exists; choose a fresh filename")
    result: dict[str, Any] = {
        "classification": "exploratory source-motivated numerical screening",
        "diagnostic_sha256": bench.digest(Path(__file__)),
        "frozen_sha256": bench.digest(args.frozen),
        "reference_sha256": bench.digest(args.reference),
        "variants": [],
        "error": "not completed",
    }
    bench.checkpoint(args.output, result)
    try:
        run(args, result)
    except Exception as error:
        result.update(error=repr(error), metadata=bench.metadata())
        bench.finish(args.output, result)
        raise
    bench.finish(args.output, result)


if __name__ == "__main__":
    main()
