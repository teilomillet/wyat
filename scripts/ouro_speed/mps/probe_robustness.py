"""Robustness / failure matrix for Ouro decision scoring on Apple GPU (MPS).

Documents, under the mandatory caps (fraction 0.6, LOW=1.0/HIGH=1.2, RSS
watchdog 14 GiB, fallback env unset):

1. device-policy gating: ``mps-fp32`` artifact on MPS loads; ``cpu-fp32``
   artifact rejected on MPS and ``mps-fp32`` artifact rejected on CPU unless the
   explicit override is passed (no silent loophole);
2. which torch ops used by the scoring path run on MPS with fallback disabled,
   plus known-unsupported ops as proof that fallback is really off (they raise
   instead of silently running on CPU);
3. cold-start / load time and MPS allocator state after load;
4. repeated-call leak probe: current/driver allocated memory before/after 50
   scoring calls plus an explicit ``empty_cache`` release check;
5. deliberate approach to the 0.6 fraction cap to document the exact failure
   mode (allocation error vs watchdog abort), and recovery afterwards.

Run:
    .venv/bin/python scripts/ouro_speed/mps/probe_robustness.py
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mps_common as common  # noqa: E402

sys.path.insert(0, str(common.REPO_ROOT))

import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from wyat.loading.ouro import (  # noqa: E402
    load_ouro,
    prefill_context,
    score_candidates_from_cache,
)

MIB = common.MIB


def op_probe(name: str, fn: Callable[[], Any]) -> dict[str, Any]:
    """Run one op on MPS and classify ok/unsupported/error."""
    try:
        torch.mps.synchronize()
        value = fn()
        torch.mps.synchronize()
        shape = list(value.shape) if isinstance(value, torch.Tensor) else None
        return {"op": name, "status": "ok", "output_shape": shape}
    except Exception as error:  # noqa: BLE001 - support probe only
        return {
            "op": name,
            "status": "error",
            "error_type": type(error).__name__,
            "error": str(error)[:500],
        }


def op_support_matrix() -> list[dict[str, Any]]:
    device = torch.device("mps")
    x = torch.randn(8, 64, device=device)
    y = torch.randn(64, 64, device=device)
    q = torch.randn(1, 4, 8, 16, device=device)
    k = torch.randn(1, 4, 8, 16, device=device)
    mask = torch.zeros(1, 1, 8, 8, device=device)
    xb = x.to(torch.bfloat16)
    yb = y.to(torch.bfloat16)
    b = torch.randn(4, 4, 8, 8, device=device)
    c = torch.randn(4, 8, 8, device=device)
    probes = [
        ("matmul.fp32", lambda: x @ y),
        ("matmul.bf16", lambda: xb @ yb),
        ("baddbmm.fp32", lambda: torch.baddbmm(b, b, c)),
        (
            "scaled_dot_product_attention.fp32+additive_mask",
            lambda: F.scaled_dot_product_attention(q, k, k, attn_mask=mask),
        ),
        (
            "scaled_dot_product_attention.bf16+bf16_mask",
            lambda: F.scaled_dot_product_attention(
                q.to(torch.bfloat16),
                k.to(torch.bfloat16),
                k.to(torch.bfloat16),
                attn_mask=mask.to(torch.bfloat16),
            ),
        ),
        ("layer_norm.fp32", lambda: F.layer_norm(x, (64,))),
        ("gelu.fp32", lambda: F.gelu(x)),
        ("silu.fp32", lambda: F.silu(x)),
        ("softmax.fp32", lambda: F.softmax(x, dim=-1)),
        (
            "embedding.fp32",
            lambda: F.embedding(torch.zeros(1, 4, dtype=torch.long, device=device), y),
        ),
        ("cat.fp32", lambda: torch.cat([x, x], dim=1)),
        ("view_as_complex", lambda: torch.view_as_complex(x.reshape(8, 32, 2))),
        (
            "view_as_real",
            lambda: torch.view_as_real(torch.view_as_complex(x.reshape(8, 32, 2))),
        ),
        ("rsqrt.fp32", lambda: torch.rsqrt(x + 1.0)),
        ("einsum.fp32", lambda: torch.einsum("ij,kj->ik", x, y)),
        ("topk.fp32", lambda: torch.topk(x, 4, dim=-1)),
        ("sort.fp32", lambda: torch.sort(x, dim=-1)),
        ("argmax.fp32", lambda: torch.argmax(x, dim=-1)),
        ("floating_fft_probe.unsupported", lambda: torch.fft.fft(x)),
        ("linalg_svd_probe.unsupported", lambda: torch.linalg.svd(y)),
    ]
    return [op_probe(name, fn) for name, fn in probes]


def leak_probe(backbone: Any, head: Any, question: dict[str, Any]) -> dict[str, Any]:
    """50 repeated end-to-end scoring calls: does MPS memory grow?"""
    samples = []
    calls = 50
    before = common.mps_memory()
    started = time.perf_counter()
    with torch.inference_mode():
        context, candidates = common.to_device(question, "mps")
        for index in range(calls):
            cache = prefill_context(backbone, context, policy="last", context_steps=4)
            score_candidates_from_cache(backbone, head, cache, candidates)
            del cache
            if (index + 1) % 10 == 0:
                common.mps_sync()
                memory = common.mps_memory()
                samples.append(
                    {"calls": index + 1, **memory, "rss_gib": common.resident_gib()}
                )
    common.mps_sync()
    after = common.mps_memory()
    empty_before = common.mps_memory()
    common.empty_mps_cache()
    after_empty = common.mps_memory()
    return {
        "calls": calls,
        "seconds": round(time.perf_counter() - started, 3),
        "memory_before": before,
        "memory_after": after,
        "memory_after_empty_cache": after_empty,
        "empty_cache_before": empty_before,
        "samples_every_10_calls": samples,
        "driver_growth_gib": round(
            after.get("mps_driver_gib", 0.0) - before.get("mps_driver_gib", 0.0), 3
        ),
        "current_growth_gib": round(
            after.get("mps_current_gib", 0.0) - before.get("mps_current_gib", 0.0), 3
        ),
        "driver_after_empty_cache_gib": after_empty.get("mps_driver_gib"),
    }


def pressure_probe(
    backbone: Any,
    head: Any,
    question: dict[str, Any],
    *,
    step_mib: int = 512,
    max_allocations: int = 64,
) -> dict[str, Any]:
    """Allocate untill the 0.6 MPS fraction cap errors; document the failure mode."""
    before = common.mps_memory()
    tensors: list[torch.Tensor] = []
    count = 0
    error: dict[str, Any] | None = None
    elements = int(step_mib * MIB // 4)
    try:
        for _ in range(max_allocations):
            tensor = torch.empty(elements, dtype=torch.float32, device="mps")
            tensor.fill_(0.0)
            tensors.append(tensor)
            count += 1
    except Exception as caught:  # noqa: BLE001 - deliberate cap probe
        error = {
            "error_type": type(caught).__name__,
            "error": str(caught)[:800],
        }
        print(f"[robust] cap failure: {error}", flush=True)
    peak = common.mps_memory()
    # Recovery: release everything and score once more.
    for tensor in tensors:
        del tensor
    tensors.clear()
    common.empty_mps_cache()
    recovered = common.mps_memory()
    try:
        with torch.inference_mode():
            context, candidates = common.to_device(question, "mps")
            cache = prefill_context(backbone, context, policy="last", context_steps=4)
            scores = score_candidates_from_cache(backbone, head, cache, candidates)
            logits = common.logits_list(scores)
        recovery_ok = all(value == value for value in logits)
    except Exception as caught:  # noqa: BLE001 - recovery check
        recovery_ok = False
        recovered["recovery_error"] = f"{type(caught).__name__}: {caught}"
    return {
        "requested_step_mib": step_mib,
        "allocations_before_error": count,
        "allocated_gib_before_error": round(count * step_mib / 1024, 3),
        "memory_before": before,
        "memory_at_cap_failure": peak,
        "error": error,
        "memory_after_release": recovered,
        "recovery_scoring_ok": bool(recovery_ok),
    }


def device_policy_probe() -> list[dict[str, Any]]:
    """No silent MPS loophole: mismatched device policy must be rejected early."""
    results = []
    cases = [
        ("mps_artifact_on_cpu", common.MPS_ARTIFACT, "cpu", False, "reject_expected"),
        ("cpu_artifact_on_mps", common.CPU_ARTIFACT, "mps", False, "reject_expected"),
    ]
    for name, path, device, override, expectation in cases:
        started = time.perf_counter()
        try:
            load_ouro(
                path,
                device=device,
                allow_device_policy_override=override,
            )
            results.append(
                {
                    "case": name,
                    "device": device,
                    "status": "loaded (unexpected for this probe)",
                    "seconds": round(time.perf_counter() - started, 3),
                }
            )
        except Exception as error:  # noqa: BLE001 - policy probe
            results.append(
                {
                    "case": name,
                    "device": device,
                    "status": "rejected",
                    "expectation": expectation,
                    "error_type": type(error).__name__,
                    "error": str(error)[:400],
                    "seconds": round(time.perf_counter() - started, 3),
                }
            )
    return results


def main() -> None:
    stable = common.await_loader_stable()
    loader_at_start = stable["loader"]
    watchdog_log = Path("/tmp/ouro-speed/logs/mps-robustness-watchdog.jsonl")
    with common.bench_lock(tag="mps-probe-robustness"):
        caps = common.configure(
            device="mps",
            fraction=0.6,
            threads=6,
            rss_cap_gib=14.0,
            log_path=watchdog_log,
        )
        questions = common.load_question_set()
        small = questions["headline"]["p400-k6-s16"]
        receipt: dict = {
            "device": "mps",
            "caps_configured": caps,
            "env": {
                "PYTORCH_ENABLE_MPS_FALLBACK": common.os.environ.get(
                    "PYTORCH_ENABLE_MPS_FALLBACK"
                ),
                "PYTORCH_MPS_LOW_WATERMARK_RATIO": common.os.environ.get(
                    "PYTORCH_MPS_LOW_WATERMARK_RATIO"
                ),
                "PYTORCH_MPS_HIGH_WATERMARK_RATIO": common.os.environ.get(
                    "PYTORCH_MPS_HIGH_WATERMARK_RATIO"
                ),
            },
            "watchdog_log": str(watchdog_log),
        }

        # 1. Device policy matrix (before the heavy load; both cases fail fast).
        print("[robust] device-policy matrix", flush=True)
        receipt["device_policy"] = device_policy_probe()
        print(json.dumps(receipt["device_policy"], indent=2), flush=True)

        # 2. Cold start / load.
        started = time.perf_counter()
        scorer = load_ouro(common.MPS_ARTIFACT, device="mps")
        load_seconds = time.perf_counter() - started
        backbone, head = scorer.backbone, scorer.head
        receipt["load"] = {
            "seconds": round(load_seconds, 3),
            "device": str(scorer.device),
            "memory_after_load": common.mps_memory(),
            "rss_after_load_gib": common.resident_gib(),
            "note": (
                "load reads 2.9 GB model.safetensors + materialises FP32 on MPS; "
                "the OS page cache is warm because earlier probes already read it"
            ),
        }
        print(f"[robust] load: {json.dumps(receipt['load'])}", flush=True)

        # 3. Op support matrix (fallback disabled).
        print("[robust] op support matrix", flush=True)
        receipt["op_support"] = op_support_matrix()
        unsupported = [
            item["op"] for item in receipt["op_support"] if item["status"] != "ok"
        ]
        receipt["fallback_proof"] = {
            "env_unset": common.os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") is None,
            "unsupported_ops": unsupported,
            "interpretation": (
                "ops listed as unsupported raised on MPS instead of being executed "
                "on CPU, proving PyTorch MPS fallback is disabled for these timings"
            ),
        }
        print(f"[robust] unsupported ops: {unsupported}", flush=True)

        # 4. Leak probe (warm the model first).
        print("[robust] leak probe: 50 scoring calls", flush=True)
        with torch.inference_mode():
            context, candidates = common.to_device(small, "mps")
            cache = prefill_context(backbone, context, policy="last", context_steps=4)
            score_candidates_from_cache(backbone, head, cache, candidates)
        receipt["leak_probe"] = leak_probe(backbone, head, small)
        print(
            f"[robust] leak: {json.dumps({key: receipt['leak_probe'][key] for key in ('driver_growth_gib', 'current_growth_gib', 'driver_after_empty_cache_gib')})}",
            flush=True,
        )

        # 5. Deliberate cap pressure; write a partial receipt first so evidence
        #    survives a watchdog abort.
        receipt["loader_guard"] = common.loader_run_guard(loader_at_start)
        common.write_receipt(
            "robustness-partial-before-pressure",
            {key: receipt[key] for key in receipt},
        )
        print("[robust] approaching the 0.6 MPS fraction cap", flush=True)
        receipt["pressure_probe"] = pressure_probe(backbone, head, small)
        print(json.dumps(receipt["pressure_probe"], indent=2), flush=True)

        # 6. One-model-per-process rule: a second MPS load in the same process
        #    must fail under the 0.6 fraction cap. The full receipt above is
        #    already on disk, so evidence survives even a hard allocator abort.
        print("[robust] second model load in the same process (cap rule)", flush=True)
        second_load: dict[str, Any] = {}
        try:
            duplicate = load_ouro(common.MPS_ARTIFACT, device="mps")
            second_load = {
                "status": "loaded (unexpected: model is resident twice)",
                "device": str(duplicate.device),
                "memory": common.mps_memory(),
            }
            del duplicate
        except Exception as error:  # noqa: BLE001 - deliberate duplicate-load probe
            second_load = {
                "status": "failed_as_expected",
                "error_type": type(error).__name__,
                "error": str(error)[:800],
                "memory": common.mps_memory(),
                "interpretation": (
                    "one model per process: the MPS fraction cap (0.6 of the "
                    "recommended working set) cannot hold two FP32 1.4B models"
                ),
            }
        receipt["second_load_same_process"] = second_load
        print(f"[robust] second load: {json.dumps(second_load)[:400]}", flush=True)
        common.write_receipt(
            "robustness-second-load",
            {
                "second_load_same_process": second_load,
                "pressure_probe": receipt["pressure_probe"],
                "leak_probe": receipt["leak_probe"],
            },
        )
        receipt["loader_guard"] = common.loader_run_guard(loader_at_start)
        receipt["memory_end"] = common.mps_memory()
        receipt["totals"] = {
            "total_seconds": round(time.perf_counter() - started, 3),
            "peak_rss_gib": common.peak_rss_gib(),
        }
        common.write_receipt("robustness", receipt)
        common.emit(
            {
                "device_policy": receipt["device_policy"],
                "load_seconds": receipt["load"]["seconds"],
                "unsupported_ops": unsupported,
                "leak": {
                    key: receipt["leak_probe"][key]
                    for key in (
                        "driver_growth_gib",
                        "current_growth_gib",
                        "driver_after_empty_cache_gib",
                    )
                },
                "pressure": {
                    "allocated_gib_before_error": receipt["pressure_probe"][
                        "allocated_gib_before_error"
                    ],
                    "error": receipt["pressure_probe"]["error"],
                    "recovery_scoring_ok": receipt["pressure_probe"][
                        "recovery_scoring_ok"
                    ],
                },
            }
        )


if __name__ == "__main__":
    main()
