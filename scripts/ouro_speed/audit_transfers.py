#!/usr/bin/env python3
"""Audit Ouro device/dtype placement and host-device transfers — profiling only.

Produces the "data is transiting correctly" evidence for one scoring call:

* every parameter/buffer's device, dtype, shape and contiguity after load
  (hard anomaly if anything sits on a different device than requested)
* monkeypatched counters for ``Tensor.to`` / ``.cpu()`` / ``.item()`` /
  ``.tolist()`` / ``.numpy()`` in the load (optional) and scoring paths, split
  into device-changing vs same-attribute copies plus sync points with call sites
* a ``torch.cat`` watcher around Stage 2: outputs' device/dtype/contiguity,
  transfer bytes and any mixed-device/mixed-dtype inputs
* retained ``ContextKVCache`` banks: per-layer/step device, dtype, contiguity,
  size
* a torch.profiler cross-check of low-level ``aten::*`` transfer/sync ops
  (``_to_copy``, ``copy_``, ``_local_scalar_dense``) so C++-internal copies and
  implicit scalar syncs are not missed
* MPS current/driver allocated memory before and after each phase

Caps are identical to ``bench.py`` (BENCH_LOCK, nice 10, <=6 threads, RSS
watchdog, MPS fraction <=0.6 + watermark env, fallback disabled). This script
loads the 1.4B model, so it is a heavy run and must hold the bench lock.

Exit codes: 0 clean, 1 hard anomalies found, 2 usage/load failure, 42 RSS abort.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import traceback
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from _common import (  # noqa: E402
    DEFAULT_LOCK_TIMEOUT_S,
    DEFAULT_MPS_FRACTION,
    DEFAULT_RSS_CAP_GIB,
    DEFAULT_THREADS,
    EXIT_RSS_CAP,
    LOCK_DIR,
    MAX_MPS_FRACTION,
    MAX_RSS_CAP_GIB,
    MAX_THREADS,
    PROFILING_NOTICE,
    RECEIPT_SCHEMA,
    LockTimeout,
    RssCapExceeded,
    RssWatchdog,
    apply_nice,
    artifact_fingerprint,
    artifact_vocab_size,
    configure_mps,
    configure_threads,
    current_rss_bytes,
    device_memory,
    environment_snapshot,
    gib,
    load_ouro_scorer,
    make_inputs,
    peak_rss_bytes,
    prescan_env,
    source_fingerprint,
    sync_device,
    write_json,
)  # noqa: E402

prescan_env(sys.argv)

import torch  # noqa: E402

from wyat.loading.ouro import (  # noqa: E402
    prefill_context,
    score_candidates_from_cache,
)

EXIT_ANOMALY = 1
EXIT_USAGE = 2


# ---------------------------------------------------------------------------
# Monkeypatched transfer counters
# ---------------------------------------------------------------------------


def _call_site() -> str:
    """Best external call site (skip torch internals and this audit module)."""
    stack = traceback.extract_stack()
    for frame in reversed(stack[:-2]):
        filename = frame.filename
        if "/torch/" in filename or filename.endswith("audit_transfers.py"):
            continue
        try:
            relative = str(Path(filename).resolve().relative_to(Path.cwd()))
        except Exception:
            relative = filename
        return f"{relative}:{frame.lineno} in {frame.name}"
    return "unknown"


class TransferCounters:
    """Count host-device transfers and sync points in a scoped code region."""

    KINDS = ("to", "cpu", "item", "tolist", "numpy")

    def __init__(self, *, max_sites: int = 40) -> None:
        self.max_sites = int(max_sites)
        self.enabled = False
        self.installed = False
        self.phase = "idle"
        self.counts: Counter[str] = Counter()
        self.sites: list[dict[str, Any]] = []
        self._originals: dict[str, Any] = {}
        self._site_seen: set[tuple[str, str, str]] = set()

    # -- patch management ---------------------------------------------------
    def install(self) -> TransferCounters:
        if self.installed:
            return self
        self._originals = {kind: getattr(torch.Tensor, kind) for kind in self.KINDS}
        counters = self

        def wrapper(kind: str) -> Callable[..., Any]:
            original = counters._originals[kind]

            def patched(self_tensor: Any, *args: Any, **kwargs: Any) -> Any:
                result = original(self_tensor, *args, **kwargs)
                if counters.enabled:
                    if kind in {"to", "cpu"}:
                        counters._record(
                            kind,
                            self_tensor,
                            result.device,
                            result.dtype,
                        )
                    else:
                        counters._record(kind, self_tensor, None, None)
                return result

            patched.__name__ = f"audit_{kind}"
            return patched

        for kind in self.KINDS:
            setattr(torch.Tensor, kind, wrapper(kind))
        self.installed = True
        return self

    def uninstall(self) -> None:
        if not self.installed:
            return
        for kind, original in self._originals.items():
            setattr(torch.Tensor, kind, original)
        self._originals = {}
        self.installed = False
        self.enabled = False

    # -- accounting ---------------------------------------------------------
    def _record(
        self,
        kind: str,
        source: Any,
        target_device: Any,
        target_dtype: Any,
    ) -> None:
        self.counts[f"{kind}:calls"] += 1
        device_changed = target_device is not None and target_device != source.device
        dtype_changed = target_dtype is not None and target_dtype != source.dtype
        if kind in {"to", "cpu"}:
            if device_changed:
                tag = "device_change"
            elif dtype_changed:
                tag = "dtype_only"
            else:
                tag = "same_attrs"
            self.counts[f"{kind}:{tag}"] += 1
        if kind in {"item", "tolist"}:
            self.counts[f"sync:{kind}:{source.device.type}"] += 1
        if kind == "numpy":
            self.counts[f"numpy:from_{source.device.type}"] += 1
        key = (
            self.phase,
            kind,
            "device" if device_changed else ("dtype" if dtype_changed else "attr"),
        )
        if len(self.sites) < self.max_sites and key not in self._site_seen:
            self._site_seen.add(key)
            self.sites.append(
                {
                    "phase": self.phase,
                    "kind": kind,
                    "source_device": str(source.device),
                    "target_device": str(target_device)
                    if target_device is not None
                    else None,
                    "device_changed": device_changed,
                    "dtype_changed": dtype_changed,
                    "shape": list(source.shape),
                    "call_site": _call_site(),
                }
            )

    def report(self) -> dict[str, Any]:
        return {
            "counts": dict(sorted(self.counts.items())),
            "sites": list(self.sites),
            "limitations": (
                "Python-level Tensor method counters only; C++-internal copies and "
                "implicit scalar syncs are covered by the torch.profiler cross-check"
            ),
        }


class CatAudit:
    """Watch torch.cat outputs in Stage 2: device/dtype/contiguity/bytes."""

    def __init__(self, *, max_sites: int = 20) -> None:
        self.max_sites = int(max_sites)
        self.installed = False
        self.enabled = False
        self.expected_device: str | None = None
        self.calls = 0
        self.output_bytes = 0
        self.by_output: Counter[tuple[str, str, bool]] = Counter()
        self.mixed_device_calls = 0
        self.mixed_dtype_calls = 0
        self.off_device_calls = 0
        self.sites: list[dict[str, Any]] = []
        self._original: Any = None
        self._site_seen: set[str] = set()

    def install(self) -> CatAudit:
        if self.installed:
            return self
        audit = self
        self._original = torch.cat

        def patched(tensors: Any, *args: Any, **kwargs: Any) -> Any:
            result = audit._original(tensors, *args, **kwargs)
            if audit.enabled:
                audit._record(tensors, result)
            return result

        patched.__name__ = "audit_cat"
        torch.cat = patched
        self.installed = True
        return self

    def uninstall(self) -> None:
        if not self.installed:
            return
        torch.cat = self._original
        self.installed = False
        self.enabled = False

    def _record(self, tensors: Any, result: Any) -> None:
        flattened: list[Any] = []
        if isinstance(tensors, (list, tuple)):
            flattened = list(tensors)
        elif isinstance(tensors, torch.Tensor):
            flattened = [tensors]
        else:
            flattened = [tensor for tensor in tensors]
        devices = {str(tensor.device) for tensor in flattened}
        dtypes = {str(tensor.dtype) for tensor in flattened}
        self.calls += 1
        try:
            self.output_bytes += int(result.numel()) * int(result.element_size())
        except Exception:  # pragma: no cover
            pass
        contiguous = bool(result.is_contiguous())
        self.by_output[(str(result.device), str(result.dtype), contiguous)] += 1
        if len(devices) > 1:
            self.mixed_device_calls += 1
        if len(dtypes) > 1:
            self.mixed_dtype_calls += 1
        if (
            self.expected_device is not None
            and result.device.type != self.expected_device
        ):
            self.off_device_calls += 1
        site = _call_site()
        if len(self.sites) < self.max_sites and site not in self._site_seen:
            self._site_seen.add(site)
            self.sites.append(
                {
                    "call_site": site,
                    "output_device": str(result.device),
                    "output_dtype": str(result.dtype),
                    "output_contiguous": contiguous,
                    "input_devices": sorted(devices),
                    "input_dtypes": sorted(dtypes),
                }
            )

    def report(self) -> dict[str, Any]:
        return {
            "enabled_calls": self.calls,
            "total_output_bytes": self.output_bytes,
            "total_output_mib": round(self.output_bytes / (1024**2), 6),
            "by_output": [
                {
                    "device": key[0],
                    "dtype": key[1],
                    "contiguous": key[2],
                    "calls": calls,
                }
                for key, calls in sorted(
                    self.by_output.items(),
                    key=lambda item: (item[0][0], item[0][1], item[0][2]),
                )
            ],
            "mixed_device_calls": self.mixed_device_calls,
            "mixed_dtype_calls": self.mixed_dtype_calls,
            "off_device_calls": self.off_device_calls,
            "sites": list(self.sites),
        }


# ---------------------------------------------------------------------------
# Static tensor / KV bank inspection
# ---------------------------------------------------------------------------


def inspect_module_tensors(
    module: Any, label: str, target_device: str
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name, tensor in list(module.named_parameters()) + list(module.named_buffers()):
        rows.append(
            {
                "name": f"{label}.{name}",
                "device": str(tensor.device),
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "numel": int(tensor.numel()),
                "bytes": int(tensor.numel()) * tensor.element_size(),
                "contiguous": bool(tensor.is_contiguous()),
                "off_device": tensor.device.type != target_device,
            }
        )
    by_placement: Counter[tuple[str, str]] = Counter(
        (row["device"], row["dtype"]) for row in rows
    )
    off_device = [row for row in rows if row["off_device"]]
    noncontiguous = [row for row in rows if not row["contiguous"]]
    return {
        "label": label,
        "count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "by_device_dtype": [
            {"device": device, "dtype": dtype, "count": count}
            for (device, dtype), count in sorted(by_placement.items())
        ],
        "off_device": off_device[:20],
        "off_device_count": len(off_device),
        "noncontiguous": noncontiguous[:20],
        "noncontiguous_count": len(noncontiguous),
        "tensors": rows if len(rows) <= 400 else rows[:400],
        "truncated": len(rows) > 400,
    }


def inspect_context_banks(cache: Any, target_device: str) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for bank_index, bank in enumerate(cache.banks):
        for layer_index, entry in enumerate(bank):
            for role, tensor in (("key", entry.key), ("value", entry.value)):
                entries.append(
                    {
                        "bank": bank_index,
                        "layer": layer_index,
                        "role": role,
                        "shape": list(tensor.shape),
                        "device": str(tensor.device),
                        "dtype": str(tensor.dtype),
                        "contiguous": bool(tensor.is_contiguous()),
                        "numel": int(tensor.numel()),
                        "bytes": int(tensor.numel()) * tensor.element_size(),
                        "off_device": tensor.device.type != target_device,
                    }
                )
    off_device = [entry for entry in entries if entry["off_device"]]
    noncontiguous = [entry for entry in entries if not entry["contiguous"]]
    dtypes = sorted({entry["dtype"] for entry in entries})
    devices = sorted({entry["device"] for entry in entries})
    return {
        "policy": getattr(cache, "policy", None),
        "length": int(getattr(cache, "length", 0)),
        "banks": len(cache.banks),
        "layers_per_bank": len(cache.banks[0]) if cache.banks else 0,
        "entries": len(entries),
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "devices": devices,
        "dtypes": dtypes,
        "off_device_count": len(off_device),
        "noncontiguous_count": len(noncontiguous),
        "contiguous": len(noncontiguous) == 0,
        "off_device": off_device[:20],
        "noncontiguous": noncontiguous[:20],
        "sample": entries[:4],
    }


def profiler_transfer_ops(fn: Callable[[], None], *, device: str) -> dict[str, Any]:
    from torch.profiler import ProfilerActivity, profile

    sync_device(device)
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        fn()
    sync_device(device)
    wanted = (
        "_to_copy",
        "copy_",
        "_local_scalar_dense",
        "aten::item",
        "aten::cat",
        "aten::to",
    )
    rows = []
    for entry in sorted(
        prof.key_averages(), key=lambda item: getattr(item, "count", 0), reverse=True
    ):
        if any(token in entry.key for token in wanted):
            rows.append(
                {
                    "op": entry.key,
                    "calls": int(getattr(entry, "count", 0)),
                    "self_cpu_ms": round(
                        getattr(entry, "self_cpu_time_total", 0.0) / 1000.0, 4
                    ),
                }
            )
    return {
        "activity": "CPU only (ProfilerActivity.MPS unavailable in this torch build)",
        "ops": rows,
        "note": (
            "cross-check for C++-internal transfers/syncs not visible to the "
            "Tensor-method monkeypatch"
        ),
    }


# ---------------------------------------------------------------------------
# CLI / execution
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit Ouro device placement and transfers (profiling only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    parser.add_argument("--allow-device-override", action="store_true")
    parser.add_argument(
        "--policy-override",
        dest="allow_device_override",
        action="store_true",
        help="alias for --allow-device-override",
    )
    parser.add_argument("--candidate-batching", action="store_true")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--nice", type=int, default=10)
    parser.add_argument("--rss-cap-gib", type=float, default=DEFAULT_RSS_CAP_GIB)
    parser.add_argument("--mps-fraction", type=float, default=DEFAULT_MPS_FRACTION)
    parser.add_argument("--no-lock", action="store_true")
    parser.add_argument("--lock-timeout-s", type=float, default=DEFAULT_LOCK_TIMEOUT_S)
    parser.add_argument("--context-tokens", type=int, default=400)
    parser.add_argument("--candidates", type=int, default=6)
    parser.add_argument("--candidate-length", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument(
        "--include-load",
        action="store_true",
        help="also count transfers during load_ouro (counters installed before load)",
    )
    parser.add_argument("--max-sites", type=int, default=40)
    parser.add_argument("--json-out", default=None)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not 1 <= args.threads <= MAX_THREADS:
        parser.error(f"--threads must be in 1..{MAX_THREADS} (mandatory cap)")
    if not 0 <= args.nice <= 20:
        parser.error("--nice must be in 0..20")
    if not 0 < args.rss_cap_gib <= MAX_RSS_CAP_GIB:
        parser.error(f"--rss-cap-gib must be in (0, {MAX_RSS_CAP_GIB}]")
    if not 0 < args.mps_fraction <= MAX_MPS_FRACTION:
        parser.error(f"--mps-fraction must be in (0, {MAX_MPS_FRACTION}]")
    if args.context_tokens <= 0 or args.candidates <= 0:
        parser.error("token counts must be positive")


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.gmtime())


def run_audit(args: argparse.Namespace) -> int:
    device = args.device.lower()
    print("=" * 78)
    print(PROFILING_NOTICE)
    print("=" * 78)
    print(
        f"[caps] device={device} nice={args.nice} threads={args.threads} "
        f"rss_cap={args.rss_cap_gib}GiB mps_fraction={args.mps_fraction}",
        flush=True,
    )
    apply_nice(args.nice)
    threads_info = configure_threads(args.threads)
    mps_info = configure_mps(args.mps_fraction) if device == "mps" else None
    env = environment_snapshot()
    env["threads"] = threads_info
    env["mps_configured"] = mps_info

    lock = None
    lock_waited = 0.0
    if not args.no_lock:
        from _common import BenchLock

        lock = BenchLock(LOCK_DIR, timeout_s=args.lock_timeout_s, log=print)
        try:
            lock_waited = lock.acquire()
        except LockTimeout as error:
            print(f"[lock] ERROR: {error}", file=sys.stderr)
            return EXIT_USAGE

    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "kind": "audit_transfers",
        "notice": PROFILING_NOTICE,
        "status": "running",
        "started_at": utc_now(),
        "command": {"argv": sys.argv, "cwd": os.getcwd()},
        "environment": env,
        "caps": {
            "threads": args.threads,
            "nice": os.nice(0),
            "rss_cap_gib": args.rss_cap_gib,
            "rss_cap_max_gib": MAX_RSS_CAP_GIB,
            "mps_fraction": args.mps_fraction if device == "mps" else None,
            "mps_configured": mps_info,
            "lock": {
                "path": str(LOCK_DIR),
                "held": lock is not None,
                "waited_s": round(lock_waited, 3),
            },
            "sequential_only": True,
            "candidate_batching": bool(args.candidate_batching),
        },
        "provenance": {
            "source_start": source_fingerprint(),
            "artifact": artifact_fingerprint(args.artifact),
        },
        "anomalies": [],
        "warnings": [],
        "phases": {},
    }

    def abort_receipt(report: dict[str, Any]) -> None:
        receipt.update(
            {
                "status": "aborted",
                "abort_reason": "rss_cap_exceeded",
                "rss_abort": report,
                "finished_at": utc_now(),
            }
        )
        if args.json_out:
            try:
                write_json(args.json_out, receipt)
            except Exception:  # pragma: no cover
                traceback.print_exc()

    watchdog = RssWatchdog(args.rss_cap_gib, hard=True, on_exceed=abort_receipt)
    counters = TransferCounters(max_sites=args.max_sites)
    cat_audit = CatAudit(max_sites=min(args.max_sites, 20))
    scorer = None
    exit_code = 0
    try:
        watchdog.start()
        if args.include_load:
            counters.install()
            counters.phase = "load"
            counters.enabled = True
        started = time.perf_counter()
        scorer = load_ouro_scorer(
            args.artifact,
            device=device,
            allow_device_override=args.allow_device_override,
            candidate_batching=args.candidate_batching,
        )
        load_seconds = time.perf_counter() - started
        if args.include_load:
            counters.enabled = False
            counters.uninstall()
        receipt["phases"]["load"] = {
            "seconds": round(load_seconds, 6),
            "peak_rss_gib": gib(peak_rss_bytes()),
            "mps_memory": device_memory(device),
            "transfers": counters.report() if args.include_load else None,
        }
        counters.sites = []
        counters.counts = Counter()
        counters._site_seen = set()

        target_device = str(scorer.device).split(":")[0]
        receipt["scorer"] = {
            "device": str(scorer.device),
            "device_policy": getattr(scorer, "device_policy", None),
            "candidate_batching": bool(
                getattr(scorer, "candidate_batching", args.candidate_batching)
            ),
            "model_id": scorer.model_id,
            "max_length": int(scorer.max_length),
        }
        print(
            f"[audit] loaded on {scorer.device} in {load_seconds:.2f}s "
            f"(rss peak {gib(peak_rss_bytes()):.2f} GiB)",
            flush=True,
        )

        # ---- static placement -------------------------------------------
        backbone_audit = inspect_module_tensors(
            scorer.backbone, "backbone", target_device
        )
        head_audit = inspect_module_tensors(scorer.head, "head", target_device)
        receipt["tensors"] = {"backbone": backbone_audit, "head": head_audit}
        if backbone_audit["off_device_count"]:
            receipt["anomalies"].append(
                f"{backbone_audit['off_device_count']} backbone tensors not on "
                f"{target_device}: "
                f"{[row['name'] for row in backbone_audit['off_device'][:8]]}"
            )
        if head_audit["off_device_count"]:
            receipt["anomalies"].append(
                f"{head_audit['off_device_count']} head tensors not on {target_device}"
            )
        if backbone_audit["noncontiguous_count"]:
            receipt["warnings"].append(
                f"{backbone_audit['noncontiguous_count']} non-contiguous "
                "backbone tensors"
            )
        print(
            "[audit] tensors: "
            + ", ".join(
                f"{row['count']}x{row['device']}/{row['dtype']}"
                for row in backbone_audit["by_device_dtype"]
            )
            + f"; head {head_audit['count']} tensors; "
            "off_device="
            f"{backbone_audit['off_device_count'] + head_audit['off_device_count']}",
            flush=True,
        )

        vocab_size = artifact_vocab_size(args.artifact)
        inputs = make_inputs(
            seed=args.seed,
            vocab_size=vocab_size,
            context_tokens=args.context_tokens,
            candidates=args.candidates,
            candidate_length=args.candidate_length,
        )
        receipt["inputs"] = dict(inputs["spec"])
        context = inputs["context"].to(device)
        candidates = [candidate.to(device) for candidate in inputs["candidates"]]

        # ---- transfer counters around prefill + candidate scoring -------
        counters.install()
        counters.enabled = True
        cat_audit.install()
        cat_audit.expected_device = target_device
        cache: Any = None
        timings: dict[str, float] = {}
        try:
            counters.phase = "prefill"
            started = time.perf_counter()
            with torch.inference_mode():
                cache = prefill_context(scorer.backbone, context, policy="last")
            sync_device(device)
            timings["prefill_s"] = round(time.perf_counter() - started, 6)
            receipt["phases"]["prefill"] = {
                "seconds": timings["prefill_s"],
                "mps_memory": device_memory(device),
            }
            counters.phase = "candidate_scoring"
            cat_audit.enabled = True
            started = time.perf_counter()
            with torch.inference_mode():
                logits = score_candidates_from_cache(
                    scorer.backbone, scorer.head, cache, candidates
                )
            sync_device(device)
            timings["candidate_scoring_s"] = round(time.perf_counter() - started, 6)
            cat_audit.enabled = False
            receipt["phases"]["candidate_scoring"] = {
                "seconds": timings["candidate_scoring_s"],
                "mps_memory": device_memory(device),
                "logits": [
                    float(value) for value in logits.detach().float().cpu().tolist()
                ],
            }
        finally:
            counters.enabled = False
            counters.uninstall()
            cat_audit.uninstall()
        receipt["phases"]["transfers"] = {
            "prefill_and_candidate": counters.report(),
            "cat_stage2": cat_audit.report(),
        }

        # ---- KV bank audit ----------------------------------------------
        bank_audit = inspect_context_banks(cache, target_device)
        receipt["kv_banks"] = bank_audit
        if bank_audit["off_device_count"]:
            receipt["anomalies"].append(
                f"{bank_audit['off_device_count']} KV bank tensors not on "
                f"{target_device}"
            )
        if bank_audit["noncontiguous_count"]:
            receipt["warnings"].append(
                f"{bank_audit['noncontiguous_count']} non-contiguous KV bank tensors"
            )
        if any(not entry["contiguous"] for entry in bank_audit["noncontiguous"]):
            pass
        print(
            f"[audit] KV banks: {bank_audit['banks']} bank(s) x "
            f"{bank_audit['layers_per_bank']} layers, length={bank_audit['length']}, "
            f"devices={bank_audit['devices']}, dtypes={bank_audit['dtypes']}, "
            f"bytes={bank_audit['total_bytes']}, contiguous={bank_audit['contiguous']}",
            flush=True,
        )

        # ---- profiler cross-check ---------------------------------------
        receipt["profiler_cross_check"] = profiler_transfer_ops(
            lambda: score_candidates_from_cache(
                scorer.backbone, scorer.head, cache, candidates
            ),
            device=device,
        )
        prof_ops = receipt["profiler_cross_check"]["ops"]
        print(
            "[audit] profiler transfer/sync ops: "
            + ", ".join(f"{row['op']}x{row['calls']}" for row in prof_ops[:8]),
            flush=True,
        )

        # ---- sync point analysis ----------------------------------------
        transfers = receipt["phases"]["transfers"]["prefill_and_candidate"]
        counts = transfers["counts"]
        sync_item = sum(
            value for key, value in counts.items() if key.startswith("sync:item:")
        )
        sync_tolist = sum(
            value for key, value in counts.items() if key.startswith("sync:tolist:")
        )
        device_changes = sum(
            value for key, value in counts.items() if key.endswith(":device_change")
        )
        same_attrs = sum(
            value for key, value in counts.items() if key.endswith(":same_attrs")
        )
        receipt["transfer_summary"] = {
            "python_level": {
                "sync_item_calls": sync_item,
                "sync_tolist_calls": sync_tolist,
                "device_change_calls": device_changes,
                "same_attrs_copies": same_attrs,
                "numpy_calls": counts.get("numpy:calls", 0),
            },
            "profiler_local_scalar_dense": sum(
                row["calls"] for row in prof_ops if "_local_scalar_dense" in row["op"]
            ),
            "profiler_to_copy": sum(
                row["calls"] for row in prof_ops if "_to_copy" in row["op"]
            ),
            "spurious_sync_points": [
                site
                for site in transfers["sites"]
                if site["kind"] in {"item", "tolist"}
            ],
            "call_sites": transfers["sites"],
        }
        if device_changes:
            receipt["warnings"].append(
                f"{device_changes} host-device transfer(s) during prefill+candidate "
                "scoring (see transfer_summary.call_sites)"
            )
        if sync_item + sync_tolist:
            receipt["warnings"].append(
                f"{sync_item + sync_tolist} explicit .item()/.tolist() sync point(s) "
                "during scoring"
            )
        if cat_audit.mixed_device_calls or cat_audit.mixed_dtype_calls:
            receipt["anomalies"].append(
                f"torch.cat mixed inputs: device={cat_audit.mixed_device_calls}, "
                f"dtype={cat_audit.mixed_dtype_calls}"
            )
        if cat_audit.off_device_calls:
            receipt["anomalies"].append(
                f"{cat_audit.off_device_calls} torch.cat output(s) off {target_device}"
            )
        if device == "mps":
            memory = device_memory("mps")
            current = memory.get("current_allocated_gib") or 0.0
            if current <= 0.5:
                receipt["anomalies"].append(
                    "MPS current_allocated_memory <= 0.5 GiB after load: weights may "
                    "not be resident on the Apple GPU"
                )
            if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") is not None:
                receipt["warnings"].append(
                    "PYTORCH_ENABLE_MPS_FALLBACK is set: ops may silently run on CPU"
                )
            for phase in ("prefill", "candidate_scoring"):
                memory_phase = receipt["phases"].get(phase, {}).get("mps_memory", {})
                receipt["phases"].setdefault("mps_memory", {})[phase] = memory_phase

        receipt["memory"] = {
            "peak_rss_gib": gib(peak_rss_bytes()),
            "current_rss_gib": gib(current_rss_bytes()),
            "rss_cap_gib": args.rss_cap_gib,
            "rss_cap_hit": bool(watchdog.triggered),
            "mps": device_memory(device),
        }
        receipt["provenance"]["source_end"] = source_fingerprint()
        receipt["provenance"]["source_changed_during_run"] = receipt["provenance"][
            "source_start"
        ].get("sha256") != receipt["provenance"]["source_end"].get("sha256")
        receipt["duration_s"] = round(time.perf_counter() - load_seconds, 3)
        receipt["passed"] = not receipt["anomalies"]
        receipt["status"] = "ok"
        receipt["finished_at"] = utc_now()
        exit_code = 0 if receipt["passed"] else EXIT_ANOMALY
    except RssCapExceeded as error:
        receipt["status"] = "aborted"
        receipt["error"] = str(error)
        receipt["finished_at"] = utc_now()
        exit_code = EXIT_RSS_CAP
    except Exception:
        receipt["status"] = "error"
        receipt["error"] = traceback.format_exc()
        receipt["finished_at"] = utc_now()
        exit_code = EXIT_USAGE
        traceback.print_exc()
    finally:
        watchdog.stop()
        counters.uninstall()
        cat_audit.uninstall()
        if scorer is not None:
            del scorer
        gc.collect()
        if args.device == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:  # pragma: no cover
                pass
        if lock is not None:
            lock.release()
            print(f"[lock] released {LOCK_DIR}", flush=True)

    if args.json_out:
        target = write_json(args.json_out, receipt)
        print(f"[receipt] audit receipt written to {target}", flush=True)
    print("-" * 78)
    print(
        "RESULT audit "
        + ("PASS" if receipt.get("passed") else "FAIL")
        + f" device={device}"
        + f" anomalies={len(receipt.get('anomalies', []))}"
        + f" warnings={len(receipt.get('warnings', []))}"
        + f" transfers={receipt.get('transfer_summary', {}).get('python_level')}"
        + f" peak_rss={receipt.get('memory', {}).get('peak_rss_gib')}GiB"
        + f" mps={receipt.get('memory', {}).get('mps', {})}"
    )
    for warning in receipt.get("warnings", []):
        print(f"RESULT warning: {warning}")
    for anomaly in receipt.get("anomalies", []):
        print(f"RESULT anomaly: {anomaly}")
    print("-" * 78)
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    return run_audit(args)


if __name__ == "__main__":
    raise SystemExit(main())
