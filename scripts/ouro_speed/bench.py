#!/usr/bin/env python3
"""Benchmark Ouro decision scoring on Apple silicon — profiling only.

This is the standardized harness described in ``scripts/ouro_speed/README.md``.
It measures, in one process:

  * load time (``load_ouro``)
  * Stage 1 shared-context prefill (``prefill_context``, policy=last)
  * Stage 2 candidate scoring from the retained cache
    (``score_candidates_from_cache``)
  * end-to-end ``score_candidates`` (Stage 1 + Stage 2)
  * per-candidate-count scaling and, optionally, a context-length grid

Mandatory resource caps (the Mac must stay usable and progress must be visible):

  * exclusive lock ``/tmp/ouro-speed/BENCH_LOCK`` (``mkdir``, 10 s retry,
    removed on exit; no second model load runs concurrently)
  * ``nice`` (default 10)
  * torch intra-op threads <= 6
  * resident-set watchdog that hard-aborts (exit 42) above
    ``--rss-cap-gib`` (default 10 GiB, absolute maximum 12 GiB)
  * MPS: process memory fraction <= 0.6 and
    ``PYTORCH_MPS_LOW_WATERMARK_RATIO=1.0`` /
    ``PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.2`` set before the MPS allocator
    initialises

Inputs are deterministic synthetic token IDs drawn from a seeded
``torch.Generator`` (no tokenizer or dataset needed); the exact generator spec
and a SHA-256 over the token bytes are recorded in the receipt.

Live progress lines look like::

  [   12.34s] rss_now= 8.12GiB rss_peak= 8.44GiB | stage=prefill iter=2/5 dt=1.961

The JSON receipt records environment, git revision + ``wyat/loading/ouro.py``
SHA-256 (re-hashed at the end to detect concurrent edits), artifact identity,
medians/min/max, memory high-water, logits and (optionally) an aten op
breakdown. A markdown sibling can be emitted with ``--markdown-out``.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from _common import (  # noqa: E402
    DEFAULT_ATOL,
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
    Progress,
    RssCapExceeded,
    RssWatchdog,
    apply_nice,
    artifact_fingerprint,
    artifact_vocab_size,
    compare_logits,
    configure_mps,
    configure_threads,
    current_rss_bytes,
    device_memory,
    environment_snapshot,
    format_comparison,
    gib,
    load_ouro_scorer,
    make_inputs,
    peak_rss_bytes,
    prescan_env,
    source_fingerprint,
    summarize_samples,
    sync_device,
    write_json,
)  # noqa: E402

prescan_env(sys.argv)  # thread + MPS watermark env vars must precede torch import

import torch  # noqa: E402

from wyat.loading.ouro import (  # noqa: E402
    prefill_context,
    score_candidates,
    score_candidates_from_cache,
)

EXIT_LOCK_TIMEOUT = 3
EXIT_UNSTABLE_LOADER = 4
EXIT_GOLDEN_MISMATCH = 5
EXIT_SELF_INCONSISTENT = 6


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def load_tokens_inputs(args: argparse.Namespace) -> dict[str, Any]:
    """Load frozen context/candidate token IDs from a question-set JSON."""
    from _common import file_sha256

    target = Path(args.tokens_file)
    if not target.exists():
        raise SystemExit(f"--tokens-file not found: {target}")
    payload = json.loads(target.read_text(encoding="utf-8"))
    selection: dict[str, Any] = {}
    if args.tokens_name is not None:
        entries = payload.get("headline")
        if not isinstance(entries, dict) or args.tokens_name not in entries:
            raise SystemExit(
                f"--tokens-name {args.tokens_name!r} not in {target} headline entries"
            )
        entry = entries[args.tokens_name]
        selection = {"section": "headline", "name": args.tokens_name}
    else:
        entries = payload.get(args.tokens_section)
        if not isinstance(entries, list):
            raise SystemExit(
                f"--tokens-file has no list section {args.tokens_section!r}"
            )
        index = int(args.tokens_index)
        if not 0 <= index < len(entries):
            raise SystemExit(
                f"--tokens-index {index} out of range for section "
                f"{args.tokens_section!r} ({len(entries)} entries)"
            )
        entry = entries[index]
        selection = {
            "section": args.tokens_section,
            "index": index,
            "id": entry.get("id"),
        }
    if (
        not isinstance(entry, dict)
        or "context" not in entry
        or "candidates" not in entry
    ):
        raise SystemExit(f"{target}: selected entry lacks context/candidates token IDs")
    context_list = [int(value) for value in entry["context"]]
    candidate_lists = [
        [int(value) for value in candidate] for candidate in entry["candidates"]
    ]
    if (
        not context_list
        or not candidate_lists
        or any(not candidate for candidate in candidate_lists)
    ):
        raise SystemExit(f"{target}: selected entry has empty token lists")
    context = torch.tensor([context_list], dtype=torch.long)
    candidate_tensors = [
        torch.tensor([candidate], dtype=torch.long) for candidate in candidate_lists
    ]
    core_spec: dict[str, Any] = {
        "generator": "frozen token IDs from --tokens-file (no tokenizer/dataset)",
        "seed": entry.get("seed"),
        "context_tokens": len(context_list),
        "candidate_count": len(candidate_lists),
        "candidate_lengths": [len(candidate) for candidate in candidate_lists],
        "policy": args.policy,
        "tokens_file": str(target.resolve()),
        "tokens_file_sha256": file_sha256(target),
        "question_set_sha256": payload.get("sha256"),
        "selection": selection,
        "note": (
            "deterministic frozen token IDs loaded verbatim; not generated by "
            "this process"
        ),
    }
    from _common import fingerprint_token_inputs

    spec = fingerprint_token_inputs(core_spec, context, candidate_tensors)
    return {
        "spec": spec,
        "context": context,
        "candidates": candidate_tensors,
        "sha256": spec["input_sha256"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark an Ouro decision artifact (profiling only). Enforces the "
            "exclusive BENCH_LOCK, nice 10, <=6 threads and a hard RSS watchdog."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--artifact",
        required=True,
        help="path to a wyat-ouro-decision-v1 artifact directory",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    parser.add_argument(
        "--allow-device-override",
        action="store_true",
        help=(
            "pass the loader's explicit allow_device_policy_override opt-in "
            "(needed to run a cpu-fp32 artifact on MPS after the device gate "
            "landed)"
        ),
    )
    parser.add_argument(
        "--policy-override",
        dest="allow_device_override",
        action="store_true",
        help=(
            "alias for --allow-device-override: forward "
            "allow_device_policy_override=True for cross-device-policy runs"
        ),
    )
    parser.add_argument(
        "--candidate-batching",
        action="store_true",
        help=(
            "forward the runtime's opt-in candidate_batching flag when "
            "load_ouro supports it (default off = standardized baseline)"
        ),
    )
    parser.add_argument(
        "--device-policy-override",
        dest="allow_device_override",
        action="store_true",
        help="alias for --allow-device-override (cross-device-policy arms)",
    )
    parser.add_argument(
        "--tokens-file",
        default=None,
        help=(
            "frozen question-set JSON (e.g. artifacts/.../mps/inputs/"
            "question-set.json) to use verbatim instead of synthetic tokens"
        ),
    )
    parser.add_argument(
        "--tokens-name",
        default=None,
        help="headline entry name inside --tokens-file (e.g. p400-k6-s16)",
    )
    parser.add_argument(
        "--tokens-section",
        default=None,
        choices=["corpus", "real", "headline"],
        help="list section inside --tokens-file (with --tokens-index)",
    )
    parser.add_argument("--tokens-index", type=int, default=None)
    parser.add_argument(
        "--check-baseline",
        default=None,
        help=(
            "golden receipt or logits file; exit nonzero if candidate logits "
            "differ from it by more than 2e-5 abs or rank order/top-1 changes"
        ),
    )
    parser.add_argument(
        "--baseline-key",
        default=None,
        help=(
            "dot path into --check-baseline JSON (e.g. 'headline_logits.p400-k6-s16')"
        ),
    )
    parser.add_argument(
        "--save-golden",
        default=None,
        help="write a golden receipt with this run's full-precision logits",
    )
    parser.add_argument(
        "--skip-self-consistency",
        action="store_true",
        help=(
            "skip the multi-candidate-vs-per-candidate self-consistency gate "
            "(default: run it; it catches pooled-KV corruption for K>=2)"
        ),
    )
    # caps
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help=f"torch intra-op threads, hard cap {MAX_THREADS}",
    )
    parser.add_argument("--nice", type=int, default=10, help="process niceness 0..20")
    parser.add_argument(
        "--rss-cap-gib",
        type=float,
        default=DEFAULT_RSS_CAP_GIB,
        help=f"hard RSS abort cap in GiB (max {MAX_RSS_CAP_GIB})",
    )
    parser.add_argument(
        "--mps-fraction",
        type=float,
        default=DEFAULT_MPS_FRACTION,
        help=f"MPS process memory fraction, hard cap {MAX_MPS_FRACTION}",
    )
    parser.add_argument("--no-lock", action="store_true", help="skip BENCH_LOCK")
    parser.add_argument(
        "--lock-timeout-s",
        type=float,
        default=DEFAULT_LOCK_TIMEOUT_S,
        help="max seconds to wait for the exclusive lock (retry every 10 s)",
    )
    parser.add_argument(
        "--watchdog-interval-s",
        type=float,
        default=0.25,
        help="RSS watchdog polling interval",
    )
    # workload
    parser.add_argument("--context-tokens", type=int, default=400)
    parser.add_argument("--candidates", type=int, default=6)
    parser.add_argument("--candidate-length", type=int, default=16)
    parser.add_argument(
        "--candidate-lengths",
        default=None,
        help="comma list of per-candidate lengths (overrides --candidate-length)",
    )
    parser.add_argument(
        "--policy",
        default="last",
        choices=["last", "matched"],
        help="ContextKVCache policy for Stage 1",
    )
    # timing
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261021)
    # scaling
    parser.add_argument(
        "--scale-candidates",
        default="",
        help='comma list of candidate counts for Stage-2 scaling, e.g. "1,2,4,6,8"',
    )
    parser.add_argument(
        "--scale-contexts",
        default="",
        help=(
            "comma list of context lengths for the prefill/scaling grid, "
            'e.g. "256,400,832"'
        ),
    )
    parser.add_argument("--scale-repeats", type=int, default=3)
    parser.add_argument("--scale-warmups", type=int, default=1)
    parser.add_argument(
        "--scale-e2e",
        action="store_true",
        help="also time end-to-end for every context length in the grid",
    )
    # reporting
    parser.add_argument("--json-out", default=None, help="receipt JSON output path")
    parser.add_argument(
        "--markdown-out", default=None, help="human-readable markdown receipt path"
    )
    parser.add_argument(
        "--save-logits",
        default=None,
        help="save logits tensor (.pt/.pth) or JSON (.json) for parity checks",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="collect a torch.profiler aten op breakdown of Stage 2 (CPU activity)",
    )
    parser.add_argument("--profile-top", type=int, default=15)
    parser.add_argument("--profile-iters", type=int, default=3)
    return parser


def parse_lengths(text: str) -> list[int]:
    values = [part.strip() for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("empty comma list")
    parsed = [int(value) for value in values]
    if any(value <= 0 for value in parsed):
        raise ValueError("values must be positive integers")
    return parsed


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not 1 <= args.threads <= MAX_THREADS:
        parser.error(f"--threads must be in 1..{MAX_THREADS} (mandatory cap)")
    if not 0 <= args.nice <= 20:
        parser.error("--nice must be in 0..20")
    if not 0 < args.rss_cap_gib <= MAX_RSS_CAP_GIB:
        parser.error(f"--rss-cap-gib must be in (0, {MAX_RSS_CAP_GIB}] (hard ceiling)")
    if not 0 < args.mps_fraction <= MAX_MPS_FRACTION:
        parser.error(f"--mps-fraction must be in (0, {MAX_MPS_FRACTION}] (hard cap)")
    if args.repeats < 1:
        parser.error("--repeats must be >= 1 (protocol requires >= 5)")
    if args.warmups < 0:
        parser.error("--warmups must be >= 0")
    if args.warmups < 1:
        print(
            "[caps] warning: --warmups 0 violates the standardized protocol "
            "(required >= 1, protocol default 2)",
            file=sys.stderr,
            flush=True,
        )
    if args.repeats < 5:
        print(
            f"[caps] warning: --repeats {args.repeats} < 5; standardized protocol "
            "requires >= 5 repetitions",
            file=sys.stderr,
            flush=True,
        )
    if args.context_tokens <= 0:
        parser.error("--context-tokens must be positive")
    if args.candidates <= 0:
        parser.error("--candidates must be positive")
    if args.candidate_length <= 0:
        parser.error("--candidate-length must be positive")
    if args.candidate_lengths is not None:
        try:
            lengths = parse_lengths(args.candidate_lengths)
        except ValueError as error:
            parser.error(f"--candidate-lengths: {error}")
        if len(lengths) != args.candidates:
            parser.error(
                f"--candidate-lengths has {len(lengths)} entries but --candidates "
                f"is {args.candidates}"
            )
    for flag, text in (
        ("--scale-candidates", args.scale_candidates),
        ("--scale-contexts", args.scale_contexts),
    ):
        if not text:
            continue
        try:
            parse_lengths(text)
        except ValueError as error:
            parser.error(f"{flag}: {error}")
    if args.tokens_file:
        if not Path(args.tokens_file).exists():
            parser.error(f"--tokens-file not found: {args.tokens_file}")
        if args.tokens_name is None and (
            args.tokens_section is None or args.tokens_index is None
        ):
            parser.error(
                "--tokens-file requires --tokens-name or "
                "--tokens-section with --tokens-index"
            )
        if args.scale_contexts:
            parser.error(
                "--tokens-file cannot be combined with --scale-contexts "
                "(the grid synthesises its own contexts)"
            )
    elif args.tokens_name or args.tokens_section or args.tokens_index is not None:
        parser.error("--tokens-name/--tokens-section/--tokens-index need --tokens-file")
    if args.baseline_key and not args.check_baseline:
        parser.error("--baseline-key requires --check-baseline")
    if args.scale_repeats < 1:
        parser.error("--scale-repeats must be >= 1")
    if args.profile_iters < 1 or args.profile_top < 1:
        parser.error("--profile-iters and --profile-top must be >= 1")
    if args.json_out:
        json_path = Path(args.json_out)
        if json_path.resolve() == Path(args.markdown_out or "").resolve():
            parser.error("--json-out and --markdown-out must differ")


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.gmtime())


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------


class Measurer:
    """Wall-clock timing with device synchronisation around each call."""

    def __init__(self, device: str) -> None:
        self.device = str(device).lower()

    def call(self, fn: Any) -> tuple[float, Any]:
        sync_device(self.device)
        started = time.perf_counter()
        try:
            result = fn()
        finally:
            sync_device(self.device)
        return time.perf_counter() - started, result

    def repeat(
        self,
        stage: str,
        fn: Any,
        *,
        warmups: int,
        repeats: int,
        progress: Progress,
        watchdog: RssWatchdog,
        result_namespace: dict[str, Any],
        result_key: str,
        extra_fields: dict[str, Any] | None = None,
    ) -> tuple[list[float], Any, dict[str, Any]]:
        """Warmup + timed repeats, live progress lines, receipt stage summary."""
        extra = extra_fields or {}
        last: Any = None
        for index in range(warmups):
            seconds, last = self.call(fn)
            progress.line(
                f"stage={stage} warmup={index + 1}/{warmups}",
                dt=seconds,
                **extra,
            )
        samples: list[float] = []
        for index in range(repeats):
            watchdog.check()
            seconds, last = self.call(fn)
            samples.append(seconds)
            result_namespace[result_key] = last
            progress.line(
                f"stage={stage} iter={index + 1}/{repeats}",
                dt=seconds,
                **extra,
            )
        summary: dict[str, Any] = {
            **summarize_samples(samples),
            "warmups": warmups,
            "mps_memory_after": device_memory(self.device),
        }
        summary.update({f"extra_{key}": value for key, value in extra.items()})
        return samples, last, summary


def to_float_list(tensor: Any) -> list[float]:
    return [float(value) for value in tensor.detach().float().cpu().tolist()]


def logits_digest(logits: Any) -> dict[str, Any]:
    import hashlib

    values = to_float_list(logits)
    digest = hashlib.sha256()
    for value in values:
        digest.update(repr(value).encode("ascii"))
        digest.update(b"\0")
    return {"values": values, "sha256": digest.hexdigest()}


def save_logits_file(
    path: str | Path,
    *,
    logits: Any,
    e2e_logits: Any | None,
    spec: dict[str, Any],
    model_id: str | None,
    source_sha256: str | None,
    device: str,
) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "logits": to_float_list(logits),
        "e2e_logits": to_float_list(e2e_logits) if e2e_logits is not None else None,
        "input_sha256": spec.get("input_sha256"),
        "input_spec": spec,
        "artifact_model_id": model_id,
        "source_ouro_sha256": source_sha256,
        "device": device,
        "schema": "ouro-speed-logits/v1",
    }
    if target.suffix.lower() in {".pt", ".pth"}:
        torch.save(payload, target)
    elif target.suffix.lower() == ".json":
        write_json(target, payload)
    else:
        raise SystemExit(f"--save-logits must end in .pt/.pth/.json: {target}")
    return {
        "path": str(target),
        "sha256": _sha256_file(target),
        "format": target.suffix.lstrip("."),
        "logits_sha256": payload["logits"] and _sha256_floats(payload["logits"]),
    }


def _sha256_file(path: str | Path) -> str:
    from _common import file_sha256

    return file_sha256(path)


def _sha256_floats(values: Sequence[float]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for value in values:
        digest.update(repr(float(value)).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _dot_path(payload: Any, key: str) -> Any:
    current = payload
    for part in key.split("."):
        if isinstance(current, dict):
            if part not in current:
                raise SystemExit(f"--baseline-key {key!r}: missing key {part!r}")
            current = current[part]
        elif isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError) as error:
                raise SystemExit(
                    f"--baseline-key {key!r}: bad list index {part!r}"
                ) from error
        else:
            raise SystemExit(f"--baseline-key {key!r}: cannot descend at {part!r}")
    return current


def load_golden_logits(
    path: str | Path, *, key: str | None = None
) -> tuple[list[float], dict[str, Any]]:
    """Load expected logits from a golden receipt, logits file or question-set.

    ``key`` is an optional dot path into a JSON payload (for example
    ``headline_logits.p400-k6-s16`` in mps-probe's reference file).
    """
    from _common import file_sha256

    target = Path(path)
    if not target.exists():
        raise SystemExit(f"--check-baseline not found: {target}")
    suffix = target.suffix.lower()
    if suffix in {".pt", ".pth"}:
        payload = torch.load(target, map_location="cpu", weights_only=False)
    elif suffix == ".json":
        payload = json.loads(target.read_text(encoding="utf-8"))
    else:
        raise SystemExit(f"unsupported --check-baseline suffix: {target.suffix}")
    metadata: dict[str, Any] = {"path": str(target), "sha256": file_sha256(target)}
    if key is not None:
        metadata["key"] = key
        payload = _dot_path(payload, key)
    values: Any
    if isinstance(payload, dict):
        if "logits" in payload:
            values = payload["logits"]
            if isinstance(values, dict) and "values" in values:
                values = values["values"]
        elif "values" in payload and "logits" not in payload:
            values = payload["values"]
        else:
            raise SystemExit(f"{target}: no 'logits'/'values' field")
        for key in ("inputs", "input_sha256", "artifact", "scorer", "tolerance"):
            if key in payload:
                metadata[key] = payload[key]
    else:
        values = payload
    if isinstance(values, torch.Tensor):
        values = values.detach().float().cpu().reshape(-1).tolist()
    if not isinstance(values, (list, tuple)):
        raise SystemExit(f"{target}: unsupported golden logits type")
    return [float(value) for value in values], metadata


def build_golden_payload(
    receipt: dict[str, Any], logits_values: Sequence[float]
) -> dict[str, Any]:
    """Canonical golden receipt payload for --check-baseline."""
    return {
        "schema": "ouro-speed-golden/v1",
        "kind": "golden",
        "created_at": utc_now(),
        "notice": PROFILING_NOTICE,
        "tolerance": {
            "abs": DEFAULT_ATOL,
            "rank_top1_required": True,
            "gating": "max_abs_diff <= abs and identical rank order/top-1",
        },
        "artifact": receipt.get("provenance", {}).get("artifact"),
        "scorer": receipt.get("scorer"),
        "inputs": receipt.get("inputs"),
        "logits": [float(value) for value in logits_values],
        "workload": {
            "context_tokens": receipt.get("inputs", {}).get("context_tokens"),
            "candidate_count": receipt.get("inputs", {}).get("candidate_count"),
            "candidate_lengths": receipt.get("inputs", {}).get("candidate_lengths"),
            "seed": receipt.get("inputs", {}).get("seed"),
        },
        "source": {
            "command": receipt.get("command"),
            "source_end": receipt.get("provenance", {}).get("source_end"),
            "self_consistency_passed": (
                receipt.get("self_consistency", {}).get("passed")
            ),
        },
        "derived_from_receipt_caps": receipt.get("caps"),
    }


def render_markdown(receipt: dict[str, Any]) -> str:
    """Human-readable markdown rendering of a bench receipt."""
    lines: list[str] = []
    stages = receipt.get("stages", {})
    lines.append("# Ouro speed receipt")
    lines.append("")
    lines.append(f"- schema: `{receipt.get('schema')}`")
    lines.append(f"- generated: {receipt.get('generated_at')}")
    lines.append(f"- status: **{receipt.get('status')}**")
    lines.append(f"- notice: {receipt.get('notice')}")
    for key in ("device", "artifact", "model_id", "threads", "nice", "rss_cap_gib"):
        value = receipt.get("caps", {}).get(
            key, receipt.get("artifact_info", {}).get(key)
        )
        if value is not None:
            lines.append(f"- {key}: `{value}`")
    environment = receipt.get("environment", {})
    if environment:
        lines.append(
            f"- machine: {environment.get('machine')} / {environment.get('mac_ver')} "
            f"/ {environment.get('cpu_count')} cores / "
            f"{environment.get('total_memory_gib')} GiB / torch "
            f"{environment.get('torch_version')}"
        )
    source = receipt.get("provenance", {}).get("source_end", {})
    if source:
        lines.append(
            f"- `wyat/loading/ouro.py` sha256: `{source.get('sha256')}` "
            f"(git {source.get('git_revision')}, dirty={source.get('git_dirty')})"
        )
    inputs = receipt.get("inputs", {})
    if inputs:
        lines.append(
            f"- inputs: seed={inputs.get('seed')} "
            f"context={inputs.get('context_tokens')} "
            f"candidates={inputs.get('candidate_count')} "
            f"lengths={inputs.get('candidate_lengths')} "
            f"sha256=`{inputs.get('input_sha256')}`"
        )
    protocol = receipt.get("protocol", {})
    if protocol:
        lines.append(
            f"- protocol: warmups={protocol.get('warmups')} "
            f"repeats={protocol.get('repeats')} "
            f"sync={protocol.get('sync')}"
        )
    lines.append("")
    lines.append("## Stages")
    lines.append("")
    lines.append("| stage | median s | min s | max s | samples |")
    lines.append("| :--- | ---: | ---: | ---: | ---: |")
    for name, summary in stages.items():
        if not isinstance(summary, dict) or "median_s" not in summary:
            continue
        lines.append(
            f"| {name} | {summary['median_s']:.4f} | {summary['min_s']:.4f} | "
            f"{summary['max_s']:.4f} | {summary['n']} |"
        )
    memory = receipt.get("memory", {})
    if memory:
        lines.append("")
        lines.append("## Memory")
        lines.append("")
        lines.append(f"- peak RSS: {memory.get('peak_rss_gib')} GiB")
        lines.append(
            f"- cap: {memory.get('rss_cap_gib')} GiB (hit={memory.get('rss_cap_hit')})"
        )
        mps = memory.get("mps") or {}
        if mps:
            lines.append(f"- MPS fraction: {mps.get('fraction')}")
            lines.append(f"- MPS memory after load: {mps.get('after_load')}")
            lines.append(f"- MPS memory at end: {mps.get('at_end')}")
    scaling = receipt.get("scaling", {})
    candidate_scaling = scaling.get("candidate_count", {})
    if candidate_scaling:
        lines.append("")
        lines.append("## Candidate-count scaling")
        lines.append("")
        lines.append("| candidates K | median s | min s | max s |")
        lines.append("| ---: | ---: | ---: | ---: |")
        for key in sorted(candidate_scaling, key=lambda item: int(item)):
            summary = candidate_scaling[key]
            lines.append(
                f"| {key} | {summary['median_s']:.4f} | {summary['min_s']:.4f} | "
                f"{summary['max_s']:.4f} |"
            )
    grid = scaling.get("context_grid", {})
    if grid:
        k_lists = sorted(
            {
                int(k)
                for context in grid.values()
                for k in context.get("candidate_from_cache_by_k", {})
            }
        )
        lines.append("")
        lines.append("## Context grid (median seconds)")
        lines.append("")
        header = "| context | prefill | " + " | ".join(f"K={k}" for k in k_lists) + " |"
        divider = "| ---: | ---: | " + " | ".join("---:" for _ in k_lists) + " |"
        lines.append(header)
        lines.append(divider)
        for context_length in sorted(grid, key=lambda item: int(item)):
            entry = grid[context_length]
            prefill = entry.get("prefill", {}).get("median_s")
            cells = []
            for k in k_lists:
                summary = entry.get("candidate_from_cache_by_k", {}).get(str(k))
                cells.append(f"{summary['median_s']:.4f}" if summary else "-")
            lines.append(
                f"| {context_length} | {prefill:.4f} | " + " | ".join(cells) + " |"
            )
    logits = receipt.get("logits", {})
    if logits.get("values"):
        lines.append("")
        lines.append("## Logits")
        lines.append("")
        lines.append("| candidate | score |")
        lines.append("| ---: | ---: |")
        for index, value in enumerate(logits["values"]):
            lines.append(f"| {index} | {value:+.8f} |")
        lines.append("")
        lines.append(f"logits sha256: `{logits.get('sha256')}`")
    profile = receipt.get("profile", {})
    if profile.get("top_ops"):
        lines.append("")
        lines.append("## Top aten ops (Stage 2, CPU activity)")
        lines.append("")
        lines.append("| op | self ms | calls |")
        lines.append("| :--- | ---: | ---: |")
        for row in profile["top_ops"]:
            lines.append(
                f"| `{row['op']}` | {row['self_cpu_ms']:.2f} | {row['calls']} |"
            )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Benchmark execution
# ---------------------------------------------------------------------------


def run_bench(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    validate_args(parser, args)
    device = args.device.lower()
    run_started = time.perf_counter()

    print("=" * 78)
    print(f"{PROFILING_NOTICE}")
    print("=" * 78)
    print(
        f"[caps] nice={args.nice} threads={args.threads} "
        f"rss_cap={args.rss_cap_gib}GiB (hard max {MAX_RSS_CAP_GIB}) "
        f"mps_fraction={args.mps_fraction} device={device}",
        flush=True,
    )
    apply_nice(args.nice)
    threads_info = configure_threads(args.threads)
    mps_info = configure_mps(args.mps_fraction) if device == "mps" else None
    env = environment_snapshot()
    env["threads"] = threads_info
    env["mps_configured"] = mps_info
    env["loadavg_at_start"] = env.get("loadavg_at_start")

    lock = None
    lock_waited = 0.0
    if not args.no_lock:
        from _common import BenchLock

        lock = BenchLock(
            LOCK_DIR, timeout_s=args.lock_timeout_s, retry_s=10.0, log=print
        )
        try:
            lock_waited = lock.acquire()
        except LockTimeout as error:
            print(f"[lock] ERROR: {error}", file=sys.stderr, flush=True)
            return EXIT_LOCK_TIMEOUT
    else:
        print("[lock] --no-lock: caller must hold /tmp/ouro-speed/BENCH_LOCK")

    progress = Progress(device=device)
    source_start = source_fingerprint()
    artifact_info = artifact_fingerprint(args.artifact)
    if not artifact_info.get("exists"):
        print(
            f"ERROR: artifact not found: {args.artifact}", file=sys.stderr, flush=True
        )
        if lock is not None:
            lock.release()
        return 2
    vocab_size = artifact_vocab_size(args.artifact)

    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
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
            "watchdog_interval_s": args.watchdog_interval_s,
            "mps_fraction": args.mps_fraction if device == "mps" else None,
            "mps_memory_fraction": mps_info,
            "sequential_only": True,
            "lock": {
                "path": str(LOCK_DIR),
                "held": lock is not None,
                "waited_s": round(lock_waited, 3),
            },
            "device": device,
            "allow_device_override": bool(args.allow_device_override),
            "candidate_batching": bool(args.candidate_batching),
        },
        "provenance": {
            "source_start": source_start,
            "artifact": artifact_info,
        },
        "stages": {},
        "scaling": {},
        "logits": {},
        "memory": {"rss_cap_gib": args.rss_cap_gib},
    }

    def write_abort_receipt(report: dict[str, Any]) -> None:
        receipt.update(
            {
                "status": "aborted",
                "abort_reason": "rss_cap_exceeded",
                "rss_abort": report,
                "finished_at": utc_now(),
                "provenance_end": {
                    "source_end": source_fingerprint(),
                },
                "memory": {
                    "rss_cap_gib": args.rss_cap_gib,
                    "peak_rss_gib": gib(peak_rss_bytes()),
                    "current_rss_gib": gib(current_rss_bytes()),
                    "rss_cap_hit": True,
                },
                "protocol": {
                    "warmups": args.warmups,
                    "repeats": args.repeats,
                    "policy": args.policy,
                    "sync": "torch.mps.synchronize when device=mps",
                },
                "inputs": {},
            }
        )
        if args.json_out:
            try:
                target = write_json(args.json_out, receipt)
                print(f"[watchdog] abort receipt written to {target}", file=sys.stderr)
            except Exception:  # pragma: no cover - abort path must not crash
                traceback.print_exc()
        if args.markdown_out:
            try:
                Path(args.markdown_out).write_text(
                    render_markdown(receipt), encoding="utf-8"
                )
            except Exception:  # pragma: no cover
                traceback.print_exc()

    watchdog = RssWatchdog(
        args.rss_cap_gib,
        interval_s=args.watchdog_interval_s,
        hard=True,
        on_exceed=write_abort_receipt,
    )

    measurer = Measurer(device)
    scorer = None
    exit_code = 0
    try:
        watchdog.start()
        # ---- Stage 0: load -------------------------------------------------
        progress.line(
            "stage=load begin",
            artifact=Path(args.artifact).name,
            device=device,
        )
        seconds, scorer = measurer.call(
            lambda: load_ouro_scorer(
                args.artifact,
                device=device,
                allow_device_override=args.allow_device_override,
                candidate_batching=args.candidate_batching,
            )
        )
        load_peak_gib = gib(peak_rss_bytes())
        receipt["stages"]["load"] = {
            "seconds": round(seconds, 6),
            "peak_rss_gib": load_peak_gib,
            "mps_memory_after": device_memory(device),
        }
        receipt["scorer"] = {
            "device": str(scorer.device),
            "device_policy": getattr(scorer, "device_policy", None),
            "candidate_batching": bool(
                getattr(scorer, "candidate_batching", args.candidate_batching)
            ),
            "model_id": scorer.model_id,
            "max_length": int(scorer.max_length),
            "backbone_dtype": str(next(scorer.backbone.parameters()).dtype),
            "head_dtype": str(next(scorer.head.parameters()).dtype),
        }
        receipt["memory"]["after_load"] = {
            "peak_rss_gib": load_peak_gib,
            "current_rss_gib": gib(current_rss_bytes()),
        }
        if device == "mps":
            receipt["memory"]["mps"] = {
                "fraction": args.mps_fraction,
                "after_load": device_memory("mps"),
            }
        progress.line(
            "stage=load done",
            load_s=seconds,
            dtype=receipt["scorer"]["backbone_dtype"],
        )
        if str(scorer.device).split(":")[0] != device:
            raise RuntimeError(
                f"loaded scorer on {scorer.device} but requested device={device}"
            )

        # ---- Inputs --------------------------------------------------------
        if args.tokens_file:
            inputs = load_tokens_inputs(args)
            progress.line(
                "stage=inputs loaded from frozen tokens",
                file=Path(args.tokens_file).name,
                selection=json.dumps(inputs["spec"]["selection"]),
            )
        else:
            inputs = make_inputs(
                seed=args.seed,
                vocab_size=vocab_size,
                context_tokens=args.context_tokens,
                candidates=args.candidates,
                candidate_length=args.candidate_length,
                candidate_lengths=parse_lengths(args.candidate_lengths)
                if args.candidate_lengths
                else None,
                policy=args.policy,
            )
        spec = dict(inputs["spec"])
        spec["artifact_vocab_size"] = vocab_size
        spec["artifact_model_id"] = scorer.model_id
        context = inputs["context"].to(device)
        candidates = [candidate.to(device) for candidate in inputs["candidates"]]
        receipt["inputs"] = spec
        receipt["protocol"] = {
            "warmups": args.warmups,
            "repeats": args.repeats,
            "policy": args.policy,
            "sync": "torch.mps.synchronize when device=mps",
            "clock": "time.perf_counter",
            "sequential_only": True,
            "candidate_lengths": spec["candidate_lengths"],
        }
        progress.line(
            "stage=inputs ready",
            context=args.context_tokens,
            candidates=args.candidates,
            lengths=",".join(str(length) for length in spec["candidate_lengths"]),
            input_sha=spec["input_sha256"][:12],
        )

        # ---- Stage 1: shared-context prefill ------------------------------
        with torch.inference_mode():
            cache_holder: dict[str, Any] = {}

            def prefill() -> Any:
                cache_holder["cache"] = prefill_context(
                    scorer.backbone, context, policy=args.policy
                )
                return cache_holder["cache"]

            _, _, prefill_summary = measurer.repeat(
                "prefill_stage1",
                prefill,
                warmups=args.warmups,
                repeats=args.repeats,
                progress=progress,
                watchdog=watchdog,
                result_namespace=cache_holder,
                result_key="cache",
            )
            cache = cache_holder["cache"]
            prefill_summary.update(
                {
                    "cache_policy": cache.policy,
                    "cache_length": int(cache.length),
                    "cache_banks": len(cache.banks),
                    "cache_layers_per_bank": len(cache.banks[0]) if cache.banks else 0,
                }
            )
            receipt["stages"]["prefill_stage1"] = prefill_summary
            progress.line(
                "stage=prefill_stage1 done",
                median=prefill_summary["median_s"],
                cache_len=cache.length,
            )

            # ---- Stage 2: candidate scoring from cache --------------------
            def candidates_from_cache() -> Any:
                return score_candidates_from_cache(
                    scorer.backbone,
                    scorer.head,
                    cache,
                    candidates,
                    candidate_batching=args.candidate_batching,
                )

            _, cached_logits, candidate_summary = measurer.repeat(
                "candidate_scoring_from_cache",
                candidates_from_cache,
                warmups=args.warmups,
                repeats=args.repeats,
                progress=progress,
                watchdog=watchdog,
                result_namespace={},
                result_key="logits",
            )
            receipt["stages"]["candidate_scoring_from_cache"] = candidate_summary
            progress.line(
                "stage=candidate_scoring_from_cache done",
                median=candidate_summary["median_s"],
            )

            # ---- Stage 3: end-to-end --------------------------------------
            def end_to_end() -> Any:
                return score_candidates(
                    scorer.backbone,
                    scorer.head,
                    context,
                    candidates,
                    policy=args.policy,
                    candidate_batching=args.candidate_batching,
                )

            _, e2e_logits, e2e_summary = measurer.repeat(
                "end_to_end",
                end_to_end,
                warmups=args.warmups,
                repeats=args.repeats,
                progress=progress,
                watchdog=watchdog,
                result_namespace={},
                result_key="logits",
            )
            receipt["stages"]["end_to_end"] = e2e_summary
            consistency = compare_logits(
                to_float_list(cached_logits),
                to_float_list(e2e_logits),
                atol=0.0,
                label_a="cached",
                label_b="end_to_end",
            )
            receipt["stages"]["cached_vs_e2e"] = {
                "bitwise_equal": consistency["bitwise_equal"],
                "max_abs_diff": consistency["max_abs_diff"],
                "same_inputs": spec["input_sha256"],
            }
            progress.line(
                "stage=end_to_end done",
                median=e2e_summary["median_s"],
                cached_e2e_bitwise=consistency["bitwise_equal"],
            )

            # ---- multi-candidate self-consistency gate -------------------
            # The pooled-KV bug class corrupts K>=2 multi-candidate calls while
            # K=1 stays exact. Score every candidate alone and require the same
            # logits; this catches corruption without relying on a golden file.
            if not args.skip_self_consistency:
                started_consistency = time.perf_counter()
                single_logits: list[float] = []
                for candidate in candidates:
                    single = score_candidates_from_cache(
                        scorer.backbone,
                        scorer.head,
                        cache,
                        [candidate],
                        candidate_batching=args.candidate_batching,
                    )
                    single_logits.append(float(single[0]))
                consistency_seconds = time.perf_counter() - started_consistency
                self_report = compare_logits(
                    to_float_list(cached_logits),
                    single_logits,
                    atol=DEFAULT_ATOL,
                    rank_fail=True,
                    label_a="multi_candidate_call",
                    label_b="per_candidate_calls",
                )
                receipt["self_consistency"] = {
                    "seconds": round(consistency_seconds, 6),
                    "per_candidate_logits": single_logits,
                    "comparison": self_report,
                    "passed": self_report["passed"],
                }
                progress.line(
                    "stage=self_consistency done",
                    passed=self_report["passed"],
                    max_diff=self_report["max_abs_diff"],
                )

            # ---- optional: per-candidate scaling --------------------------
            if args.scale_candidates:
                scale_ks = parse_lengths(args.scale_candidates)
                candidate_scaling: dict[str, Any] = {}
                for k in scale_ks:
                    if k > len(candidates):
                        print(
                            f"[scale] skip K={k}: only {len(candidates)} candidates "
                            "generated",
                            file=sys.stderr,
                            flush=True,
                        )
                        continue
                    subset = candidates[:k]
                    _, _, summary = measurer.repeat(
                        f"scale_k{k}",
                        lambda sub=subset: score_candidates_from_cache(
                            scorer.backbone,
                            scorer.head,
                            cache,
                            sub,
                            candidate_batching=args.candidate_batching,
                        ),
                        warmups=args.scale_warmups,
                        repeats=args.scale_repeats,
                        progress=progress,
                        watchdog=watchdog,
                        result_namespace={},
                        result_key="logits",
                    )
                    candidate_scaling[str(k)] = summary
                receipt["scaling"]["candidate_count"] = candidate_scaling
                progress.line(
                    "stage=scaling candidate_count done",
                    ks=",".join(sorted(candidate_scaling, key=int)),
                )

            # ---- optional: context grid -----------------------------------
            if args.scale_contexts:
                contexts = parse_lengths(args.scale_contexts)
                scale_ks = (
                    parse_lengths(args.scale_candidates)
                    if args.scale_candidates
                    else [1, 2, 4, 6, 8]
                )
                grid: dict[str, Any] = {}
                for context_length in contexts:
                    generator = torch.Generator(device="cpu").manual_seed(
                        args.seed * 1000 + context_length
                    )
                    context_ids = torch.randint(
                        0,
                        vocab_size,
                        (1, context_length),
                        generator=generator,
                        dtype=torch.long,
                    ).to(device)
                    context_sha = _sha256_bytes(context_ids.cpu().numpy().tobytes())
                    progress.line(
                        "stage=grid context",
                        context=context_length,
                        candidates=len(candidates),
                    )
                    grid_cache: dict[str, Any] = {}

                    def grid_prefill(
                        ids: Any = context_ids, holder: Any = grid_cache
                    ) -> Any:
                        holder["cache"] = prefill_context(
                            scorer.backbone, ids, policy=args.policy
                        )
                        return holder["cache"]

                    _, _, grid_prefill_summary = measurer.repeat(
                        f"grid_prefill_p{context_length}",
                        grid_prefill,
                        warmups=args.scale_warmups,
                        repeats=args.scale_repeats,
                        progress=progress,
                        watchdog=watchdog,
                        result_namespace=grid_cache,
                        result_key="cache",
                    )
                    grid_entry: dict[str, Any] = {
                        "context_tokens": context_length,
                        "context_sha256": context_sha,
                        "prefill": grid_prefill_summary,
                        "candidate_from_cache_by_k": {},
                        "logits_by_k": {},
                    }
                    grid_entry["prefill"].update(
                        {"cache_length": int(grid_cache["cache"].length)}
                    )
                    by_k: dict[str, Any] = {}
                    for k in scale_ks:
                        if k > len(candidates):
                            continue
                        subset = candidates[:k]
                        _, subset_logits, summary = measurer.repeat(
                            f"grid_p{context_length}_k{k}",
                            lambda sub=subset, c=grid_cache["cache"]: (
                                score_candidates_from_cache(
                                    scorer.backbone,
                                    scorer.head,
                                    c,
                                    sub,
                                    candidate_batching=args.candidate_batching,
                                )
                            ),
                            warmups=args.scale_warmups,
                            repeats=args.scale_repeats,
                            progress=progress,
                            watchdog=watchdog,
                            result_namespace={},
                            result_key="logits",
                        )
                        by_k[str(k)] = summary
                        grid_entry["logits_by_k"][str(k)] = logits_digest(
                            subset_logits
                        )["values"]
                    grid_entry["candidate_from_cache_by_k"] = by_k
                    if args.scale_e2e:
                        _, _, grid_e2e = measurer.repeat(
                            f"grid_e2e_p{context_length}",
                            lambda ids=context_ids: score_candidates(
                                scorer.backbone,
                                scorer.head,
                                ids,
                                candidates,
                                policy=args.policy,
                                candidate_batching=args.candidate_batching,
                            ),
                            warmups=args.scale_warmups,
                            repeats=args.scale_repeats,
                            progress=progress,
                            watchdog=watchdog,
                            result_namespace={},
                            result_key="logits",
                        )
                        grid_entry["end_to_end"] = grid_e2e
                    grid[str(context_length)] = grid_entry
                    progress.line(
                        "stage=grid context done",
                        context=context_length,
                        prefill_median=grid_prefill_summary["median_s"],
                    )
                receipt["scaling"]["context_grid"] = grid
                receipt["scaling"]["protocol"] = {
                    "scale_warmups": args.scale_warmups,
                    "scale_repeats": args.scale_repeats,
                    "scale_e2e": args.scale_e2e,
                    "candidates_used": len(candidates),
                }

            # ---- optional: aten op profile --------------------------------
            if args.profile:
                from torch.profiler import ProfilerActivity, profile

                progress.line("stage=profile begin", iters=args.profile_iters)
                with profile(
                    activities=[ProfilerActivity.CPU],
                    record_shapes=False,
                    with_stack=False,
                ) as prof:
                    for _ in range(args.profile_iters):
                        score_candidates_from_cache(
                            scorer.backbone, scorer.head, cache, candidates
                        )
                key_averages = list(prof.key_averages())
                rows = sorted(
                    key_averages,
                    key=lambda entry: getattr(entry, "self_cpu_time_total", 0.0),
                    reverse=True,
                )[: args.profile_top]
                top_ops = [
                    {
                        "op": entry.key,
                        "self_cpu_ms": round(
                            getattr(entry, "self_cpu_time_total", 0.0)
                            / args.profile_iters,
                            4,
                        ),
                        "calls": int(
                            round(getattr(entry, "count", 0) / args.profile_iters)
                        ),
                    }
                    for entry in rows
                ]
                receipt["profile"] = {
                    "stage": "candidate_scoring_from_cache",
                    "activity": "CPU only (ProfilerActivity.MPS unavailable in this "
                    "torch build; MPS op timings not profiled)",
                    "iters": args.profile_iters,
                    "top_ops": top_ops,
                }
                progress.line(
                    "stage=profile done",
                    ops=len(top_ops),
                    top_op=top_ops[0]["op"] if top_ops else "n/a",
                )

            # ---- logits evidence ------------------------------------------
            digest = logits_digest(cached_logits)
            receipt["logits"] = {
                **digest,
                "source": "score_candidates_from_cache (Stage 2)",
                "tolerance_contract": {
                    "abs": DEFAULT_ATOL,
                    "rank_top1_required": True,
                    "note": (
                        "Cross-backend/parity deltas are allowed within abs 2e-5; "
                        "rank order and top-1 are decision-relevant and must stay "
                        "identical"
                    ),
                },
            }
            if args.save_logits:
                saved = save_logits_file(
                    args.save_logits,
                    logits=cached_logits,
                    e2e_logits=e2e_logits,
                    spec=spec,
                    model_id=scorer.model_id,
                    source_sha256=source_start.get("sha256"),
                    device=device,
                )
                receipt["logits"]["saved"] = saved
                progress.line("stage=logits saved", path=saved["path"])

            if args.check_baseline:
                golden_values, golden_meta = load_golden_logits(
                    args.check_baseline, key=args.baseline_key
                )
                golden_report = compare_logits(
                    digest["values"],
                    golden_values,
                    atol=DEFAULT_ATOL,
                    rank_fail=True,
                    label_a="this_run",
                    label_b=f"golden:{Path(args.check_baseline).name}",
                )
                receipt["golden_check"] = {
                    "path": golden_meta["path"],
                    "golden_sha256": golden_meta.get("sha256"),
                    "golden_input_sha256": (
                        golden_meta.get("inputs", {}).get("input_sha256")
                        if isinstance(golden_meta.get("inputs"), dict)
                        else golden_meta.get("input_sha256")
                    ),
                    "comparison": golden_report,
                    "passed": golden_report["passed"],
                }
                print(
                    format_comparison(golden_report),
                    flush=True,
                )
                progress.line(
                    "stage=golden_check",
                    passed=golden_report["passed"],
                    max_abs=golden_report["max_abs_diff"],
                )

            if args.save_golden:
                golden_payload = build_golden_payload(receipt, digest["values"])
                target = write_json(args.save_golden, golden_payload)
                receipt["logits"]["golden_saved"] = {
                    "path": target,
                    "sha256": _sha256_file(target),
                    "schema": golden_payload["schema"],
                }
                progress.line("stage=golden saved", path=target)

        # ---- provenance / memory / finish --------------------------------
        receipt["provenance"]["source_end"] = source_fingerprint()
        receipt["provenance"]["source_changed_during_run"] = receipt["provenance"][
            "source_start"
        ].get("sha256") != receipt["provenance"]["source_end"].get("sha256")
        receipt["provenance"]["unstable_loader"] = bool(
            receipt["provenance"]["source_changed_during_run"]
        )
        if receipt["provenance"]["unstable_loader"]:
            print(
                "[integrity] ERROR: wyat/loading/ouro.py changed during this run "
                f"({receipt['provenance']['source_start'].get('sha256')} -> "
                f"{receipt['provenance']['source_end'].get('sha256')}); receipt is "
                "NOT usable as a baseline",
                file=sys.stderr,
                flush=True,
            )
        receipt["memory"].update(
            {
                "peak_rss_gib": gib(peak_rss_bytes()),
                "current_rss_gib": gib(current_rss_bytes()),
                "rss_cap_hit": bool(watchdog.triggered),
                "rss_watchdog_peak_seen_gib": watchdog.peak_gib,
                "loadavg_at_end": list(os.getloadavg()),
            }
        )
        if device == "mps":
            receipt["memory"].setdefault("mps", {})["at_end"] = device_memory("mps")
        receipt["status"] = "ok"
        receipt["finished_at"] = utc_now()
        receipt["elapsed_total_s"] = round(time.perf_counter() - run_started, 3)
    except RssCapExceeded as error:
        receipt["status"] = "aborted"
        receipt["abort_reason"] = "rss_cap_exceeded"
        receipt["error"] = str(error)
        receipt["finished_at"] = utc_now()
        exit_code = EXIT_RSS_CAP
        print(f"[watchdog] {error}", file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        receipt["status"] = "interrupted"
        receipt["error"] = "KeyboardInterrupt"
        receipt["finished_at"] = utc_now()
        exit_code = 130
    except Exception:
        receipt["status"] = "error"
        receipt["error"] = traceback.format_exc()
        receipt["finished_at"] = utc_now()
        exit_code = 1
        traceback.print_exc()
    finally:
        watchdog.stop()
        if scorer is not None:
            del scorer
        gc.collect()
        if device == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:  # pragma: no cover
                pass
        if lock is not None:
            lock.release()
            print(f"[lock] released {LOCK_DIR}", flush=True)

    if receipt["status"] == "ok":
        receipt["protocol"]["warmups"] = args.warmups
        receipt["protocol"]["repeats"] = args.repeats
        if receipt["provenance"].get("unstable_loader"):
            exit_code = EXIT_UNSTABLE_LOADER
            print(
                "[integrity] mark unstable_loader=true; do not use this receipt "
                "as a baseline",
                file=sys.stderr,
                flush=True,
            )
        elif receipt.get("golden_check") and not receipt["golden_check"]["passed"]:
            exit_code = EXIT_GOLDEN_MISMATCH
            print(
                "[integrity] golden baseline mismatch: candidate logits differ "
                f"beyond {DEFAULT_ATOL} abs or rank order changed",
                file=sys.stderr,
                flush=True,
            )
        elif (
            receipt.get("self_consistency")
            and not receipt["self_consistency"]["passed"]
        ):
            exit_code = EXIT_SELF_INCONSISTENT
            print(
                "[integrity] multi-candidate self-consistency FAILED: pooled "
                "scoring disagrees with per-candidate scoring",
                file=sys.stderr,
                flush=True,
            )
    if args.json_out:
        target = write_json(args.json_out, receipt)
        print(f"[receipt] JSON receipt written to {target}", flush=True)
    if args.markdown_out:
        Path(args.markdown_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown_out).write_text(render_markdown(receipt), encoding="utf-8")
        print(f"[receipt] markdown written to {args.markdown_out}", flush=True)

    _print_summary(receipt)
    return exit_code


def _sha256_bytes(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def _print_summary(receipt: dict[str, Any]) -> None:
    print("-" * 78)
    stages = receipt.get("stages", {})
    for name in (
        "load",
        "prefill_stage1",
        "candidate_scoring_from_cache",
        "end_to_end",
    ):
        summary = stages.get(name)
        if not summary:
            continue
        if "seconds" in summary:
            print(
                f"RESULT stage={name} seconds={summary['seconds']:.3f} "
                f"peak_rss={summary.get('peak_rss_gib', 0):.2f}GiB"
            )
        else:
            print(
                f"RESULT stage={name} median={summary['median_s']:.3f}s "
                f"min={summary['min_s']:.3f}s max={summary['max_s']:.3f}s "
                f"n={summary['n']}"
            )
    scaling = receipt.get("scaling", {}).get("candidate_count", {})
    if scaling:
        print(
            "RESULT scaling K: "
            + ", ".join(
                f"K={key}:{scaling[key]['median_s']:.3f}s"
                for key in sorted(scaling, key=int)
            )
        )
    grid = receipt.get("scaling", {}).get("context_grid", {})
    if grid:
        header = "context\\K"
        k_lists = sorted(
            {
                int(k)
                for context in grid.values()
                for k in context.get("candidate_from_cache_by_k", {})
            }
        )
        rows = ["RESULT grid: " + header + " | " + " | ".join(str(k) for k in k_lists)]
        for context_length in sorted(grid, key=lambda item: int(item)):
            entry = grid[context_length]
            cells = []
            for k in k_lists:
                summary = entry.get("candidate_from_cache_by_k", {}).get(str(k))
                cells.append(f"{summary['median_s']:.3f}" if summary else "-")
            rows.append(f"RESULT grid: {context_length} | " + " | ".join(cells))
        print("\n".join(rows))
    logits = receipt.get("logits", {}).get("values")
    if logits:
        print(
            "RESULT logits sha256="
            + str(receipt["logits"].get("sha256"))[:16]
            + " values="
            + ", ".join(f"{value:+.6f}" for value in logits)
        )
    self_consistency = receipt.get("self_consistency")
    if self_consistency:
        print(
            "RESULT self_consistency "
            + ("PASS" if self_consistency["passed"] else "FAIL")
            + f" max_abs_diff={self_consistency['comparison']['max_abs_diff']!r} "
            f"seconds={self_consistency['seconds']}"
        )
    golden = receipt.get("golden_check")
    if golden:
        print(
            "RESULT golden_check "
            + ("PASS" if golden["passed"] else "FAIL")
            + f" max_abs_diff={golden['comparison']['max_abs_diff']!r} "
            f"rank_changed={golden['comparison']['rank_order_changed']} "
            f"path={golden['path']}"
        )
    provenance = receipt.get("provenance", {})
    print(
        "RESULT loader_stability "
        + ("STABLE" if not provenance.get("unstable_loader") else "UNSTABLE")
        + f" sha_start={provenance.get('source_start', {}).get('sha256')} "
        f"sha_end={provenance.get('source_end', {}).get('sha256')}"
    )
    memory = receipt.get("memory", {})
    print(
        f"RESULT memory peak_rss={memory.get('peak_rss_gib')}GiB "
        f"cap={memory.get('rss_cap_gib')}GiB hit={memory.get('rss_cap_hit')}"
    )
    if receipt.get("status") != "ok":
        print(f"RESULT status={receipt.get('status')} error={receipt.get('error')}")
    print("-" * 78)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_bench(args, parser)


if __name__ == "__main__":
    raise SystemExit(main())
