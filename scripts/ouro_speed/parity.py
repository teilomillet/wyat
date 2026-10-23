#!/usr/bin/env python3
"""Compare Ouro candidate logits for parity evidence — profiling only.

Two modes:

1. **File mode** (default): compare two saved logit files::

     .venv/bin/python scripts/ouro_speed/parity.py \
         --a /tmp/.../logits-cpu.pt --b /tmp/.../logits-mps.pt \
         --json-out artifacts/ouro-speed-20261021/parity-cpu-vs-mps.json

   Supported inputs: ``.pt``/``.pth`` (tensor, or dict with ``logits``/``scores``
   key), ``.json`` (list or dict with ``logits``), ``.npy``.

2. **Run mode**: run the artifact/backends through ``bench.py`` twice (each run
   in its own process, sequentially, to bound memory) and compare their logits::

     .venv/bin/python scripts/ouro_speed/parity.py \
         --run-a /tmp/ouro-speed/artifact-last2-cpu-fp32 --device-a cpu \
         --run-b /tmp/ouro-speed/artifact-last2-cpu-fp32 --device-b mps \
         --json-out .../parity-cpu-vs-mps.json

   The parent process holds ``/tmp/ouro-speed/BENCH_LOCK`` for the whole
   A-then-B sequence and passes ``--no-lock`` to the children.

Declared comparison contract
----------------------------
* gating tolerance: **max abs delta <= 2e-5** (``--atol``) on candidate scores
  (float32). This is the declared cross-backend tolerance: CPU fp32 and MPS
  fp32 accumulate in different orders; keme's canonical CPU scorer also keeps
  BF16 base weights while wyat's CPU loader casts to FP32, so cross-repo logits
  are **not** expected to be bitwise identical.
* ``--fail-on-rank-change`` additionally requires identical full rank order and
  top-1 (used for the CPU-vs-MPS decision-parity target).
* relative deltas are reported for diagnostics but do not gate (scores can be
  near zero).
* exit code: 0 if the comparison passes, 1 if it violates the declared
  tolerance/rank contract, 2 on usage errors.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from _common import (  # noqa: E402
    DEFAULT_ATOL,
    DEFAULT_LOCK_TIMEOUT_S,
    DEFAULT_MPS_FRACTION,
    DEFAULT_RSS_CAP_GIB,
    DEFAULT_THREADS,
    LOCK_DIR,
    RECEIPT_SCHEMA,
    BenchLock,
    LockTimeout,
    apply_nice,
    compare_logits,
    file_sha256,
    format_comparison,
    git_state,
    prescan_env,
    source_fingerprint,
    write_json,
)  # noqa: E402

prescan_env(sys.argv)

import torch  # noqa: E402

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_USAGE = 2


# ---------------------------------------------------------------------------
# Logit file loading
# ---------------------------------------------------------------------------


def _extract_logits(payload: Any, path: Path) -> tuple[list[float], dict[str, Any]]:
    metadata: dict[str, Any] = {}
    values: Any
    if isinstance(payload, dict):
        for key in ("logits", "scores", "e2e_logits", "values"):
            if key in payload:
                values = payload[key]
                metadata = {
                    name: value
                    for name, value in payload.items()
                    if name not in {"logits", "scores", "e2e_logits", "values"}
                }
                break
        else:
            raise SystemExit(f"{path}: dict has no 'logits'/'scores' key")
    else:
        values = payload
    if isinstance(values, torch.Tensor):
        values = values.detach().float().cpu().reshape(-1).tolist()
    if not isinstance(values, (list, tuple)):
        raise SystemExit(f"{path}: unsupported logits type {type(values).__name__}")
    return [float(value) for value in values], metadata


def _dot_path(payload: Any, key: str) -> Any:
    current = payload
    for part in key.split("."):
        if isinstance(current, dict):
            if part not in current:
                raise SystemExit(f"key path {key!r}: missing key {part!r}")
            current = current[part]
        elif isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError) as error:
                raise SystemExit(f"key path {key!r}: bad index {part!r}") from error
        else:
            raise SystemExit(f"key path {key!r}: cannot descend at {part!r}")
    return current


def load_logits(
    path: str | Path, *, key: str | None = None
) -> tuple[list[float], dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        raise SystemExit(f"logit file not found: {target}")
    suffix = target.suffix.lower()
    checksum = file_sha256(target)
    if suffix in {".pt", ".pth"}:
        payload = torch.load(target, map_location="cpu", weights_only=False)
        if key is not None:
            payload = _dot_path(payload, key)
        values, metadata = _extract_logits(payload, target)
    elif suffix == ".json":
        payload = json.loads(target.read_text(encoding="utf-8"))
        if key is not None:
            payload = _dot_path(payload, key)
        values, metadata = _extract_logits(payload, target)
    elif suffix == ".npy":
        payload = torch.from_numpy(__import__("numpy").load(target))
        values = payload.detach().float().reshape(-1).tolist()
        metadata = {}
    else:
        raise SystemExit(f"unsupported logits file suffix: {target.suffix}")
    return values, {"path": str(target.resolve()), "sha256": checksum, **metadata}


# ---------------------------------------------------------------------------
# Run mode: execute bench.py twice, sequentially, under one lock
# ---------------------------------------------------------------------------


def _bench_command(
    artifact: str,
    *,
    device: str,
    logits_path: Path,
    receipt_path: Path,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        sys.executable,
        str(SCRIPT_DIR / "bench.py"),
        "--artifact",
        str(artifact),
        "--device",
        device,
        "--no-lock",
        "--context-tokens",
        str(args.context_tokens),
        "--candidates",
        str(args.candidates),
        "--candidate-length",
        str(args.candidate_length),
        "--seed",
        str(args.seed),
        "--warmups",
        "1",
        "--repeats",
        "1",
        "--threads",
        str(args.threads),
        "--nice",
        str(args.nice),
        "--rss-cap-gib",
        str(args.rss_cap_gib),
        "--mps-fraction",
        str(args.mps_fraction),
        "--json-out",
        str(receipt_path),
        "--save-logits",
        str(logits_path),
    ]
    if args.allow_device_override:
        command.append("--allow-device-override")
    if args.candidate_batching:
        command.append("--candidate-batching")
    if args.candidate_lengths:
        command.extend(["--candidate-lengths", args.candidate_lengths])
    return command


def run_two_benches(args: argparse.Namespace) -> dict[str, Any]:
    workdir = Path(tempfile.mkdtemp(prefix="ouro-parity-"))
    runs = [
        ("a", args.run_a, args.device_a),
        ("b", args.run_b, args.device_b),
    ]
    lock = None
    waited = 0.0
    if not args.no_lock:
        lock = BenchLock(LOCK_DIR, timeout_s=args.lock_timeout_s, log=print)
        try:
            waited = lock.acquire()
        except LockTimeout as error:
            raise SystemExit(f"could not acquire bench lock: {error}") from None
    results: dict[str, Any] = {
        "workdir": str(workdir),
        "lock_waited_s": waited,
        "runs": {},
    }
    try:
        for label, artifact, device in runs:
            if artifact is None:
                raise SystemExit(f"--run-{label} is required in run mode")
            logits_path = workdir / f"logits-{label}-{device}.pt"
            receipt_path = workdir / f"bench-{label}-{device}.json"
            command = _bench_command(
                artifact,
                device=device,
                logits_path=logits_path,
                receipt_path=receipt_path,
                args=args,
            )
            print(
                f"[parity] run {label}: device={device} artifact={artifact} "
                f"(sequential under exclusive lock)",
                flush=True,
            )
            started = time.perf_counter()
            completed = subprocess.run(
                command,
                cwd=str(SCRIPT_DIR.parents[1]),
                capture_output=False,
                timeout=args.run_timeout_s,
                check=False,
            )
            elapsed = time.perf_counter() - started
            if completed.returncode != 0:
                raise SystemExit(
                    f"bench.py run {label} failed with exit {completed.returncode}"
                )
            values, metadata = load_logits(logits_path)
            receipt = (
                json.loads(receipt_path.read_text(encoding="utf-8"))
                if receipt_path.exists()
                else {}
            )
            results["runs"][label] = {
                "device": device,
                "artifact": str(artifact),
                "seconds": round(elapsed, 3),
                "command": command,
                "logits": values,
                "logits_file": metadata,
                "bench_receipt": str(receipt_path),
                "bench_status": receipt.get("status"),
                "artifact_model_id": receipt.get("artifact", {}).get("model_id")
                or receipt.get("scorer", {}).get("model_id"),
                "input_sha256": receipt.get("inputs", {}).get("input_sha256"),
                "source_ouro_sha256": receipt.get("provenance", {})
                .get("source_end", {})
                .get("sha256"),
                "stages": receipt.get("stages", {}),
                "memory": receipt.get("memory", {}),
            }
    finally:
        if lock is not None:
            lock.release()
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def compare_collections(
    payload_a: dict[str, Any],
    payload_b: dict[str, Any],
    *,
    atol: float,
    rank_fail: bool,
) -> dict[str, Any]:
    """Pairwise-compare headline/corpus/real collections from two rich JSON files.

    Pairs by workload name (headline dicts) / entry ``id`` (list sections); every
    pair is gated by the same declared tolerance and optional rank gate. The
    aggregate ``passed`` is False if any pair fails or ids differ.
    """
    entries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    worst: dict[str, Any] | None = None
    sections_compared: list[str] = []

    def record(label: str, left: Sequence[float], right: Sequence[float]) -> None:
        nonlocal worst
        report = compare_logits(
            left,
            right,
            atol=atol,
            rank_fail=rank_fail,
            label_a=f"{label}:a",
            label_b=f"{label}:b",
        )
        compact = {
            "workload": label,
            "max_abs_diff": report.get("max_abs_diff"),
            "bitwise_equal": report.get("bitwise_equal"),
            "rank_order_changed": report.get("rank_order_changed"),
            "top1_changed": report.get("top1_changed"),
            "violations": report.get("violations"),
            "length_a": report.get("length_a"),
            "length_b": report.get("length_b"),
            "passed": report.get("passed"),
        }
        if not compact["passed"]:
            compact["per_candidate"] = report.get("per_candidate")
            failures.append(compact)
        if worst is None or (compact["max_abs_diff"] or 0.0) > (
            worst.get("max_abs_diff") or 0.0
        ):
            worst = compact
        entries.append(compact)

    # headline: pair dict-of-score-vectors with the best key overlap.
    dict_a = {
        key: value
        for key, value in payload_a.items()
        if isinstance(value, dict)
        and value
        and all(isinstance(item, (list, tuple)) for item in value.values())
    }
    dict_b = {
        key: value
        for key, value in payload_b.items()
        if isinstance(value, dict)
        and value
        and all(isinstance(item, (list, tuple)) for item in value.values())
    }
    best_pair: tuple[str, str] | None = None
    best_overlap = 0
    for name_a, value_a in dict_a.items():
        for name_b, value_b in dict_b.items():
            overlap = len(set(value_a) & set(value_b))
            if overlap > best_overlap:
                best_overlap = overlap
                best_pair = (name_a, name_b)
    if best_pair is not None:
        sections_compared.append(f"{best_pair[0]}<->{best_pair[1]}")
        names = sorted(set(dict_a[best_pair[0]]) & set(dict_b[best_pair[1]]))
        for name in names:
            record(
                f"{best_pair[0]}/{name}",
                dict_a[best_pair[0]][name],
                dict_b[best_pair[1]][name],
            )

    # list sections (corpus / real): pair by id.
    for section in ("corpus", "real"):
        a_items = payload_a.get(section)
        b_items = payload_b.get(section)
        if not isinstance(a_items, list) or not isinstance(b_items, list):
            continue
        sections_compared.append(section)
        by_id_a = {
            str(item.get("id")): item for item in a_items if isinstance(item, dict)
        }
        by_id_b = {
            str(item.get("id")): item for item in b_items if isinstance(item, dict)
        }
        common_ids = sorted(set(by_id_a) & set(by_id_b))
        if set(by_id_a) != set(by_id_b):
            missing = sorted(set(by_id_a) ^ set(by_id_b))
            failures.append(
                {
                    "workload": f"{section}:id-mismatch",
                    "passed": False,
                    "error": f"{len(missing)} ids only in one file: {missing[:8]}",
                }
            )
        for entry_id in common_ids:
            left = by_id_a[entry_id].get("logits") or by_id_a[entry_id].get("scores")
            right = by_id_b[entry_id].get("logits") or by_id_b[entry_id].get("scores")
            if left is None or right is None:
                failures.append(
                    {
                        "workload": f"{section}/{entry_id}",
                        "passed": False,
                        "error": "missing logits field",
                    }
                )
                continue
            record(f"{section}/{entry_id}", left, right)

    passed = not failures
    return {
        "sections_compared": sections_compared,
        "entries_total": len(entries),
        "entries_passed": sum(1 for entry in entries if entry["passed"]),
        "failures": failures,
        "worst": worst,
        "passed": bool(passed),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Ouro logits with a declared abs tolerance (profiling only)"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--a", default=None, help="logits file A (file mode)")
    parser.add_argument("--b", default=None, help="logits file B (file mode)")
    parser.add_argument(
        "--a-key",
        default=None,
        help="dot path into JSON file A (e.g. headline_logits.p400-k6-s16)",
    )
    parser.add_argument(
        "--b-key", default=None, help="dot path into JSON file B (e.g. headline.mps...)"
    )
    parser.add_argument(
        "--compare-collections",
        action="store_true",
        help=(
            "pairwise-compare headline/corpus/real collections from two rich "
            "JSON files (e.g. mps-probe reference vs MPS logits); exit nonzero "
            "if any pair violates tolerance/rank"
        ),
    )
    parser.add_argument("--run-a", default=None, help="artifact A (run mode)")
    parser.add_argument("--run-b", default=None, help="artifact B (run mode)")
    parser.add_argument("--device-a", default="cpu", choices=["cpu", "mps"])
    parser.add_argument("--device-b", default="mps", choices=["cpu", "mps"])
    parser.add_argument("--allow-device-override", action="store_true")
    parser.add_argument("--candidate-batching", action="store_true")
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument(
        "--fail-on-rank-change",
        action="store_true",
        help="also fail when full rank order or top-1 differs",
    )
    parser.add_argument("--label-a", default=None, help="label for side A in reports")
    parser.add_argument("--label-b", default=None, help="label for side B in reports")
    parser.add_argument("--json-out", default=None)
    # run-mode workload / caps (bench.py flags)
    parser.add_argument("--context-tokens", type=int, default=400)
    parser.add_argument("--candidates", type=int, default=6)
    parser.add_argument("--candidate-length", type=int, default=16)
    parser.add_argument("--candidate-lengths", default=None)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--nice", type=int, default=10)
    parser.add_argument("--rss-cap-gib", type=float, default=DEFAULT_RSS_CAP_GIB)
    parser.add_argument("--mps-fraction", type=float, default=DEFAULT_MPS_FRACTION)
    parser.add_argument("--no-lock", action="store_true")
    parser.add_argument("--lock-timeout-s", type=float, default=DEFAULT_LOCK_TIMEOUT_S)
    parser.add_argument("--run-timeout-s", type=float, default=1800.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.compare_collections:
        mode = "collections"
        if not (args.a and args.b):
            parser.error("--compare-collections requires --a and --b JSON files")
        payload_a = json.loads(Path(args.a).read_text(encoding="utf-8"))
        payload_b = json.loads(Path(args.b).read_text(encoding="utf-8"))
        collection = compare_collections(
            payload_a,
            payload_b,
            atol=args.atol,
            rank_fail=args.fail_on_rank_change,
        )
        report = {
            "schema": RECEIPT_SCHEMA,
            "kind": "parity-collection",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.gmtime()),
            "declared_contract": {
                "gating": "max_abs_diff <= atol per pair",
                "atol": args.atol,
                "fail_on_rank_change": bool(args.fail_on_rank_change),
                "rel_diff_gates": False,
            },
            "a": {"path": str(Path(args.a).resolve()), "sha256": file_sha256(args.a)},
            "b": {"path": str(Path(args.b).resolve()), "sha256": file_sha256(args.b)},
            "collection": collection,
            "provenance": {
                "git": git_state(),
                "source": source_fingerprint(),
                "argv": sys.argv,
            },
            "passed": bool(collection["passed"]),
        }
        if args.json_out:
            target = write_json(args.json_out, report)
            print(f"[receipt] parity-collection report written to {target}", flush=True)
        print(
            f"parity collections {'PASS' if collection['passed'] else 'FAIL'}: "
            f"{collection['entries_passed']}/{collection['entries_total']} pairs ok, "
            f"failures={len(collection['failures'])}, "
            f"worst={collection.get('worst', {}).get('workload')} "
            f"max_abs_diff={collection.get('worst', {}).get('max_abs_diff')!r} "
            f"atol={args.atol}",
            flush=True,
        )
        return EXIT_PASS if collection["passed"] else EXIT_FAIL
    if args.run_a or args.run_b:
        mode = "run"
        if not (args.run_a and args.run_b):
            parser.error("run mode requires both --run-a and --run-b")
        if args.a or args.b:
            parser.error("use either --a/--b (file mode) or --run-a/--run-b")
        apply_nice(args.nice)
        try:
            runs = run_two_benches(args)
        except SystemExit as error:
            print(f"[parity] ERROR: {error}", file=sys.stderr, flush=True)
            return EXIT_USAGE
        values_a = runs["runs"]["a"]["logits"]
        values_b = runs["runs"]["b"]["logits"]
        label_a = (
            args.label_a
            or f"{runs['runs']['a']['device']}:{Path(str(args.run_a)).name}"
        )
        label_b = (
            args.label_b
            or f"{runs['runs']['b']['device']}:{Path(str(args.run_b)).name}"
        )
        context: dict[str, Any] = {"mode": mode, "runs": runs["runs"]}
    else:
        mode = "file"
        if not (args.a and args.b):
            parser.error("provide --a and --b (or --run-a/--run-b)")
        labels = (args.label_a, args.label_b)
        loaded_a = load_logits(args.a, key=args.a_key)
        loaded_b = load_logits(args.b, key=args.b_key)
        values_a, meta_a = loaded_a
        values_b, meta_b = loaded_b
        label_a = labels[0] or str(Path(args.a).name)
        label_b = labels[1] or str(Path(args.b).name)
        context = {"mode": mode, "a": meta_a, "b": meta_b}

    comparison = compare_logits(
        values_a,
        values_b,
        atol=args.atol,
        rank_fail=args.fail_on_rank_change,
        label_a=label_a,
        label_b=label_b,
    )
    report: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "kind": "parity",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.gmtime()),
        "declared_contract": {
            "gating": "max_abs_diff <= atol",
            "atol": args.atol,
            "fail_on_rank_change": bool(args.fail_on_rank_change),
            "rel_diff_gates": False,
            "note": (
                "cross-backend/cross-repo logits are not expected to be bitwise "
                "identical (different accumulation order; keme CPU keeps BF16 base "
                "weights while wyat CPU loader casts to FP32)"
            ),
        },
        "context": context,
        "comparison": comparison,
        "provenance": {
            "git": git_state(),
            "source": source_fingerprint(),
            "argv": sys.argv,
        },
        "passed": bool(comparison.get("passed")),
    }
    print(format_comparison(comparison), flush=True)
    if context.get("mode") == "run":
        runs = context["runs"]
        if runs["a"].get("input_sha256") != runs["b"].get("input_sha256"):
            print(
                "[parity] WARNING: input_sha256 differs between runs "
                f"({runs['a'].get('input_sha256')} vs {runs['b'].get('input_sha256')})",
                file=sys.stderr,
                flush=True,
            )
            report["input_sha_mismatch"] = True
        if runs["a"].get("source_ouro_sha256") != runs["b"].get("source_ouro_sha256"):
            print(
                "[parity] WARNING: wyat/loading/ouro.py changed between runs "
                "(bench run A vs B used different revisions)",
                file=sys.stderr,
                flush=True,
            )
            report["source_changed_between_runs"] = True
    if args.json_out:
        target = write_json(args.json_out, report)
        print(f"[receipt] parity report written to {target}", flush=True)
    print(
        f"RESULT parity {'PASS' if report['passed'] else 'FAIL'} "
        f"max_abs_diff={comparison.get('max_abs_diff')!r} "
        f"bitwise_equal={comparison.get('bitwise_equal')} "
        f"rank_order_changed={comparison.get('rank_order_changed')} "
        f"top1_changed={comparison.get('top1_changed')} "
        f"atol={args.atol}",
        flush=True,
    )
    return EXIT_PASS if report["passed"] else EXIT_FAIL


if __name__ == "__main__":
    raise SystemExit(main())
