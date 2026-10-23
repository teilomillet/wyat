#!/usr/bin/env python3
"""Build the local, *non-release* Ouro profiling artifact — reproducible.

Inputs (all local, no network):

* base weights + source/tokenizer files: the pinned local
  ``ByteDance/Ouro-1.4B`` snapshot under
  ``artifacts/ouro-study-20261018/sources/core-model-snapshots-v1/Ouro-1.4B/``
  (BF16 ``model.safetensors``)
* adapters + scalar head: keme's local ``last2`` trained checkpoint
  ``.keme/ouro/experiments/pretrained-branch-adaptation-trained-v2/last2/final.pt``
  (attention-only step LoRA rank 16). Its adapter key stems (``model.layers...``)
  are prefixed with ``backbone.`` before ``write_ouro_artifact``;
  ``write_ouro_artifact`` prefixes the scalar-head keys with ``head.`` itself.

The built directory is a profiling input only — never a release artifact. The
script refuses to overwrite, records SHA-256 for every input and every output
file, records the manifest ``execution`` dict, and (by default) verifies the
build reproduces the existing scratch artifact byte-for-byte when that artifact
is present. The build also takes the exclusive BENCH_LOCK so it never overlaps a
model-loading benchmark.

Usage::

  .venv/bin/python scripts/ouro_speed/build_local_artifact.py \
      --out /tmp/ouro-speed/artifact-last2-cpu-fp32-repro \
      --receipt artifacts/ouro-speed-20261021/artifact-build-receipt.json

Exit codes: 0 built (whether or not it reproduces the reference), 2 refused
(output exists / inputs missing).
"""

from __future__ import annotations

import argparse
import json
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
    DEFAULT_LOCK_TIMEOUT_S,
    LOCK_DIR,
    BenchLock,
    LockTimeout,
    apply_nice,
    artifact_fingerprint,
    file_sha256,
    gib,
    git_state,
    peak_rss_bytes,
    sha256_tensor_mapping,
    source_fingerprint,
    write_json,
)

DEFAULT_BASE_DIR = (
    REPO_ROOT
    / "artifacts"
    / "ouro-study-20261018"
    / "sources"
    / "core-model-snapshots-v1"
    / "Ouro-1.4B"
)
DEFAULT_CKPT = Path(
    "/Users/teilomillet/Code/keme/.keme/ouro/experiments/"
    "pretrained-branch-adaptation-trained-v2/last2/final.pt"
)
DEFAULT_OUT = Path("/tmp/ouro-speed/artifact-last2-cpu-fp32-repro")
CANONICAL_SCRATCH_ARTIFACT = Path("/tmp/ouro-speed/artifact-last2-cpu-fp32")
BUILD_NOTICE = (
    "PROFILING ONLY — NOT A RELEASE ARTIFACT. Local non-release weights are "
    "assembled for timing/parity work only; do not publish or cite as a release."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the local non-release Ouro profiling artifact (profiling only)"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--base-dir", default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--attention", default="sdpa-math")
    parser.add_argument(
        "--device-policy",
        default="cpu-fp32",
        choices=["cpu-fp32", "cuda-bf16-autocast", "mps-fp32"],
    )
    parser.add_argument(
        "--compare-to",
        default=None,
        help=(
            "existing artifact whose manifest file hashes should be compared "
            "(default: auto-detect /tmp/ouro-speed/artifact-last2-cpu-fp32)"
        ),
    )
    parser.add_argument("--receipt", default=None, help="build receipt JSON output")
    parser.add_argument("--nice", type=int, default=10)
    parser.add_argument("--no-lock", action="store_true")
    parser.add_argument("--lock-timeout-s", type=float, default=DEFAULT_LOCK_TIMEOUT_S)
    return parser


def hash_input_tree(root: Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        files[path.name] = {
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
    return {"path": str(root), "files": files}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0 <= args.nice <= 20:
        parser.error("--nice must be in 0..20")
    out = Path(args.out)
    if out.exists():
        print(
            f"ERROR: refusing to overwrite existing artifact: {out}\n"
            "Remove it yourself if you really mean to rebuild (profiling only).",
            file=sys.stderr,
        )
        return 2
    base_dir = Path(args.base_dir)
    ckpt = Path(args.ckpt)
    missing = [str(path) for path in (base_dir, ckpt) if not path.exists()]
    if missing:
        print(f"ERROR: missing inputs: {missing}", file=sys.stderr)
        return 2

    print("=" * 78)
    print(BUILD_NOTICE)
    print("=" * 78)
    print(f"[build] base_dir={base_dir}")
    print(f"[build] ckpt={ckpt}")
    print(f"[build] out={out}")
    apply_nice(args.nice)

    lock = None
    lock_waited = 0.0
    if not args.no_lock:
        lock = BenchLock(LOCK_DIR, timeout_s=args.lock_timeout_s, log=print)
        try:
            lock_waited = lock.acquire()
        except LockTimeout as error:
            print(f"[build] ERROR: {error}", file=sys.stderr)
            return 2

    started = time.perf_counter()
    receipt: dict[str, Any] = {
        "schema": "ouro-speed-artifact-build/v1",
        "kind": "build_local_artifact",
        "notice": BUILD_NOTICE,
        "status": "running",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.gmtime()),
        "command": {"argv": sys.argv, "cwd": str(Path.cwd())},
        "out": str(out),
        "lock": {
            "path": str(LOCK_DIR),
            "held": lock is not None,
            "waited_s": round(lock_waited, 3),
        },
        "provenance": {"git": git_state(), "source": source_fingerprint()},
    }
    exit_code = 0
    try:
        # Heavy imports stay after validation so --help / refusal are fast.
        import torch

        from wyat.loading import ouro_artifact as contract
        from wyat.loading.ouro import write_ouro_artifact

        receipt["environment"] = {
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
        }
        print("[build] hashing inputs ...", flush=True)
        base_hashes = hash_input_tree(base_dir)
        ckpt_hashes = {
            "final.pt": {
                "sha256": file_sha256(ckpt),
                "size_bytes": ckpt.stat().st_size,
            }
        }
        receipt["inputs"] = {
            "base_dir": base_hashes,
            "checkpoint_dir": str(ckpt.parent),
            "checkpoint": ckpt_hashes,
            "params": {
                "max_length": args.max_length,
                "attention": args.attention,
                "device_policy": args.device_policy,
                "input_rendering": None,
            },
        }
        # Optional keme sidecar provenance (the receipt that produced last2/final.pt).
        sidecar = ckpt.parent / "receipt.json"
        if sidecar.exists():
            receipt["inputs"]["checkpoint_sidecar_receipt"] = {
                "path": str(sidecar),
                "sha256": file_sha256(sidecar),
            }
        print(
            f"[build] hashed {len(base_hashes['files'])} base files + checkpoint "
            f"({time.perf_counter() - started:.1f}s)",
            flush=True,
        )

        print("[build] loading keme last2 checkpoint ...", flush=True)
        checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)
        if checkpoint.get("format") != "ouro-branch-adapter-v1":
            raise SystemExit(
                f"unexpected checkpoint format: {checkpoint.get('format')!r}"
            )
        adapters = {
            (
                name if name.startswith("backbone.") else f"backbone.{name}"
            ): tensor.detach().to(torch.float32).contiguous()
            for name, tensor in checkpoint.get("adapters", {}).items()
        }
        head = {
            (name if name.startswith("head.") else f"head.{name}"): tensor.detach()
            .to(torch.float32)
            .contiguous()
            for name, tensor in checkpoint.get("head", {}).items()
        }
        if not adapters or not head:
            raise SystemExit("checkpoint is missing adapters or head tensors")
        receipt["derived_tensors"] = {
            "adapter_count": len(adapters),
            "adapter_mapping_sha256": sha256_tensor_mapping(adapters),
            "head_count": len(head),
            "head_mapping_sha256": sha256_tensor_mapping(head),
            "dtype": "float32",
            "key_rule": (
                "adapter keys prefixed 'backbone.'; head keys prefixed 'head.' "
                "(write_ouro_artifact also enforces the head prefix)"
            ),
            "src_ckpt_format": checkpoint.get("format"),
        }
        print(
            f"[build] extracted {len(adapters)} adapter + {len(head)} head FP32 "
            f"tensors (rss peak {gib(peak_rss_bytes()):.2f} GiB)",
            flush=True,
        )
        del checkpoint

        print("[build] writing artifact directory ...", flush=True)
        build_started = time.perf_counter()
        manifest = write_ouro_artifact(
            out,
            source_dir=base_dir,
            tokenizer_dir=base_dir,
            base_weights=base_dir / contract.BASE_WEIGHT_FILE,
            adapters=adapters,
            head=head,
            max_length=args.max_length,
            attention=args.attention,
            device_policy=args.device_policy,
            input_rendering=None,
            contract_prompt=contract.PROMPT,
        )
        build_seconds = time.perf_counter() - build_started
        artifact = artifact_fingerprint(out)
        receipt.update(
            {
                "model_id": manifest.get("model_id"),
                "manifest_execution": manifest.get("execution"),
                "manifest_files_sha256": manifest.get("files_sha256"),
                "manifest_sha256": artifact.get("manifest_sha256"),
                "artifact": artifact,
                "output_total_bytes": artifact.get("total_listed_bytes"),
                "build_seconds": round(build_seconds, 3),
                "peak_rss_gib": gib(peak_rss_bytes()),
            }
        )

        # ---- reproduce check against a reference artifact ------------------
        compare_to = args.compare_to
        if compare_to is None and CANONICAL_SCRATCH_ARTIFACT.exists():
            compare_to = str(CANONICAL_SCRATCH_ARTIFACT)
        if compare_to is not None:
            reference = artifact_fingerprint(compare_to)
            reference_files = reference.get("files_sha256") or {}
            produced_files = manifest.get("files_sha256") or {}
            mismatches = {
                name: {
                    "reference": reference_files.get(name),
                    "produced": produced_files.get(name),
                }
                for name in sorted(set(reference_files) | set(produced_files))
                if reference_files.get(name) != produced_files.get(name)
            }
            receipt["reproducible_build"] = {
                "compared_to": reference.get("path"),
                "compared_to_model_id": reference.get("model_id"),
                "model_id_match": reference.get("model_id") == manifest.get("model_id"),
                "execution_match": reference.get("execution")
                == manifest.get("execution"),
                "files_sha256_match": not mismatches,
                "mismatches": mismatches,
                "reference_manifest_sha256": reference.get("manifest_sha256"),
            }
            status = "BYTE-IDENTICAL" if not mismatches else "MISMATCH"
            print(
                f"[build] reproducibility vs {compare_to}: {status} "
                f"(model_id_match={receipt['reproducible_build']['model_id_match']})",
                flush=True,
            )
            if mismatches:
                for name, delta in mismatches.items():
                    print(
                        f"[build]   mismatch {name}: reference="
                        f"{delta['reference']} produced={delta['produced']}",
                        file=sys.stderr,
                    )
        receipt["status"] = "ok"
        receipt["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.gmtime())
        receipt["elapsed_total_s"] = round(time.perf_counter() - started, 3)
        print(
            f"[build] model_id={manifest.get('model_id')}\n"
            f"[build] execution={json.dumps(manifest.get('execution'))}\n"
            f"[build] artifact built at {out} in {build_seconds:.1f}s "
            f"(peak rss {gib(peak_rss_bytes()):.2f} GiB)",
            flush=True,
        )
        print(BUILD_NOTICE, flush=True)
    except SystemExit as error:
        receipt["status"] = "refused"
        receipt["error"] = str(error)
        receipt["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.gmtime())
        exit_code = 2
        print(f"ERROR: {error}", file=sys.stderr)
    except Exception:
        receipt["status"] = "error"
        receipt["error"] = traceback.format_exc()
        receipt["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.gmtime())
        exit_code = 1
        traceback.print_exc()
    finally:
        if lock is not None:
            lock.release()

    if args.receipt:
        target = write_json(args.receipt, receipt)
        print(f"[receipt] build receipt written to {target}", flush=True)
    print(
        f"RESULT build status={receipt.get('status')} "
        f"model_id={receipt.get('model_id')} out={out} "
        "reproducible="
        f"{receipt.get('reproducible_build', {}).get('files_sha256_match')}"
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
