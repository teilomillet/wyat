"""Shared protocol for Ouro-1.4B decision scoring on Apple GPU (MPS).

PROFILING ONLY: synthetic token tensors and a local non-release artifact. These
numbers are not product benchmarks and must not be cited as release results.

Mandatory resource protocol (mirrors ``scripts/ouro_speed/_common.py`` and the
Lead's original MPS probe):

1. ``PYTORCH_MPS_LOW_WATERMARK_RATIO=1.0`` and
   ``PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.2`` are exported *before* ``import torch``.
   (Default LOW=1.4/HIGH=1.7 is unsafe on a 24 GiB unified-memory Mac; setting
   HIGH=1.0 alone trips ``invalid low watermark ratio 1.4``.)
2. ``PYTORCH_ENABLE_MPS_FALLBACK`` is removed from the environment: an op that
   MPS cannot run must fail, never silently fall back to CPU inside a timing.
3. ``torch.mps.set_per_process_memory_fraction(0.6)`` caps the MPS allocator at
   0.6 of the recommended working set (17.76 GiB on this host -> 10.66 GiB).
4. ``os.nice(10)`` and ``torch.set_num_threads(6)``; one heavy process at a time,
   guarded by the exclusive lock ``/tmp/ouro-speed/BENCH_LOCK`` (``mkdir`` + 10 s
   retry, removed on exit).
5. A resident-set watchdog hard-aborts (``os._exit(42)``) above 14 GiB RSS.
6. Every timed stage is bracketed by ``torch.mps.synchronize()`` so wall time is
   real device time, not enqueue time.

Artifact policy (Lead decision 2026-10-21): the canonical MPS arm is
``/tmp/ouro-speed/artifact-last2-mps-fp32`` (manifest ``device_policy="mps-fp32"``,
FP32 materialisation on Apple GPU). ``/tmp/ouro-speed/artifact-last2-cpu-fp32``
is used only to compute CPU FP32 reference logits. Both artifacts ship identical
``model.safetensors`` / ``head.safetensors`` / ``adapters.safetensors`` bytes.

Every receipt records: artifact fingerprints (manifest, tensor-file hashes,
declared base checkpoint hash), loader fingerprint (``wyat/loading/ouro.py`` and
``wyat/loading/ouro_artifact.py`` SHA-256 + git HEAD, because other teammates are
editing the loader concurrently), host/torch/MPS environment and caps.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import platform
import resource
import shutil
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Environment: must be in place before torch initialises the MPS allocator.
# ---------------------------------------------------------------------------

os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "1.0"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "1.2"
os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)

import torch  # noqa: E402  (must follow the watermark exports above)

GIB = float(1024**3)
MIB = float(1024**2)

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRATCH = Path("/tmp/ouro-speed")
LOCK_DIR = SCRATCH / "BENCH_LOCK"
MPS_ARTIFACT = SCRATCH / "artifact-last2-mps-fp32"
CPU_ARTIFACT = SCRATCH / "artifact-last2-cpu-fp32"
KEME_DEV_TOKENIZED = Path(
    "/Users/teilomillet/Code/keme/.keme/ouro/experiments/"
    "pretrained-arc-adaptation-frozen/dev-tokenized.json"
)
RECEIPT_DIR = REPO_ROOT / "artifacts" / "ouro-speed-20261021" / "mps"
LOGIT_DIR = RECEIPT_DIR / "logits"
INPUT_DIR = RECEIPT_DIR / "inputs"

VOCAB = 49152
DEFAULT_THREADS = 6
MAX_THREADS = 6
DEFAULT_MPS_FRACTION = 0.6
MAX_MPS_FRACTION = 0.6
DEFAULT_RSS_CAP_GIB = 14.0
LOCK_RETRY_S = 10.0
DEFAULT_LOCK_TIMEOUT_S = 1800.0
EXIT_RSS_CAP = 42
DECLARED_ATOL = 2e-5
RECEIPT_SCHEMA = "ouro-speed-mps-receipt/v1"
INPUT_SCHEMA = "ouro-speed-mps-question-set/v1"

PROFILING_NOTICE = (
    "PROFILING ONLY - NOT A RELEASE ARTIFACT. Synthetic token tensors, local "
    "non-release weights; do not cite these numbers as product benchmarks."
)


# ---------------------------------------------------------------------------
# Process setup: nice, threads, MPS fraction, lock, watchdog
# ---------------------------------------------------------------------------


def resident_gib() -> float | None:
    """Current resident set size of this process in GiB (macOS/Linux via ps)."""
    try:
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        return int(out.stdout.strip()) / 1024 / 1024
    except Exception:  # pragma: no cover - probe only
        return None


def peak_rss_gib() -> float:
    """Peak resident set size of this process in GiB (ru_maxrss bytes on macOS)."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return round(value / GIB, 3)
    return round(value / 1024 / 1024, 3)  # Linux reports KiB


def mps_memory() -> dict[str, float]:
    """Allocator and driver memory for the MPS device, in GiB."""
    if not torch.backends.mps.is_available():
        return {}
    try:
        current = torch.mps.current_allocated_memory()
        driver = torch.mps.driver_allocated_memory()
        recommended = torch.mps.recommended_max_memory()
        return {
            "mps_current_gib": round(current / GIB, 3),
            "mps_driver_gib": round(driver / GIB, 3),
            "mps_recommended_max_gib": round(recommended / GIB, 3),
            "mps_fraction_cap_gib": round(DEFAULT_MPS_FRACTION * recommended / GIB, 3),
            "rss_gib": resident_gib(),
        }
    except Exception as error:  # pragma: no cover - probe only
        return {"mps_error": f"{type(error).__name__}: {error}"}


def mps_sync() -> None:
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def empty_mps_cache() -> None:
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()


class RssWatchdog(threading.Thread):
    """Abort the process if RSS or MPS driver memory exceeds the hard cap."""

    def __init__(
        self,
        cap_gib: float = DEFAULT_RSS_CAP_GIB,
        interval_s: float = 0.5,
        log_path: Path | None = None,
        sample_mps: bool = True,
    ) -> None:
        super().__init__(daemon=True, name="rss-watchdog")
        self.cap_gib = cap_gib
        self.interval_s = interval_s
        self.log_path = log_path
        self.sample_mps = sample_mps
        self._stop = threading.Event()
        self.samples: list[dict[str, Any]] = []
        self.triggered = False

    def run(self) -> None:  # pragma: no cover - thread body
        while not self._stop.wait(self.interval_s):
            rss = resident_gib()
            sample: dict[str, Any] = {"t": round(time.time(), 3), "rss_gib": rss}
            if self.sample_mps:
                mem = mps_memory()
                if "mps_driver_gib" in mem:
                    sample["driver_gib"] = mem["mps_driver_gib"]
            self.samples.append(sample)
            if self.log_path is not None:
                try:
                    with self.log_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(sample) + "\n")
                except OSError:
                    pass
            rss_over = rss is not None and rss > self.cap_gib
            driver_over = sample.get("driver_gib", 0.0) > self.cap_gib
            if rss_over or driver_over:
                self.triggered = True
                print(
                    f"WATCHDOG ABORT: rss={rss:.2f} GiB driver="
                    f"{sample.get('driver_gib')} GiB cap={self.cap_gib:.2f} GiB",
                    file=sys.stderr,
                    flush=True,
                )
                os._exit(EXIT_RSS_CAP)

    def stop(self) -> None:
        self._stop.set()


_WATCHDOG: RssWatchdog | None = None
_LOCK_HELD = False


@atexit.register
def _cleanup() -> None:
    if _WATCHDOG is not None:
        _WATCHDOG.stop()
    _release_lock()


# ---------------------------------------------------------------------------
# Exclusive bench lock
# ---------------------------------------------------------------------------


def _lock_owner_alive() -> tuple[bool, str]:
    """Best-effort stale-lock probe: is the recorded owner PID still alive?"""
    owner_path = LOCK_DIR / "owner.json"
    try:
        owner = json.loads(owner_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False, "no readable owner.json"
    pid = owner.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return False, "invalid owner pid"
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False, f"owner pid {pid} is gone"
    except PermissionError:
        return True, "owner pid exists (permission denied probe)"
    return True, f"owner pid {pid} alive"


def acquire_lock(
    timeout_s: float = DEFAULT_LOCK_TIMEOUT_S,
    retry_s: float = LOCK_RETRY_S,
    tag: str = "mps-probe",
    stale_grace_s: float = 60.0,
) -> dict[str, Any]:
    """Take the exclusive bench lock; wait (10 s retry) for other heavy runs.

    A lock directory whose recorded owner PID is gone (or which has no readable
    ``owner.json`` and is older than ``stale_grace_s``) is treated as stale and
    removed, so a crashed/killed teammate process cannot wedge the queue.
    """
    global _LOCK_HELD
    SCRATCH.mkdir(parents=True, exist_ok=True)
    started = time.time()
    while True:
        try:
            os.mkdir(LOCK_DIR)
            break
        except FileExistsError:
            waited = time.time() - started
            alive, reason = _lock_owner_alive()
            if not alive:
                try:
                    age = time.time() - LOCK_DIR.stat().st_mtime
                except OSError:
                    age = stale_grace_s
                if age > stale_grace_s:
                    print(
                        f"[lock] removing stale lock ({reason}, age {age:.0f}s)",
                        flush=True,
                    )
                    _remove_stale_lock()
                    continue
            if waited > timeout_s:
                owner = ""
                try:
                    owner = (LOCK_DIR / "owner.json").read_text(encoding="utf-8")
                except OSError:
                    pass
                raise SystemExit(
                    f"bench lock is busy after {waited:.0f}s (owner={owner!r})"
                )
            print(
                f"[lock] busy; retry in {retry_s:.0f}s (waited {waited:.0f}s)",
                flush=True,
            )
            time.sleep(retry_s)
    owner = {
        "pid": os.getpid(),
        "tag": tag,
        "argv": sys.argv,
        "started_at": now_iso(),
        "host": platform.node(),
    }
    (LOCK_DIR / "owner.json").write_text(json.dumps(owner, indent=2), encoding="utf-8")
    _LOCK_HELD = True
    print(f"[lock] acquired {LOCK_DIR} tag={tag} pid={os.getpid()}", flush=True)
    return owner


def _remove_stale_lock() -> None:
    """Remove a lock directory only after re-checking that its owner is gone."""
    alive, _reason = _lock_owner_alive()
    if alive:
        return
    try:
        owner_path = LOCK_DIR / "owner.json"
        if owner_path.is_file():
            owner_path.unlink()
        LOCK_DIR.rmdir()
    except OSError:
        pass


def _release_lock() -> None:
    global _LOCK_HELD
    if not _LOCK_HELD:
        return
    _LOCK_HELD = False
    try:
        shutil.rmtree(LOCK_DIR)
        print(f"[lock] released {LOCK_DIR}", flush=True)
    except OSError:
        pass


@contextmanager
def bench_lock(tag: str = "mps-probe", timeout_s: float = DEFAULT_LOCK_TIMEOUT_S):
    """Context manager form of :func:`acquire_lock`."""
    acquire_lock(timeout_s=timeout_s, tag=tag)
    try:
        yield
    finally:
        _release_lock()


def configure(
    device: str = "mps",
    fraction: float = DEFAULT_MPS_FRACTION,
    threads: int = DEFAULT_THREADS,
    rss_cap_gib: float = DEFAULT_RSS_CAP_GIB,
    log_path: Path | None = None,
) -> dict[str, Any]:
    """Apply nice/threads/MPS-fraction/watchdog caps and return the caps receipt."""
    try:
        niceness = os.nice(10)
    except OSError:  # pragma: no cover - platform probe only
        niceness = None
    if type(threads) is not int or not 1 <= threads <= MAX_THREADS:
        raise ValueError(f"threads must be an int in [1, {MAX_THREADS}]")
    torch.set_num_threads(threads)
    if type(fraction) is not float and type(fraction) is not int:
        raise ValueError("mps fraction must be numeric")
    if not 0 < float(fraction) <= MAX_MPS_FRACTION:
        raise ValueError(f"mps fraction must be in (0, {MAX_MPS_FRACTION}]")
    if device == "mps":
        if not torch.backends.mps.is_available():
            raise SystemExit("MPS is unavailable on this host")
        torch.mps.set_per_process_memory_fraction(float(fraction))
        mps_sync()
    global _WATCHDOG
    _WATCHDOG = RssWatchdog(
        cap_gib=rss_cap_gib, log_path=log_path, sample_mps=(device == "mps")
    )
    _WATCHDOG.start()
    caps = {
        "device": device,
        "niceness": niceness,
        "torch_num_threads": torch.get_num_threads(),
        "mps_fraction": float(fraction),
        "PYTORCH_MPS_LOW_WATERMARK_RATIO": os.environ.get(
            "PYTORCH_MPS_LOW_WATERMARK_RATIO"
        ),
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO": os.environ.get(
            "PYTORCH_MPS_HIGH_WATERMARK_RATIO"
        ),
        "PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
        "rss_watchdog_cap_gib": rss_cap_gib,
        "declared_atol": DECLARED_ATOL,
        "lock_dir": str(LOCK_DIR),
    }
    if device == "mps":
        caps.update(mps_memory())
    print(f"[caps] {json.dumps(caps, sort_keys=True)}", flush=True)
    return caps


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def synchronize(device: str = "mps") -> None:
    if device == "mps":
        mps_sync()
    elif device == "cuda":  # pragma: no cover - not used locally
        torch.cuda.synchronize()


def timed(
    fn: Callable[[], Any],
    *,
    repeats: int = 5,
    warmup: int = 1,
    device: str = "mps",
    label: str = "",
) -> dict[str, Any]:
    """Wall-clock stage timing with device synchronisation around each rep."""
    for _ in range(warmup):
        fn()
    synchronize(device)
    samples: list[float] = []
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        fn()
        synchronize(device)
        samples.append(time.perf_counter() - start)
    ordered = sorted(samples)
    result = {
        "label": label,
        "median_s": round(statistics.median(samples), 4),
        "min_s": round(ordered[0], 4),
        "max_s": round(ordered[-1], 4),
        "mean_s": round(statistics.fmean(samples), 4),
        "samples_s": [round(value, 4) for value in samples],
        "repeats": repeats,
        "warmup": warmup,
        "device": device,
    }
    if label:
        print(f"[timed] {label}: {result['median_s']:.4f}s median", flush=True)
    return result


# ---------------------------------------------------------------------------
# Question sets (deterministic synthetic tokens + real keme dev rows)
# ---------------------------------------------------------------------------


def make_question(
    seed: int, context_length: int, candidate_lengths: Sequence[int]
) -> dict[str, Any]:
    """Deterministic synthetic question: one context, K continuations."""
    generator = torch.Generator().manual_seed(int(seed))
    context = torch.randint(0, VOCAB, (1, int(context_length)), generator=generator)
    candidates = [
        torch.randint(0, VOCAB, (1, int(length)), generator=generator)
        for length in candidate_lengths
    ]
    return {
        "id": f"synthetic-seed{int(seed)}-p{int(context_length)}"
        f"-k{len(candidate_lengths)}",
        "seed": int(seed),
        "context": [int(value) for value in context.flatten().tolist()],
        "candidates": [
            [int(value) for value in candidate.flatten().tolist()]
            for candidate in candidates
        ],
    }


def headline_questions() -> dict[str, dict[str, Any]]:
    """Headline shapes from the Lead's probe plus the extended pair."""
    return {
        # Seed/order reproduces the Lead's /tmp/ouro-speed baseline inputs.
        "p400-k6-s16": make_question(20261021, 400, [16] * 6),
        "p832-k10-s64": make_question(20261023, 832, [64] * 10),
        "p512-k9-hetero-8to24": make_question(
            20261024, 512, [8, 10, 12, 14, 16, 18, 20, 22, 24]
        ),
    }


def synthetic_corpus(seed: int = 20261022, count: int = 32) -> list[dict[str, Any]]:
    """32 synthetic questions with varied P, K and S (incl. mixed lengths)."""
    context_lengths = [256, 400, 512, 832]
    candidate_counts = [2, 3, 4, 6, 8, 10]
    uniform_lengths = [8, 16, 24, 32, 48, 64]
    questions: list[dict[str, Any]] = []
    for index in range(count):
        context_length = context_lengths[index % 4]
        count_k = candidate_counts[index % 6]
        if index % 3 == 0:
            lengths = [8 + ((index * 7 + item * 5) % 17) for item in range(count_k)]
        else:
            lengths = [uniform_lengths[(index // 3 + index) % 6]] * count_k
        question = make_question(seed + index, context_length, lengths)
        question["id"] = (
            f"corpus-{index:02d}-p{context_length}-k{count_k}"
            f"-smin{min(lengths)}-smax{max(lengths)}"
        )
        questions.append(question)
    return questions


def keme_dev_rows() -> list[dict[str, Any]] | None:
    """32 real ARC dev decisions in the frozen legacy prefix/branch schema.

    Mapping (read-only; network-free): shared context = ``prefix_ids`` and one
    candidate continuation per option = ``branches[i]["input_ids"]``; the
    decision is ``argmax`` over the per-candidate scalar logits. Returns None if
    the keme protocol directory is not reachable.
    """
    if not KEME_DEV_TOKENIZED.is_file():
        return None
    rows = json.loads(KEME_DEV_TOKENIZED.read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "id": row["id"],
                "suite": row["suite"],
                "gold_index": int(row["gold_index"]),
                "context": [int(value) for value in row["prefix_ids"]],
                "candidates": [
                    [int(value) for value in branch["input_ids"]]
                    for branch in row["branches"]
                ],
            }
        )
    return out


def question_set() -> dict[str, Any]:
    """Full question set used by every probe (headline + corpus + real rows)."""
    return {
        "schema": INPUT_SCHEMA,
        "notice": PROFILING_NOTICE,
        "headline": headline_questions(),
        "corpus": synthetic_corpus(),
        "real": keme_dev_rows(),
    }


def question_set_path() -> Path:
    return INPUT_DIR / "question-set.json"


def write_question_set(data: dict[str, Any]) -> Path:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = question_set_path()
    payload = dict(data)
    payload["sha256"] = canonical_sha256(
        {key: value for key, value in data.items() if key != "sha256"}
    )
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    return path


def load_question_set() -> dict[str, Any]:
    path = question_set_path()
    if not path.is_file():
        raise SystemExit(
            f"missing question set {path}; run ref_cpu.py first (it writes the "
            "frozen inputs and the CPU FP32 reference logits)"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def reference_path() -> Path:
    return RECEIPT_DIR / "reference-cpu-fp32.json"


def load_reference() -> dict[str, Any]:
    path = reference_path()
    if not path.is_file():
        raise SystemExit(f"missing CPU FP32 reference {path}; run ref_cpu.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def to_device(
    question: dict[str, Any], device: str
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Materialise one question's context and candidates on ``device``."""
    context = torch.tensor(
        question["context"], dtype=torch.long, device=device
    ).unsqueeze(0)
    candidates = [
        torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        for ids in question["candidates"]
    ]
    return context, candidates


def run_question(
    backbone: Any,
    head: Any,
    question: dict[str, Any],
    device: str,
    *,
    context_steps: int = 4,
    candidate_steps: int = 2,
    candidate_batching: bool = False,
) -> torch.Tensor:
    """Serially score one question with the production loader path."""
    from wyat.loading.ouro import prefill_context, score_candidates_from_cache

    context, candidates = to_device(question, device)
    cache = prefill_context(
        backbone, context, policy="last", context_steps=context_steps
    )
    return score_candidates_from_cache(
        backbone,
        head,
        cache,
        candidates,
        candidate_steps=candidate_steps,
        candidate_batching=candidate_batching,
    )


def logits_list(scores: torch.Tensor) -> list[float]:
    flat = scores.detach().float().cpu().reshape(-1).tolist()
    return [float(value) for value in flat]


# ---------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------


def compare_logits(
    candidate: Sequence[float], reference: Sequence[float]
) -> dict[str, Any]:
    """Rank/top-1 preserving drift between two scalar-logit vectors."""
    left = torch.tensor(list(candidate), dtype=torch.float64)
    right = torch.tensor(list(reference), dtype=torch.float64)
    if left.shape != right.shape:
        raise ValueError("logit vectors have different shapes")
    difference = (left - right).abs()
    rank_left = torch.argsort(left, descending=True).tolist()
    rank_right = torch.argsort(right, descending=True).tolist()
    return {
        "max_abs_diff": float(difference.max()) if len(difference) else 0.0,
        "mean_abs_diff": float(difference.mean()) if len(difference) else 0.0,
        "rank_equal": bool(rank_left == rank_right),
        "top1_equal": bool(rank_left[0] == rank_right[0]) if len(rank_left) else True,
        "argmax": int(rank_left[0]) if len(rank_left) else None,
    }


def parity_summary(
    per_question: Sequence[dict[str, Any]],
    *,
    atol: float = DECLARED_ATOL,
) -> dict[str, Any]:
    """Aggregate per-question comparisons into one declared-tolerance verdict."""
    if not per_question:
        return {"questions": 0}
    max_abs = max(item["max_abs_diff"] for item in per_question)
    mean_abs = statistics.fmean(item["mean_abs_diff"] for item in per_question)
    rank_equal = sum(1 for item in per_question if item["rank_equal"])
    top1_equal = sum(1 for item in per_question if item["top1_equal"])
    return {
        "questions": len(per_question),
        "global_max_abs_diff": max_abs,
        "mean_per_question_max_abs_diff": round(mean_abs, 9),
        "rank_equal_questions": rank_equal,
        "top1_equal_questions": top1_equal,
        "atol": atol,
        "within_atol": bool(max_abs <= atol),
        "rank_stable": bool(rank_equal == len(per_question)),
        "top1_stable": bool(top1_equal == len(per_question)),
    }


def decision_flips(
    candidate_logits: Sequence[Sequence[float]],
    reference_logits: Sequence[Sequence[float]],
    gold_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Count argmax decision changes against a reference arm (optionally vs gold)."""
    flips = 0
    details = []
    correct_candidate = 0
    correct_reference = 0
    for index, (left, right) in enumerate(
        zip(candidate_logits, reference_logits, strict=False)
    ):
        arg_left = max(range(len(left)), key=lambda item: left[item])
        arg_right = max(range(len(right)), key=lambda item: right[item])
        flipped = arg_left != arg_right
        flips += int(flipped)
        if gold_indices is not None:
            correct_candidate += int(arg_left == gold_indices[index])
            correct_reference += int(arg_right == gold_indices[index])
        if flipped:
            details.append({"index": index, "arm": arg_left, "reference": arg_right})
    result: dict[str, Any] = {
        "rows": len(candidate_logits),
        "decision_flips": flips,
        "flip_details": details,
    }
    if gold_indices is not None:
        result["arm_correct"] = correct_candidate
        result["reference_correct"] = correct_reference
    return result


# ---------------------------------------------------------------------------
# Fingerprints and receipts
# ---------------------------------------------------------------------------


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def sha256_file(path: Path, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def host_fingerprint() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "node": platform.node(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "torch_mps_built": bool(torch.backends.mps.is_built()),
        "torch_mps_available": bool(torch.backends.mps.is_available()),
        "unified_memory_gib": None,
    }
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            info["unified_memory_gib"] = round(int(out.stdout.strip()) / GIB, 2)
            out = subprocess.run(
                ["sysctl", "-n", "hw.perflevel0.physicalcpu"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            info["performance_cores"] = int(out.stdout.strip())
        except Exception:  # pragma: no cover - platform probe only
            pass
    return info


_ARTIFACT_CACHE: dict[str, dict[str, Any]] = {}

_ARTIFACT_SMALL_FILES = (
    "ouro_manifest.json",
    "config.json",
    "head.safetensors",
    "adapters.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "modeling_ouro.py",
    "configuration_ouro.py",
)


def artifact_fingerprint(path: Path) -> dict[str, Any]:
    """Manifest/model_id/device_policy plus hashes of every small artifact file.

    ``model.safetensors`` (2.9 GB) is not re-hashed on every probe; the manifest's
    declared ``base.checkpoint_sha256`` plus file size/mtime are recorded instead
    (see ``hash_base`` for an explicit full re-hash).
    """
    key = str(path)
    cached = _ARTIFACT_CACHE.get(key)
    if cached is not None:
        return cached
    manifest_path = path / "ouro_manifest.json"
    if not manifest_path.is_file():
        return {"path": key, "error": "artifact directory or manifest missing"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = {
        name: sha256_file(path / name)
        for name in _ARTIFACT_SMALL_FILES
        if (path / name).is_file()
    }
    base = path / "model.safetensors"
    info = {
        "path": key,
        "model_id": manifest.get("model_id"),
        "device_policy": manifest.get("execution", {}).get("device_policy"),
        "attention": manifest.get("execution", {}).get("attention"),
        "execution": manifest.get("execution"),
        "manifest_sha256": files.get("ouro_manifest.json"),
        "files_sha256": files,
        "base_checkpoint_sha256_declared": manifest.get("base", {}).get(
            "checkpoint_sha256"
        ),
        "base_file_bytes": base.stat().st_size if base.is_file() else None,
        "base_file_mtime": base.stat().st_mtime if base.is_file() else None,
    }
    _ARTIFACT_CACHE[key] = info
    return info


def loader_fingerprint() -> dict[str, Any]:
    """Fingerprint the concurrently-edited loader modules and git state."""
    files = ["wyat/loading/ouro.py", "wyat/loading/ouro_artifact.py"]
    hashes: dict[str, str] = {}
    for relative in files:
        path = REPO_ROOT / relative
        if path.is_file():
            hashes[relative] = sha256_file(path)
    git = {"head": None, "dirty_paths": None}
    try:
        git["head"] = (
            subprocess.run(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            ).stdout.strip()
            or None
        )
        status = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--"] + files,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        ).stdout.strip()
        git["dirty_paths"] = status.splitlines() if status else []
    except Exception:  # pragma: no cover - probe only
        pass
    return {"files_sha256": hashes, "git": git}


def await_loader_stable(
    stable_seconds: float | None = None,
    timeout_s: float = 900.0,
    poll_s: float = 5.0,
) -> dict[str, Any]:
    if stable_seconds is None:
        stable_seconds = float(os.environ.get("MPS_STABILITY_SECONDS", "45"))
    """Wait until no teammate edit has touched the loader for ``stable_seconds``.

    Other teammates edit ``wyat/loading/ouro.py`` concurrently; a timing run must
    not straddle two loader revisions. Returns the stable fingerprint.
    """
    started = time.time()
    last: str | None = None
    last_change = time.time()
    while time.time() - started < timeout_s:
        current = loader_fingerprint()
        sha = json.dumps(current["files_sha256"], sort_keys=True) + str(
            current["git"]["head"]
        )
        if sha != last:
            last = sha
            last_change = time.time()
        elif time.time() - last_change >= stable_seconds:
            print(
                f"[loader] stable for {stable_seconds:.0f}s after "
                f"{time.time() - started:.0f}s: {current['files_sha256'].get('wyat/loading/ouro.py')}",
                flush=True,
            )
            return {
                "stable_seconds": stable_seconds,
                "waited_s": round(time.time() - started, 1),
                "loader": current,
            }
        time.sleep(poll_s)
    raise SystemExit(
        f"loader did not stay unchanged for {stable_seconds:.0f}s within {timeout_s:.0f}s"
    )


def loader_run_guard(start: dict[str, Any]) -> dict[str, Any]:
    """Compare the loader fingerprint at start vs receipt-write time."""
    end = loader_fingerprint()
    same_sha = start["files_sha256"] == end["files_sha256"]
    same_head = start["git"]["head"] == end["git"]["head"]
    guard = {
        "loader_at_start": start,
        "loader_at_write": end,
        "loader_unchanged_during_run": bool(same_sha and same_head),
    }
    if not guard["loader_unchanged_during_run"]:
        print(
            "[loader] WARNING: loader changed during this run; receipt is flagged "
            "and the arm must be re-run against a pinned revision",
            flush=True,
        )
    return guard


def write_receipt(name: str, payload: dict[str, Any]) -> Path:
    """Write one JSON receipt with host/loader/artifact/caps headers attached."""
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "notice": PROFILING_NOTICE,
        "name": name,
        "generated_at": now_iso(),
        "command": sys.argv,
        "host": host_fingerprint(),
        "loader": loader_fingerprint(),
        "artifacts": {
            "mps": artifact_fingerprint(MPS_ARTIFACT),
            "cpu_reference": artifact_fingerprint(CPU_ARTIFACT),
        },
        "caps": {
            "PYTORCH_MPS_LOW_WATERMARK_RATIO": os.environ.get(
                "PYTORCH_MPS_LOW_WATERMARK_RATIO"
            ),
            "PYTORCH_MPS_HIGH_WATERMARK_RATIO": os.environ.get(
                "PYTORCH_MPS_HIGH_WATERMARK_RATIO"
            ),
            "PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get(
                "PYTORCH_ENABLE_MPS_FALLBACK"
            ),
            "mps_fraction": DEFAULT_MPS_FRACTION,
            "rss_watchdog_cap_gib": DEFAULT_RSS_CAP_GIB,
            "torch_num_threads": torch.get_num_threads(),
            "declared_atol": DECLARED_ATOL,
        },
        "peak_rss_gib": peak_rss_gib(),
    }
    data.update(payload)
    path = RECEIPT_DIR / f"{name}.json"
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"[receipt] wrote {path}", flush=True)
    return path


def write_logits(name: str, payload: dict[str, Any]) -> Path:
    """Write a logits/parity replay file (inputs live in the question set)."""
    LOGIT_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "schema": "ouro-speed-mps-logits/v1",
        "notice": PROFILING_NOTICE,
        "name": name,
        "generated_at": now_iso(),
        "loader": loader_fingerprint(),
        "artifact": artifact_fingerprint(MPS_ARTIFACT),
        "input_question_set_sha256": _question_set_sha(),
        "declared_atol": DECLARED_ATOL,
    }
    data.update(payload)
    path = LOGIT_DIR / f"{name}.json"
    path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    print(f"[logits] wrote {path}", flush=True)
    return path


def _question_set_sha() -> str | None:
    path = question_set_path()
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("sha256")


def require_question_set_sha() -> str:
    sha = _question_set_sha()
    if sha is None:
        raise SystemExit("question-set.json is missing; run ref_cpu.py first")
    return sha


def stage_breakdown(
    backbone: Any,
    head: Any,
    question: dict[str, Any],
    device: str,
    *,
    repeats: int = 5,
    warmup: int = 1,
    context_steps: int = 4,
    candidate_steps: int = 2,
) -> dict[str, Any]:
    """Prefill / candidate-only / readout / end-to-end split for one question."""
    from wyat.loading.ouro import (
        prefill_context,
        run_candidate_steps,
        score_candidates_from_cache,
    )

    context, candidates = to_device(question, device)

    def prefill() -> Any:
        return prefill_context(
            backbone, context, policy="last", context_steps=context_steps
        )

    cache = prefill()

    def candidate_stage() -> torch.Tensor:
        return score_candidates_from_cache(
            backbone,
            head,
            cache,
            candidates,
            candidate_steps=candidate_steps,
        )

    hidden = torch.cat(
        [
            run_candidate_steps(
                backbone, cache, candidate, candidate_steps=candidate_steps
            )[:, -1]
            for candidate in candidates
        ],
        dim=0,
    )

    def readout() -> torch.Tensor:
        return head(hidden)

    def end_to_end() -> torch.Tensor:
        fresh = prefill_context(
            backbone, context, policy="last", context_steps=context_steps
        )
        return score_candidates_from_cache(
            backbone,
            head,
            fresh,
            candidates,
            candidate_steps=candidate_steps,
        )

    result = {
        "context_tokens": int(context.shape[1]),
        "candidate_count": len(candidates),
        "candidate_lengths": [int(item.shape[1]) for item in candidates],
        "prefill": timed(prefill, repeats=repeats, warmup=warmup, device=device),
        "candidate_from_cache": timed(
            candidate_stage, repeats=repeats, warmup=warmup, device=device
        ),
        "readout_head_only": timed(
            readout, repeats=repeats, warmup=warmup, device=device
        ),
        "end_to_end": timed(end_to_end, repeats=repeats, warmup=warmup, device=device),
        "memory": mps_memory(),
    }
    core = result["candidate_from_cache"]["median_s"]
    head_only = result["readout_head_only"]["median_s"]
    prefill_s = result["prefill"]["median_s"]
    e2e = result["end_to_end"]["median_s"]
    result["split_median_s"] = {
        "prefill": prefill_s,
        "candidate_only_est": round(max(core - head_only, 0.0), 4),
        "readout_head": head_only,
        "candidate_stage_incl_readout": core,
        "end_to_end": e2e,
        "unaccounted_e2e_minus_stages": round(prefill_s + core - e2e, 4),
        "prefill_share": round(prefill_s / e2e, 4) if e2e else None,
        "candidate_share": round(core / e2e, 4) if e2e else None,
    }
    return result


def profile_candidate_stage(
    backbone: Any,
    head: Any,
    question: dict[str, Any],
    device: str,
    *,
    repeats: int = 2,
    context_steps: int = 4,
    candidate_steps: int = 2,
    row_limit: int = 18,
) -> dict[str, Any]:
    """Low-overhead torch.profiler top-op table for the candidate stage."""
    from wyat.loading.ouro import prefill_context, score_candidates_from_cache

    context, candidates = to_device(question, device)
    cache = prefill_context(
        backbone, context, policy="last", context_steps=context_steps
    )
    score_candidates_from_cache(
        backbone,
        head,
        cache,
        candidates,
        candidate_steps=candidate_steps,
    )
    mps_sync()
    activities = [torch.profiler.ProfilerActivity.CPU]
    mps_activity = getattr(torch.profiler.ProfilerActivity, "MPS", None)
    if mps_activity is not None and torch.backends.mps.is_available():
        activities.append(mps_activity)
    try:
        with torch.profiler.profile(activities=activities) as profiler:
            for _ in range(repeats):
                score_candidates_from_cache(
                    backbone,
                    head,
                    cache,
                    candidates,
                    candidate_steps=candidate_steps,
                )
            mps_sync()
        events = list(profiler.key_averages())
        sort_by = (
            "self_device_time_total"
            if any(hasattr(event, "self_device_time_total") for event in events)
            else "self_cpu_time_total"
        )
        table = profiler.key_averages().table(sort_by=sort_by, row_limit=row_limit)
        rows = []
        for event in events:
            device_us = getattr(event, "self_device_time_total", 0.0)
            rows.append(
                {
                    "key": event.key,
                    "calls": event.count,
                    "self_device_us": round(device_us, 2),
                    "self_cpu_us": round(getattr(event, "self_cpu_time_total", 0.0), 2),
                }
            )
        rows.sort(key=lambda item: item["self_device_us"], reverse=True)
        return {
            "activities": [str(item) for item in activities],
            "repeats": repeats,
            "sort_by": sort_by,
            "table": table,
            "top_ops": rows[:row_limit],
        }
    except Exception as error:  # pragma: no cover - profiler availability probe
        return {
            "error": f"{type(error).__name__}: {error}",
            "note": "torch.profiler MPS activity unavailable; timing still valid",
        }


def emit(payload: Any) -> None:
    """Print a compact JSON blob to stdout for the operator log."""
    print(json.dumps(payload, indent=2, sort_keys=False), flush=True)
