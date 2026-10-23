"""Shared utilities for the Ouro-on-Mac speed harness (profiling only).

Nothing here builds or claims a release artifact. All heavy runs are expected
to enforce, in this order:

  1. exclusive bench lock ``/tmp/ouro-speed/BENCH_LOCK`` (``mkdir`` + 10 s retry)
  2. ``nice`` 10 (configurable, never negative)
  3. torch intra-op threads <= 6
  4. resident-set watchdog: hard abort above ``--rss-cap-gib`` (default 10 GiB,
     absolute maximum 12 GiB)
  5. for MPS: process memory fraction <= 0.6 plus
     ``PYTORCH_MPS_LOW_WATERMARK_RATIO=1.0`` and
     ``PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.2`` set before the MPS allocator
     initialises (HIGH=1.0 alone trips "invalid low watermark ratio 1.4" when
     ``torch.mps.set_per_process_memory_fraction`` runs, per Lead's measurement)

This module deliberately does *not* import torch: the benchmark CLIs call
:func:`prescan_env` on ``sys.argv`` before importing torch so ``OMP_NUM_THREADS``
and the MPS watermark variable are in place early. Timing/tolerance/receipt
helpers are torch-free; device-memory helpers are lazy to keep it that way.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import math
import os
import platform
import resource
import shutil
import statistics
import struct
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

GIB = float(1024**3)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRATCH = Path("/tmp/ouro-speed")
LOCK_DIR = SCRATCH / "BENCH_LOCK"
OURO_SOURCE = REPO_ROOT / "wyat" / "loading" / "ouro.py"

DEFAULT_THREADS = 6
MAX_THREADS = 6
DEFAULT_RSS_CAP_GIB = 10.0
MAX_RSS_CAP_GIB = 12.0
DEFAULT_MPS_FRACTION = 0.6
MAX_MPS_FRACTION = 0.6
DEFAULT_LOCK_TIMEOUT_S = 3600.0
LOCK_RETRY_S = 10.0
WATCHDOG_INTERVAL_S = 0.25
EXIT_RSS_CAP = 42

DEFAULT_ATOL = 2e-5
RECEIPT_SCHEMA = "ouro-speed-receipt/v1"

PROFILING_NOTICE = (
    "PROFILING ONLY - NOT A RELEASE ARTIFACT. Synthetic token tensors, local "
    "non-release weights; do not cite these numbers as product benchmarks."
)


# ---------------------------------------------------------------------------
# Resident memory probes
# ---------------------------------------------------------------------------


class _TimeValue(threading.Thread.__class__ if False else object):  # pragma: no cover
    pass


def _darwin_resident_bytes() -> int | None:
    """Current resident set size via Mach ``task_info`` (macOS only)."""
    if sys.platform != "darwin":
        return None
    try:
        import ctypes

        class _TimeVal(ctypes.Structure):
            _fields_ = [
                ("seconds", ctypes.c_int32),
                ("microseconds", ctypes.c_int32),
            ]

        class _MachTaskBasicInfo(ctypes.Structure):
            _fields_ = [
                ("virtual_size", ctypes.c_uint64),
                ("resident_size", ctypes.c_uint64),
                ("resident_size_max", ctypes.c_uint64),
                ("user_time", _TimeVal),
                ("system_time", _TimeVal),
                ("policy", ctypes.c_int),
                ("suspend_count", ctypes.c_int),
            ]

        libc = ctypes.CDLL(None, use_errno=True)
        task = ctypes.c_uint.in_dll(libc, "mach_task_self_").value
        info = _MachTaskBasicInfo()
        count = ctypes.c_uint32(ctypes.sizeof(info) // ctypes.sizeof(ctypes.c_int))
        rc = libc.task_info(
            ctypes.c_uint(task),
            ctypes.c_uint(20),  # MACH_TASK_BASIC_INFO
            ctypes.byref(info),
            ctypes.byref(count),
        )
        if rc != 0:
            return None
        return int(info.resident_size)
    except Exception:  # pragma: no cover - platform probe fallback
        return None


def peak_rss_bytes() -> int:
    """Peak (high-water) RSS of this process, normalised to bytes."""
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return raw
    return raw * 1024  # Linux reports KiB


def current_rss_bytes() -> int:
    """Current RSS in bytes with a graceful fallback to the peak probe."""
    if sys.platform == "linux":
        try:
            with open("/proc/self/statm", encoding="ascii") as handle:
                pages = int(handle.read().split()[1])
            return pages * os.sysconf("SC_PAGE_SIZE")
        except Exception:  # pragma: no cover
            return peak_rss_bytes()
    value = _darwin_resident_bytes()
    return value if value else peak_rss_bytes()


def gib(value_bytes: float) -> float:
    return round(float(value_bytes) / GIB, 4)


# ---------------------------------------------------------------------------
# RSS watchdog
# ---------------------------------------------------------------------------


class RssCapExceeded(RuntimeError):
    """Raised when the process resident set exceeds the configured cap."""

    def __init__(self, cap_gib: float, peak_gib: float, current_gib: float) -> None:
        self.cap_gib = cap_gib
        self.peak_gib = peak_gib
        self.current_gib = current_gib
        super().__init__(
            f"RSS cap exceeded: peak {peak_gib:.2f} GiB / current {current_gib:.2f} "
            f"GiB > cap {cap_gib:.2f} GiB"
        )


class RssWatchdog:
    """Background resident-set watchdog with an optional hard process abort.

    ``hard=True`` (default for CLIs) writes the ``on_exceed`` receipt and calls
    ``os._exit(EXIT_RSS_CAP)`` from the watchdog thread, because a long-running
    torch op cannot be interrupted safely from Python. Set ``hard=False`` (unit
    tests) to only latch ``triggered`` and let the caller observe
    :meth:`check`.
    """

    def __init__(
        self,
        cap_gib: float,
        *,
        interval_s: float = WATCHDOG_INTERVAL_S,
        hard: bool = True,
        probe_current: Callable[[], int] | None = None,
        probe_peak: Callable[[], int] | None = None,
        on_exceed: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.cap_gib = float(cap_gib)
        self.cap_bytes = int(self.cap_gib * GIB)
        self.interval_s = float(interval_s)
        self.hard = bool(hard)
        self._probe_current = probe_current or current_rss_bytes
        self._probe_peak = probe_peak or peak_rss_bytes
        self.on_exceed = on_exceed
        self.triggered = False
        self.peak_gib = 0.0
        self.current_gib = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- synchronous checks -------------------------------------------------
    def poll(self) -> tuple[float, float]:
        """Return ``(current_gib, peak_gib)`` and raise if over the cap."""
        self.current_gib = gib(self._probe_current())
        self.peak_gib = gib(self._probe_peak())
        if self.cap_bytes > 0 and (self._probe_current_peak_bytes() > self.cap_bytes):
            raise RssCapExceeded(self.cap_gib, self.peak_gib, self.current_gib)
        return self.current_gib, self.peak_gib

    def _probe_current_peak_bytes(self) -> int:
        return max(int(self._probe_current()), int(self._probe_peak()))

    def check(self) -> tuple[float, float]:
        """Poll synchronously; raises :class:`RssCapExceeded` on violation."""
        if not self.enabled:
            return self.poll_quiet()
        return self.poll()

    def poll_quiet(self) -> tuple[float, float]:
        self.current_gib = gib(self._probe_current())
        self.peak_gib = gib(self._probe_peak())
        return self.current_gib, self.peak_gib

    @property
    def enabled(self) -> bool:
        return self.cap_gib > 0

    # -- background thread --------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self.poll()
            except RssCapExceeded:
                self.triggered = True
                report = {
                    "event": "rss_cap_exceeded",
                    "cap_gib": self.cap_gib,
                    "peak_gib": self.peak_gib,
                    "current_gib": self.current_gib,
                    "aborted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }
                print(
                    f"[watchdog] ABORT: RSS {self.current_gib:.2f} GiB current / "
                    f"{self.peak_gib:.2f} GiB peak > cap {self.cap_gib:.2f} GiB",
                    file=sys.stderr,
                    flush=True,
                )
                if self.on_exceed is not None:
                    try:
                        self.on_exceed(report)
                    except Exception:  # pragma: no cover - receipt must not mask abort
                        traceback.print_exc()
                if self.hard:
                    sys.stderr.flush()
                    sys.stdout.flush()
                    os._exit(EXIT_RSS_CAP)
                return

    def start(self) -> RssWatchdog:
        if self.enabled and self._thread is None:
            self._thread = threading.Thread(
                target=self._loop, name="rss-watchdog", daemon=True
            )
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=1.0)
            self._thread = None

    def __enter__(self) -> RssWatchdog:
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()


# ---------------------------------------------------------------------------
# Exclusive bench lock
# ---------------------------------------------------------------------------


class LockTimeout(RuntimeError):
    """Raised when the exclusive benchmark lock cannot be acquired in time."""


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - pid exists, other user
        return True
    except OSError:  # pragma: no cover
        return True
    return True


class BenchLock:
    """Exclusive model-loading/benchmark lock implemented with ``mkdir``.

    Sequential execution is a hard protocol requirement: never load the 1.4B
    model while another benchmarking process holds the lock.
    """

    def __init__(
        self,
        path: str | Path = LOCK_DIR,
        *,
        timeout_s: float = DEFAULT_LOCK_TIMEOUT_S,
        retry_s: float = LOCK_RETRY_S,
        stale_after_s: float = 3600.0,
        log: Callable[[str], None] | None = print,
    ) -> None:
        self.path = Path(path)
        self.timeout_s = float(timeout_s)
        self.retry_s = float(retry_s)
        self.stale_after_s = float(stale_after_s)
        self._log = log or (lambda _message: None)
        self.waited_s = 0.0
        self.acquired = False
        self._atexit_registered = False

    def _owner_info(self) -> dict[str, Any]:
        return {
            "pid": os.getpid(),
            "script": str(Path(sys.argv[0]).name),
            "argv": [str(item) for item in sys.argv[:16]],
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "host": platform.node(),
        }

    def _try_break_stale(self) -> bool:
        owner_file = self.path / "owner.json"
        try:
            age = time.time() - self.path.stat().st_mtime
            owner = json.loads(owner_file.read_text(encoding="utf-8"))
            pid = int(owner.get("pid", -1))
        except (OSError, ValueError, TypeError):
            return False
        if age < self.stale_after_s or pid <= 0 or _pid_alive(pid):
            return False
        self._log(
            f"[lock] breaking stale lock {self.path} held by dead pid {pid} "
            f"(age {age:.0f}s)"
        )
        shutil.rmtree(self.path, ignore_errors=True)
        return True

    def acquire(self) -> float:
        start = time.monotonic()
        deadline = start + self.timeout_s
        waited_before = 0.0
        while True:
            try:
                self.path.mkdir(parents=False, exist_ok=False)
            except FileExistsError:
                waited = time.monotonic() - start
                if waited_before == 0.0 and waited >= 0.0:
                    self._log(
                        f"[lock] waiting for exclusive bench lock {self.path} "
                        f"(retry every {self.retry_s:.0f}s)"
                    )
                if time.monotonic() >= deadline:
                    self.waited_s = time.monotonic() - start
                    raise LockTimeout(
                        f"could not acquire {self.path} within {self.timeout_s:.0f}s"
                    ) from None
                if self._try_break_stale():
                    continue
                time.sleep(min(self.retry_s, max(0.0, deadline - time.monotonic())))
                waited_before = time.monotonic() - start
                continue
            self.waited_s = time.monotonic() - start
            self.acquired = True
            try:
                (self.path / "owner.json").write_text(
                    json.dumps(self._owner_info(), indent=2) + "\n",
                    encoding="utf-8",
                )
            except OSError:  # pragma: no cover - owner metadata is advisory
                pass
            atexit.register(self.release)
            self._atexit_registered = True
            self._log(f"[lock] acquired {self.path} after {self.waited_s:.1f}s wait")
            return self.waited_s

    def release(self) -> None:
        if self._atexit_registered:
            try:
                atexit.unregister(self.release)
            except Exception:  # pragma: no cover
                pass
            self._atexit_registered = False
        if self.acquired:
            shutil.rmtree(self.path, ignore_errors=True)
            self.acquired = False

    def __enter__(self) -> BenchLock:
        self.acquire()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()


# ---------------------------------------------------------------------------
# Caps: nice, threads, MPS
# ---------------------------------------------------------------------------


def apply_nice(nice: int) -> int:
    """Raise process niceness by ``nice`` (validated 0..20); returns previous."""
    if type(nice) is not int or not 0 <= nice <= 20:
        raise SystemExit(f"--nice must be an integer in 0..20, got {nice!r}")
    if nice == 0:
        return os.nice(0)
    return os.nice(nice) - nice


def validate_threads(threads: int) -> int:
    if type(threads) is not int or not 1 <= threads <= MAX_THREADS:
        raise SystemExit(
            f"--threads must be an integer in 1..{MAX_THREADS} "
            f"(mandatory resource cap), got {threads!r}"
        )
    return threads


def configure_threads(threads: int) -> dict[str, Any]:
    """Cap torch intra-op threads; inter-op stays at 1 for determinism."""
    import torch

    validate_threads(threads)
    torch.set_num_threads(threads)
    interop = None
    try:
        torch.set_num_interop_threads(1)
        interop = 1
    except RuntimeError:  # already initialised; not fatal for benchmarking
        interop = torch.get_num_interop_threads()
    return {
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(interop),
        "env_omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }


def validate_mps_fraction(fraction: float) -> float:
    if not isinstance(fraction, (int, float)) or not 0 < float(fraction) <= (
        MAX_MPS_FRACTION
    ):
        raise SystemExit(
            f"--mps-fraction must be in (0, {MAX_MPS_FRACTION}] "
            f"(mandatory resource cap), got {fraction!r}"
        )
    return float(fraction)


def configure_mps(fraction: float) -> dict[str, Any]:
    """Apply MPS fraction cap and confirm the watermark env vars are in place."""
    import torch

    fraction = validate_mps_fraction(fraction)
    if not torch.backends.mps.is_available():
        raise SystemExit("--device mps requested but torch.backends.mps is unavailable")
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "1.0")
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "1.2")
    # A silent CPU fallback would make "MPS" timings and device audits lies.
    fallback_before = os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
    torch.mps.set_per_process_memory_fraction(fraction)
    return {
        "mps_fraction": fraction,
        "PYTORCH_MPS_LOW_WATERMARK_RATIO": os.environ.get(
            "PYTORCH_MPS_LOW_WATERMARK_RATIO"
        ),
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO": os.environ.get(
            "PYTORCH_MPS_HIGH_WATERMARK_RATIO"
        ),
        "PYTORCH_ENABLE_MPS_FALLBACK_unset_was": fallback_before,
        "mps_recommended_max_memory_gib": gib(torch.mps.recommended_max_memory()),
    }


def sync_device(device: str) -> None:
    """Block until queued device work finishes so wall-clock timings are real."""
    if str(device).lower() != "mps":
        return
    import torch

    torch.mps.synchronize()


def load_ouro_scorer(
    artifact: str | Path,
    *,
    device: str,
    max_length: int | None = None,
    attention: str | None = None,
    allow_device_override: bool = False,
    candidate_batching: bool = False,
) -> Any:
    """Load an artifact, forwarding the MPS opt-in kwarg when the loader has one.

    The artifact contract currently runs ``cpu-fp32`` artifacts on MPS without an
    explicit opt-in; a ``mps-fp32`` device policy with an explicit override flag is
    being added by the runtime owner. This helper keeps the harness working
    across that transition: if ``load_ouro`` exposes an override keyword it is
    passed through, otherwise a plain load is attempted and the mismatch error
    surfaces unchanged.
    """
    import inspect

    from wyat.loading.ouro import load_ouro

    parameters = inspect.signature(load_ouro).parameters
    override_kwarg = next(
        (
            name
            for name in (
                "allow_device_policy_override",
                "allow_device_override",
                "device_override",
                "allow_mps",
            )
            if name in parameters
        ),
        None,
    )
    kwargs: dict[str, Any] = {"device": device}
    if max_length is not None:
        kwargs["max_length"] = max_length
    if attention is not None:
        kwargs["attention"] = attention
    if "candidate_batching" in parameters:
        kwargs["candidate_batching"] = bool(candidate_batching)
    elif candidate_batching:
        print(
            "[caps] note: --candidate-batching given but load_ouro exposes no "
            "candidate_batching kwarg; running without it",
            flush=True,
        )
    if override_kwarg is not None:
        kwargs[override_kwarg] = bool(allow_device_override)
    elif allow_device_override:
        print(
            "[caps] note: --allow-device-override given but "
            "wyat.loading.ouro.load_ouro exposes no override kwarg yet; "
            "relying on current permissive MPS path",
            flush=True,
        )
    return load_ouro(artifact, **kwargs)


def validate_rss_cap(cap_gib: float) -> float:
    if not isinstance(cap_gib, (int, float)) or not 0 < float(cap_gib) <= (
        MAX_RSS_CAP_GIB
    ):
        raise SystemExit(
            f"--rss-cap-gib must be in (0, {MAX_RSS_CAP_GIB}] "
            f"(hard resource ceiling), got {cap_gib!r}"
        )
    return float(cap_gib)


def device_memory(device: str) -> dict[str, float | None]:
    """Current/driver MPS memory (GiB); ``None`` entries on CPU."""
    if str(device).lower() != "mps":
        return {"current_allocated_gib": None, "driver_allocated_gib": None}
    try:
        import torch

        return {
            "current_allocated_gib": gib(torch.mps.current_allocated_memory()),
            "driver_allocated_gib": gib(torch.mps.driver_allocated_memory()),
        }
    except Exception:  # pragma: no cover - MPS probe is best effort
        return {"current_allocated_gib": None, "driver_allocated_gib": None}


def prescan_env(argv: Sequence[str]) -> None:
    """Set thread/MPS env vars from argv *before* torch is imported."""

    def value(flag: str) -> str | None:
        for index, item in enumerate(argv):
            if item == flag and index + 1 < len(argv):
                return str(argv[index + 1])
            if item.startswith(flag + "="):
                return item.split("=", 1)[1]
        return None

    threads = value("--threads")
    if threads is not None:
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ.setdefault(name, threads)
    for flag in ("--device", "--device-a", "--device-b"):
        device = value(flag)
        if device and device.lower().startswith("mps"):
            os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "1.0")
            os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "1.2")
            os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)


# ---------------------------------------------------------------------------
# Hashing / provenance
# ---------------------------------------------------------------------------


def file_sha256(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def sha256_tensor(tensor: Any, label: str = "") -> str:
    """Stable SHA-256 over a tensor's shape/dtype bytes (CPU round-trip)."""
    detached = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(f"{label}|{tuple(detached.shape)}|{detached.dtype}\n".encode())
    digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def sha256_tensor_mapping(mapping: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(mapping):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_tensor(mapping[name], name).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def git_state(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    root = str(repo_root)

    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", root, *args],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    revision = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    status_lines = [line for line in status.splitlines() if line]
    return {
        "git_revision": revision or None,
        "git_dirty": bool(status_lines),
        "git_dirty_paths_count": len(status_lines),
        "git_dirty_paths_sample": status_lines[:12],
    }


def source_fingerprint() -> dict[str, Any]:
    """Provenance of the runtime under test; recorded in every receipt."""
    info: dict[str, Any] = {
        "path": str(OURO_SOURCE),
        "exists": OURO_SOURCE.exists(),
        **git_state(),
    }
    if OURO_SOURCE.exists():
        stat = OURO_SOURCE.stat()
        info.update(
            {
                "sha256": file_sha256(OURO_SOURCE),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return info


def artifact_fingerprint(path: str | Path) -> dict[str, Any]:
    """Manifest-derived identity of a ``wyat-ouro-decision-v1`` artifact."""
    root = Path(path)
    manifest_path = root / "ouro_manifest.json"
    info: dict[str, Any] = {
        "path": str(root.resolve()) if root.exists() else str(root),
        "exists": root.exists(),
        "manifest_path": str(manifest_path),
    }
    if not manifest_path.exists():
        return info
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files_sha256", {})
    sizes = {}
    total = 0
    for name in files:
        candidate = root / name
        if candidate.exists():
            size = candidate.stat().st_size
            sizes[name] = size
            total += size
    info.update(
        {
            "model_id": manifest.get("model_id"),
            "execution": manifest.get("execution"),
            "base": manifest.get("base"),
            "cache": manifest.get("cache"),
            "files_sha256": files,
            "file_sizes_bytes": sizes,
            "total_listed_bytes": total,
            "manifest_sha256": file_sha256(manifest_path),
        }
    )
    return info


def environment_snapshot() -> dict[str, Any]:
    import torch
    import transformers

    mem_gib = None
    try:
        mem_gib = gib(
            int(
                subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=True,
                ).stdout.strip()
            )
        )
    except Exception:  # pragma: no cover
        try:
            mem_gib = gib(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
        except Exception:
            mem_gib = None
    try:
        loadavg = list(os.getloadavg())
    except OSError:  # pragma: no cover
        loadavg = None
    return {
        "platform": platform.platform(),
        "mac_ver": platform.mac_ver()[0] or None,
        "machine": platform.machine(),
        "hostname": platform.node(),
        "cpu_count": os.cpu_count(),
        "total_memory_gib": mem_gib,
        "loadavg_at_start": loadavg,
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
        "nice": os.nice(0),
        "current_rss_gib": gib(current_rss_bytes()),
        "peak_rss_gib": gib(peak_rss_bytes()),
        "env_PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get(
            "PYTORCH_ENABLE_MPS_FALLBACK"
        ),
        "env_PYTORCH_MPS_LOW_WATERMARK_RATIO": os.environ.get(
            "PYTORCH_MPS_LOW_WATERMARK_RATIO"
        ),
        "env_PYTORCH_MPS_HIGH_WATERMARK_RATIO": os.environ.get(
            "PYTORCH_MPS_HIGH_WATERMARK_RATIO"
        ),
    }


# ---------------------------------------------------------------------------
# Deterministic synthetic inputs
# ---------------------------------------------------------------------------


def fingerprint_token_inputs(
    spec_core: Mapping[str, Any],
    context: Any,
    candidates: Sequence[Any],
) -> dict[str, Any]:
    """Canonical spec + SHA-256 over synthesised/frozen token IDs.

    ``input_sha256`` hashes the spec core (sorted JSON) followed by context and
    per-candidate token bytes. ``legacy_concat_sha256`` hashes the concatenated
    token bytes exactly like the Lead's scratch baseline, so frozen question sets
    can be cross-checked (canonical P400/K6/S16 seed 20261021 -> 61aa0f6d...).
    Tensor arguments may be torch or numpy arrays; torch is imported lazily only
    for concatenation-free paths.
    """
    import hashlib

    spec = dict(spec_core)
    digest = hashlib.sha256()
    digest.update(canonical_json(spec).encode("utf-8"))
    digest.update(b"\0context\0")
    digest.update(context.numpy().tobytes())
    for index, candidate in enumerate(candidates):
        digest.update(f"\0candidate:{index}\0".encode("ascii"))
        digest.update(candidate.numpy().tobytes())
    spec["input_sha256"] = digest.hexdigest()
    import numpy as np

    legacy = hashlib.sha256()
    contiguous = np.concatenate(
        [context.numpy()] + [candidate.numpy() for candidate in candidates], axis=1
    )
    legacy.update(contiguous.tobytes())
    spec["legacy_concat_sha256"] = legacy.hexdigest()
    spec["context_shape"] = list(context.shape)
    spec["candidate_shapes"] = [list(candidate.shape) for candidate in candidates]
    return spec


def make_inputs(
    *,
    seed: int,
    vocab_size: int,
    context_tokens: int,
    candidates: int,
    candidate_length: int,
    candidate_lengths: Sequence[int] | None = None,
    policy: str = "last",
) -> dict[str, Any]:
    """Reproducible synthetic token inputs.

    A single CPU ``torch.Generator`` seeded with ``seed`` draws the context and
    then every candidate in order, so the same seed yields byte-identical token
    IDs regardless of device. The generator itself is described in the receipt;
    no tokenizer or encoded dataset is involved.

    Returns a dict with tensors under ``context``/``candidates`` plus the
    canonical input spec and its SHA-256 fingerprint.
    """
    import torch

    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if context_tokens <= 0 or candidates <= 0 or candidate_length <= 0:
        raise ValueError("token counts must be positive")
    if candidate_lengths is not None:
        lengths = [int(length) for length in candidate_lengths]
        if len(lengths) != candidates:
            raise ValueError(
                f"--candidate-lengths has {len(lengths)} entries but "
                f"--candidates is {candidates}"
            )
        if any(length <= 0 for length in lengths):
            raise ValueError("candidate lengths must be positive")
    else:
        lengths = [int(candidate_length)] * candidates
    generator = torch.Generator(device="cpu").manual_seed(seed)
    context = torch.randint(
        0, vocab_size, (1, context_tokens), generator=generator, dtype=torch.long
    )
    candidate_tensors = [
        torch.randint(0, vocab_size, (1, length), generator=generator, dtype=torch.long)
        for length in lengths
    ]
    spec = {
        "generator": f"torch.Generator(device='cpu').manual_seed({seed})",
        "seed": seed,
        "vocab_size": int(vocab_size),
        "context_tokens": int(context_tokens),
        "candidate_count": int(candidates),
        "candidate_lengths": lengths,
        "policy": policy,
        "note": "synthetic uniform token IDs; no tokenizer or dataset used",
    }
    spec = fingerprint_token_inputs(spec, context, candidate_tensors)
    return {
        "spec": spec,
        "context": context,
        "candidates": candidate_tensors,
        "sha256": spec["input_sha256"],
    }


def artifact_vocab_size(artifact: str | Path) -> int:
    config = json.loads((Path(artifact) / "config.json").read_text(encoding="utf-8"))
    return int(config["vocab_size"])


# ---------------------------------------------------------------------------
# Timing / receipts
# ---------------------------------------------------------------------------


def summarize_samples(samples: Sequence[float]) -> dict[str, Any]:
    """Median/min/max/mean/stdev plus raw samples (seconds, 6 dp)."""
    values = [float(value) for value in samples]
    if not values:
        raise ValueError("no timing samples to summarize")
    return {
        "median_s": round(statistics.median(values), 6),
        "min_s": round(min(values), 6),
        "max_s": round(max(values), 6),
        "mean_s": round(statistics.fmean(values), 6),
        "stdev_s": round(statistics.stdev(values), 6) if len(values) > 1 else 0.0,
        "samples_s": [round(value, 6) for value in values],
        "n": len(values),
    }


class Progress:
    """Live human-readable progress lines: stage, elapsed, RSS, device memory."""

    def __init__(
        self,
        *,
        device: str = "cpu",
        stream: Any = None,
        device_memory_fn: Callable[[str], dict[str, Any]] | None = None,
    ) -> None:
        self.device = str(device).lower()
        self.stream = stream or sys.stdout
        self._device_memory_fn = device_memory_fn or device_memory
        self.t0 = time.perf_counter()

    def line(self, stage: str, **fields: Any) -> str:
        elapsed = time.perf_counter() - self.t0
        memory = self._device_memory_fn(self.device)
        parts = [
            f"[{elapsed:8.2f}s]",
            f"rss_now={gib(current_rss_bytes()):5.2f}GiB",
            f"rss_peak={gib(peak_rss_bytes()):5.2f}GiB",
        ]
        if self.device == "mps":
            parts.append(
                "mps_cur={:.2f}GiB mps_driver={:.2f}GiB".format(
                    memory.get("current_allocated_gib") or 0.0,
                    memory.get("driver_allocated_gib") or 0.0,
                )
            )
        rendered = " ".join(parts)
        for key, value in fields.items():
            if isinstance(value, float):
                rendered += f" {key}={value:.3f}"
            else:
                rendered += f" {key}={value}"
        line = f"{rendered} | {stage}"
        print(line, file=self.stream, flush=True)
        return line


def write_json(path: str | Path, payload: Mapping[str, Any]) -> str:
    target = Path(path)
    if not target.is_absolute():
        target = Path.cwd() / target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return str(target)


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Logit comparison (declared tolerance; shared by parity.py and receipts)
# ---------------------------------------------------------------------------


def _float32_bits(value: float) -> bytes:
    return struct.pack("<f", float(value))


def compare_logits(
    a: Sequence[float],
    b: Sequence[float],
    *,
    atol: float = DEFAULT_ATOL,
    rank_fail: bool = False,
    label_a: str = "a",
    label_b: str = "b",
) -> dict[str, Any]:
    """Compare two candidate-score vectors against a declared absolute tolerance.

    Reports: lengths/shape, bitwise equality, max abs/rel diff, per-candidate
    deltas with pass/fail, and rank-order changes. ``passed`` is False when
    lengths differ, any value is non-finite, or any ``abs_diff > atol``. Rank
    changes are reported as a warning unless ``rank_fail=True``.
    """
    values_a = [float(value) for value in a]
    values_b = [float(value) for value in b]
    report: dict[str, Any] = {
        "label_a": label_a,
        "label_b": label_b,
        "tolerance": {"abs": float(atol), "gates": "max_abs_diff <= atol"},
        "length_a": len(values_a),
        "length_b": len(values_b),
        "rank_change_fails": bool(rank_fail),
    }
    if len(values_a) != len(values_b):
        report.update(
            {
                "passed": False,
                "error": f"length mismatch: {len(values_a)} vs {len(values_b)}",
            }
        )
        return report
    if not values_a:
        report.update({"passed": False, "error": "empty score vectors"})
        return report
    finite = all(math.isfinite(value) for value in values_a + values_b)
    report["finite"] = finite
    bitwise = len(values_a) == len(values_b) and all(
        _float32_bits(left) == _float32_bits(right)
        for left, right in zip(values_a, values_b, strict=True)
    )
    deltas = [left - right for left, right in zip(values_a, values_b, strict=True)]
    abs_deltas = [abs(delta) for delta in deltas]
    max_abs = max(abs_deltas)
    rel_deltas = [
        abs_delta / max(abs(left), abs(right), 1e-12)
        for abs_delta, left, right in zip(abs_deltas, values_a, values_b, strict=True)
    ]
    per_candidate = []
    for index, (left, right, delta, abs_delta, rel_delta) in enumerate(
        zip(values_a, values_b, deltas, abs_deltas, rel_deltas, strict=True)
    ):
        per_candidate.append(
            {
                "index": index,
                "a": left,
                "b": right,
                "delta_a_minus_b": delta,
                "abs_diff": abs_delta,
                "rel_diff": rel_delta,
                "within_tolerance": abs_delta <= atol,
            }
        )

    def ranks(values: Sequence[float]) -> list[int]:
        return sorted(range(len(values)), key=lambda index: (-values[index], index))

    order_a = ranks(values_a)
    order_b = ranks(values_b)
    rank_changed = order_a != order_b
    top1_changed = order_a[0] != order_b[0]
    pairs = []
    if rank_changed:
        position_b = {candidate: position for position, candidate in enumerate(order_b)}
        for position, candidate in enumerate(order_a):
            if position_b[candidate] != position:
                pairs.append(
                    {
                        "candidate": candidate,
                        "rank_a": position,
                        "rank_b": position_b[candidate],
                    }
                )
    passed = finite and max_abs <= atol
    if rank_fail and rank_changed:
        passed = False
    report.update(
        {
            "bitwise_equal": bool(bitwise),
            "max_abs_diff": max_abs,
            "max_rel_diff": max(rel_deltas),
            "mean_abs_diff": statistics.fmean(abs_deltas),
            "rank_order_a": order_a,
            "rank_order_b": order_b,
            "rank_order_changed": rank_changed,
            "top1_changed": top1_changed,
            "rank_changes": pairs,
            "per_candidate": per_candidate,
            "violations": sum(
                1 for item in per_candidate if not item["within_tolerance"]
            ),
            "passed": bool(passed),
        }
    )
    return report


def format_comparison(report: Mapping[str, Any]) -> str:
    """One-screen human summary for a :func:`compare_logits` report."""
    lines = [
        f"parity {report.get('label_a')} vs {report.get('label_b')}: "
        f"{'PASS' if report.get('passed') else 'FAIL'} "
        f"(tolerance abs {report.get('tolerance', {}).get('abs')})",
        f"  bitwise_equal={report.get('bitwise_equal')} "
        f"max_abs_diff={report.get('max_abs_diff')!r} "
        f"max_rel_diff={report.get('max_rel_diff')!r} "
        f"violations={report.get('violations')}",
        f"  rank_order_changed={report.get('rank_order_changed')} "
        f"top1_changed={report.get('top1_changed')} "
        f"ranks={report.get('rank_order_a')} vs {report.get('rank_order_b')}",
    ]
    for item in report.get("per_candidate", []):
        lines.append(
            "  candidate {index}: a={a:+.8f} b={b:+.8f} "
            "abs_diff={abs_diff:.3e} rel_diff={rel_diff:.3e} "
            "ok={within_tolerance}".format(**item)
        )
    return "\n".join(lines)


__all__ = [
    "BenchLock",
    "DEFAULT_ATOL",
    "DEFAULT_LOCK_TIMEOUT_S",
    "DEFAULT_MPS_FRACTION",
    "DEFAULT_RSS_CAP_GIB",
    "DEFAULT_THREADS",
    "EXIT_RSS_CAP",
    "GIB",
    "LOCK_DIR",
    "LOCK_RETRY_S",
    "LockTimeout",
    "MAX_MPS_FRACTION",
    "MAX_RSS_CAP_GIB",
    "MAX_THREADS",
    "PROFILING_NOTICE",
    "Progress",
    "RECEIPT_SCHEMA",
    "REPO_ROOT",
    "RssCapExceeded",
    "RssWatchdog",
    "SCRATCH",
    "apply_nice",
    "artifact_fingerprint",
    "artifact_vocab_size",
    "compare_logits",
    "configure_mps",
    "configure_threads",
    "current_rss_bytes",
    "device_memory",
    "environment_snapshot",
    "file_sha256",
    "format_comparison",
    "gib",
    "git_state",
    "load_ouro_scorer",
    "make_inputs",
    "peak_rss_bytes",
    "prescan_env",
    "read_json",
    "sha256_json",
    "sha256_tensor",
    "sha256_tensor_mapping",
    "source_fingerprint",
    "summarize_samples",
    "sync_device",
    "validate_mps_fraction",
    "validate_rss_cap",
    "validate_threads",
    "WATCHDOG_INTERVAL_S",
    "write_json",
]
