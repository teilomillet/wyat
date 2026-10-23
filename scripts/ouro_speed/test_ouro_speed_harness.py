"""Tiny self-tests for the Ouro speed harness.

These tests deliberately never load the 1.4B model: they exercise the CLIs'
``--help`` / argument validation, the RSS watchdog, the bench lock, deterministic
synthetic inputs, the tolerance comparator, the transfer/cat counters and the
receipt renderers with fake or tiny tensors only.

Run with::

  cd /Users/teilomillet/Code/wyat && .venv/bin/python -m pytest \
      scripts/ouro_speed/test_ouro_speed_harness.py -q
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import _common as common  # noqa: E402
import audit_transfers  # noqa: E402
import bench  # noqa: E402
import make_results_table  # noqa: E402
import parity  # noqa: E402
import torch  # noqa: E402

SCRIPTS = (
    "bench.py",
    "parity.py",
    "audit_transfers.py",
    "build_local_artifact.py",
)


# ---------------------------------------------------------------------------
# CLI smoke tests (no model load)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("script", SCRIPTS)
def test_script_help_smoke(script: str) -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / script), "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()


def test_bench_rejects_unsafe_caps() -> None:
    parser = bench.build_parser()
    base = ["--artifact", "/tmp/does-not-exist"]
    cases = (
        ["--threads", "7"],
        ["--threads", "0"],
        ["--rss-cap-gib", "13"],
        ["--rss-cap-gib", "0"],
        ["--mps-fraction", "0.7"],
        ["--nice", "21"],
        ["--repeats", "0"],
        ["--context-tokens", "0"],
        ["--candidates", "0"],
        ["--candidate-lengths", "16,16"],  # count mismatch with default K=6
    )
    for extra in cases:
        with pytest.raises(SystemExit):
            args = parser.parse_args(base + extra)
            bench.validate_args(parser, args)


def test_bench_accepts_safe_caps() -> None:
    parser = bench.build_parser()
    args = parser.parse_args(
        [
            "--artifact",
            "/tmp/does-not-exist",
            "--threads",
            "6",
            "--rss-cap-gib",
            "10",
            "--mps-fraction",
            "0.6",
            "--nice",
            "10",
            "--repeats",
            "5",
            "--candidate-lengths",
            "16,16,16,16,16,16",
        ]
    )
    bench.validate_args(parser, args)
    assert args.candidates == 6
    assert bench.parse_lengths(args.candidate_lengths) == [16] * 6


def test_parity_default_tolerance_is_declared_2e5() -> None:
    parser = parity.build_parser()
    args = parser.parse_args(["--a", "a.pt", "--b", "b.pt"])
    assert args.atol == 2e-5


# ---------------------------------------------------------------------------
# RSS watchdog / caps
# ---------------------------------------------------------------------------


def test_rss_watchdog_under_cap_is_quiet() -> None:
    watchdog = common.RssWatchdog(
        10.0 if False else 1.0,
        hard=False,
        probe_current=lambda: 100 * 1024 * 1024,
        probe_peak=lambda: 200 * 1024 * 1024,
        interval_s=0.05,
    )
    current, peak = watchdog.check()
    assert current == pytest.approx(common.gib(100 * 1024 * 1024))
    assert peak == pytest.approx(common.gib(200 * 1024 * 1024))
    assert watchdog.triggered is False


def test_rss_watchdog_raises_over_cap() -> None:
    watchdog = common.RssWatchdog(
        0.001,
        hard=False,
        probe_current=lambda: 10**9,
        probe_peak=lambda: 2 * 10**9,
        interval_s=0.05,
    )
    with pytest.raises(common.RssCapExceeded) as error:
        watchdog.check()
    assert error.value.cap_gib == 0.001
    assert "RSS cap exceeded" in str(error.value)


def test_rss_watchdog_thread_latches_without_hard_exit() -> None:
    watchdog = common.RssWatchdog(
        0.001,
        hard=False,
        probe_current=lambda: 10**9,
        probe_peak=lambda: 10**9,
        interval_s=0.01,
    )
    watchdog.start()
    try:
        deadline = time.monotonic() + 2.0
        while not watchdog.triggered and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        watchdog.stop()
    assert watchdog.triggered is True


def test_cap_validators() -> None:
    with pytest.raises(SystemExit):
        common.validate_threads(7)
    with pytest.raises(SystemExit):
        common.validate_rss_cap(12.5)
    with pytest.raises(SystemExit):
        common.validate_mps_fraction(0.61)
    assert common.validate_threads(6) == 6
    assert common.validate_rss_cap(10.0) == 10.0
    assert common.validate_mps_fraction(0.6) == 0.6


# ---------------------------------------------------------------------------
# Bench lock
# ---------------------------------------------------------------------------


def test_lock_acquire_release_roundtrip(tmp_path: Path) -> None:
    lock_path = tmp_path / "BENCH_LOCK"
    lock = common.BenchLock(lock_path, timeout_s=0.5, retry_s=0.05, log=None)
    waited = lock.acquire()
    try:
        assert waited >= 0.0
        assert lock.acquired is True
        assert (lock_path / "owner.json").exists()
    finally:
        lock.release()
    assert not lock_path.exists()
    again = common.BenchLock(lock_path, timeout_s=0.5, retry_s=0.05, log=None)
    again.acquire()
    again.release()


def test_lock_timeout_when_held(tmp_path: Path) -> None:
    lock_path = tmp_path / "BENCH_LOCK"
    first = common.BenchLock(lock_path, timeout_s=0.5, retry_s=0.05, log=None)
    first.acquire()
    second = common.BenchLock(lock_path, timeout_s=0.2, retry_s=0.02, log=None)
    try:
        with pytest.raises(common.LockTimeout):
            second.acquire()
    finally:
        first.release()


def test_lock_breaks_stale_dead_pid(tmp_path: Path) -> None:
    lock_path = tmp_path / "BENCH_LOCK"
    lock_path.mkdir()
    (lock_path / "owner.json").write_text(
        json.dumps({"pid": 2_147_483_600, "script": "crashed.py"}), encoding="utf-8"
    )
    os.utime(lock_path, (0, 0))
    lock = common.BenchLock(
        lock_path, timeout_s=0.5, retry_s=0.02, stale_after_s=0.0, log=None
    )
    lock.acquire()
    lock.release()
    assert not lock_path.exists()


# ---------------------------------------------------------------------------
# Deterministic synthetic inputs
# ---------------------------------------------------------------------------


def test_make_inputs_is_deterministic() -> None:
    first = common.make_inputs(
        seed=7, vocab_size=128, context_tokens=16, candidates=3, candidate_length=4
    )
    second = common.make_inputs(
        seed=7, vocab_size=128, context_tokens=16, candidates=3, candidate_length=4
    )
    assert first["sha256"] == second["sha256"]
    assert torch.equal(first["context"], second["context"])
    third = common.make_inputs(
        seed=8, vocab_size=128, context_tokens=16, candidates=3, candidate_length=4
    )
    assert first["sha256"] != third["sha256"]
    assert first["spec"]["candidate_count"] == 3
    assert first["spec"]["candidate_lengths"] == [4, 4, 4]
    for candidate in first["candidates"]:
        assert int(candidate.min()) >= 0 and int(candidate.max()) < 128


def test_make_inputs_heterogeneous_lengths() -> None:
    inputs = common.make_inputs(
        seed=3,
        vocab_size=64,
        context_tokens=8,
        candidates=3,
        candidate_length=4,
        candidate_lengths=[2, 5, 7],
    )
    assert [tensor.shape[1] for tensor in inputs["candidates"]] == [2, 5, 7]
    assert inputs["spec"]["candidate_lengths"] == [2, 5, 7]


def test_make_inputs_rejects_bad_lengths() -> None:
    with pytest.raises(ValueError):
        common.make_inputs(
            seed=3,
            vocab_size=64,
            context_tokens=8,
            candidates=2,
            candidate_length=4,
            candidate_lengths=[2, 5, 7],
        )


def test_summarize_samples() -> None:
    summary = common.summarize_samples([1.0, 2.0, 3.0])
    assert summary["median_s"] == 2.0
    assert summary["min_s"] == 1.0
    assert summary["max_s"] == 3.0
    assert summary["n"] == 3
    with pytest.raises(ValueError):
        common.summarize_samples([])


# ---------------------------------------------------------------------------
# Tolerance comparator
# ---------------------------------------------------------------------------


def test_compare_logits_pass_within_tolerance() -> None:
    a = [-0.79431, -0.168728, -0.220501]
    b = [value + 4.6e-6 for value in a]
    report = common.compare_logits(a, b, atol=2e-5)
    assert report["passed"] is True
    assert report["bitwise_equal"] is False
    assert report["max_abs_diff"] < 2e-5
    assert report["rank_order_changed"] is False
    assert all(item["within_tolerance"] for item in report["per_candidate"])


def test_compare_logits_bitwise_identical() -> None:
    a = [-0.5, 0.25, 1.5]
    report = common.compare_logits(a, list(a))
    assert report["passed"] is True
    assert report["bitwise_equal"] is True
    assert report["max_abs_diff"] == 0.0


def test_compare_logits_fails_outside_tolerance() -> None:
    report = common.compare_logits([0.0, 0.0], [1e-3, 0.0], atol=2e-5)
    assert report["passed"] is False
    assert report["violations"] == 1
    assert report["tolerance"]["abs"] == 2e-5


def test_compare_logits_at_exact_tolerance_passes() -> None:
    report = common.compare_logits([0.0], [2e-5], atol=2e-5)
    assert report["passed"] is True


def test_compare_logits_rank_change_detection() -> None:
    a = [1.0, 2.0]
    b = [2.0, 1.0]
    lenient = common.compare_logits(a, b, atol=1.5)
    assert lenient["passed"] is True
    assert lenient["rank_order_changed"] is True
    assert lenient["top1_changed"] is True
    assert lenient["rank_changes"]
    strict = common.compare_logits(a, b, atol=1.5, rank_fail=True)
    assert strict["passed"] is False


def test_compare_logits_length_mismatch() -> None:
    report = common.compare_logits([1.0, 2.0], [1.0])
    assert report["passed"] is False
    assert "length mismatch" in report["error"]


def test_compare_logits_nonfinite_fails() -> None:
    report = common.compare_logits([float("nan")], [0.0])
    assert report["passed"] is False
    assert report["finite"] is False


def test_format_comparison_mentions_numbers() -> None:
    report = common.compare_logits([1.0], [1.0 + 1e-7])
    text = common.format_comparison(report)
    assert "parity" in text and "max_abs_diff" in text


# ---------------------------------------------------------------------------
# Transfer counters / cat audit
# ---------------------------------------------------------------------------


def test_transfer_counters_count_and_restore() -> None:
    original_to = torch.Tensor.to
    original_item = torch.Tensor.item
    counters = audit_transfers.TransferCounters(max_sites=10)
    counters.install()
    try:
        assert torch.Tensor.to is not original_to
        counters.enabled = True
        counters.phase = "test"
        tensor = torch.zeros(3)
        assert torch.tensor(0.0).item() == 0.0
        assert tensor.tolist() == [0.0, 0.0, 0.0]
        tensor.to("cpu")
        tensor.to(torch.float64)
        torch.cat([tensor, tensor], dim=0)
        counts = counters.counts
        assert counts["item:calls"] == 1
        assert counts["tolist:calls"] == 1
        assert counts["to:calls"] == 2
        assert counts["to:same_attrs"] == 1
        assert counts["to:dtype_only"] == 1
        assert counts["to:device_change"] == 0
        assert counts["sync:item:cpu"] == 1
        report = counters.report()
        assert report["sites"]
        assert any("call_site" in site for site in report["sites"])
    finally:
        counters.uninstall()
    assert torch.Tensor.to is original_to
    assert torch.Tensor.item is original_item


def test_transfer_counters_ignore_disabled_region() -> None:
    counters = audit_transfers.TransferCounters(max_sites=5)
    counters.install()
    try:
        torch.zeros(2).to("cpu")
        assert counters.counts.get("to:calls", 0) == 0
        counters.enabled = True
        torch.zeros(2).to("cpu")
        assert counters.counts["to:calls"] == 1
    finally:
        counters.uninstall()


def test_cat_audit_reports_placement() -> None:
    original_cat = torch.cat
    audit = audit_transfers.CatAudit(max_sites=5)
    audit.expected_device = "cpu"
    audit.install()
    try:
        audit.enabled = True
        output = torch.cat([torch.zeros(2), torch.ones(2)], dim=0)
        assert output.numel() == 4
        report = audit.report()
        assert report["enabled_calls"] == 1
        assert report["by_output"][0]["device"] == "cpu"
        assert report["by_output"][0]["contiguous"] is True
        assert report["total_output_bytes"] > 0
        assert report["total_output_mib"] > 0
        assert report["mixed_device_calls"] == 0
        assert report["off_device_calls"] == 0
    finally:
        audit.uninstall()
    assert torch.cat is original_cat


def test_inspect_module_tensors_on_fake_model() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(4, 2), torch.nn.LayerNorm(2))
    audit = audit_transfers.inspect_module_tensors(model, "fake", "cpu")
    assert audit["count"] == 4  # linear weight+bias, layernorm weight+bias
    assert audit["off_device_count"] == 0
    assert audit["by_device_dtype"][0]["device"] == "cpu"
    assert audit["noncontiguous_count"] == 0


def test_inspect_context_banks_on_fake_cache() -> None:
    entry = SimpleNamespace(key=torch.zeros(1, 2, 3, 4), value=torch.zeros(1, 2, 3, 4))
    fake = SimpleNamespace(policy="last", length=3, banks=((entry,),))
    audit = audit_transfers.inspect_context_banks(fake, "cpu")
    assert audit["banks"] == 1
    assert audit["layers_per_bank"] == 1
    assert audit["entries"] == 2
    assert audit["contiguous"] is True
    assert audit["off_device_count"] == 0
    assert audit["total_bytes"] == 2 * 1 * 2 * 3 * 4 * 4


# ---------------------------------------------------------------------------
# Parity file handling
# ---------------------------------------------------------------------------


def test_load_logits_pt_and_json_roundtrip(tmp_path: Path) -> None:
    values = [-0.79431, -0.168728, -0.220501]
    pt_path = tmp_path / "logits.pt"
    torch.save({"logits": torch.tensor(values), "input_sha256": "abc"}, pt_path)
    loaded, metadata = parity.load_logits(pt_path)
    assert loaded == pytest.approx(values)
    assert metadata["input_sha256"] == "abc"
    assert metadata["sha256"] and len(metadata["sha256"]) == 64

    json_path = tmp_path / "logits.json"
    json_path.write_text(json.dumps({"logits": values}), encoding="utf-8")
    loaded_json, _ = parity.load_logits(json_path)
    assert loaded_json == pytest.approx(values)


def test_parity_cli_file_mode_pass_and_fail(tmp_path: Path) -> None:
    base = tmp_path / "a.pt"
    near = tmp_path / "b.pt"
    far = tmp_path / "c.pt"
    values = [-0.79431, -0.168728, -0.220501]
    torch.save({"logits": torch.tensor(values)}, base)
    torch.save({"logits": torch.tensor([v + 3e-6 for v in values])}, near)
    torch.save({"logits": torch.tensor([v + 1e-2 for v in values])}, far)
    report = tmp_path / "parity.json"
    common_pass = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_DIR / "parity.py"),
            "--a",
            str(base),
            "--b",
            str(near),
            "--json-out",
            str(report),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert common_pass.returncode == 0, common_pass.stderr
    assert "PASS" in common_pass.stdout
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["comparison"]["max_abs_diff"] < 2e-5

    common_fail = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_DIR / "parity.py"),
            "--a",
            str(base),
            "--b",
            str(far),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert common_fail.returncode == 1
    assert "FAIL" in common_fail.stdout


def test_parity_cli_rank_change_gate(tmp_path: Path) -> None:
    a_path = tmp_path / "a.pt"
    b_path = tmp_path / "b.pt"
    torch.save({"logits": torch.tensor([1.0, 2.0])}, a_path)
    torch.save({"logits": torch.tensor([2.0, 1.0])}, b_path)
    base_command = [
        sys.executable,
        str(SCRIPT_DIR / "parity.py"),
        "--a",
        str(a_path),
        "--b",
        str(b_path),
        "--atol",
        "1.5",
    ]
    lenient = subprocess.run(
        base_command, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=120
    )
    assert lenient.returncode == 0, lenient.stderr
    strict = subprocess.run(
        base_command + ["--fail-on-rank-change"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert strict.returncode == 1


# ---------------------------------------------------------------------------
# Receipt rendering / env prescan
# ---------------------------------------------------------------------------


def test_render_markdown_contains_stage_and_grid_tables() -> None:
    receipt = {
        "schema": common.RECEIPT_SCHEMA,
        "status": "ok",
        "generated_at": "2026-10-21T00:00:00+0000",
        "notice": "profiling only",
        "caps": {"device": "cpu", "threads": 6, "nice": 10, "rss_cap_gib": 10.0},
        "environment": {"machine": "arm64", "cpu_count": 12, "total_memory_gib": 24},
        "provenance": {
            "source_end": {
                "sha256": "a" * 64,
                "git_revision": "deadbeef",
                "git_dirty": True,
            }
        },
        "inputs": {
            "seed": 1,
            "context_tokens": 400,
            "candidate_count": 2,
            "candidate_lengths": [16, 16],
            "input_sha256": "b" * 64,
        },
        "protocol": {"warmups": 2, "repeats": 5, "sync": "mps"},
        "stages": {
            "load": {"seconds": 16.0},
            "end_to_end": common.summarize_samples([3.3, 3.2, 3.4]),
        },
        "memory": {"peak_rss_gib": 8.44, "rss_cap_gib": 10.0, "rss_cap_hit": False},
        "scaling": {
            "candidate_count": {"1": common.summarize_samples([0.2, 0.21])},
            "context_grid": {
                "400": {
                    "prefill": common.summarize_samples([1.9, 2.0]),
                    "candidate_from_cache_by_k": {
                        "1": common.summarize_samples([0.2]),
                        "2": common.summarize_samples([0.45]),
                    },
                }
            },
        },
        "logits": {"values": [-0.5, 0.5], "sha256": "c" * 64},
        "profile": {"top_ops": [{"op": "aten::mm", "self_cpu_ms": 1.0, "calls": 2}]},
    }
    markdown = bench.render_markdown(receipt)
    assert "| end_to_end |" in markdown
    assert "## Candidate-count scaling" in markdown
    assert "## Context grid" in markdown
    assert "| K=2 |" in markdown
    assert "| `aten::mm` |" in markdown


def test_write_json_roundtrip(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "receipt.json"
    common.write_json(target, {"hello": 1})
    assert json.loads(target.read_text(encoding="utf-8")) == {"hello": 1}


def test_prescan_env_sets_mps_and_unset_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "PYTORCH_MPS_LOW_WATERMARK_RATIO",
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
        "OMP_NUM_THREADS",
        "PYTORCH_ENABLE_MPS_FALLBACK",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    common.prescan_env(["--device", "mps", "--threads", "4"])
    assert os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] == "1.0"
    assert os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] == "1.2"
    assert os.environ["OMP_NUM_THREADS"] == "4"
    assert "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ


# ---------------------------------------------------------------------------
# Frozen token inputs, golden key paths, integrity gates
# ---------------------------------------------------------------------------


def test_canonical_synthetic_digest_matches_lead_baseline() -> None:
    """The canonical P400/K6/S16 seed-20261021 tokens must hash to 61aa0f6d..."""
    inputs = common.make_inputs(
        seed=20261021,
        vocab_size=49152,
        context_tokens=400,
        candidates=6,
        candidate_length=16,
    )
    assert (
        inputs["spec"]["legacy_concat_sha256"]
        == "61aa0f6dea97598fbfaf355a12876100aba4ec237712f6f78f5ee90a1a20a31b"
    )


def _write_question_set(path: Path) -> dict:
    payload = {
        "schema": "ouro-speed-mps-question-set/v1",
        "sha256": "q" * 64,
        "headline": {
            "p400-k6-s16": {
                "id": "p400-k6-s16",
                "seed": 20261021,
                "context": list(range(1, 9)),
                "candidates": [[10, 11, 12], [13, 14]],
            }
        },
        "corpus": [],
        "real": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_load_tokens_inputs_from_question_set(tmp_path: Path) -> None:
    qs_path = tmp_path / "question-set.json"
    payload = _write_question_set(qs_path)
    parser = bench.build_parser()
    args = parser.parse_args(
        [
            "--artifact",
            "/tmp/does-not-exist",
            "--tokens-file",
            str(qs_path),
            "--tokens-name",
            "p400-k6-s16",
        ]
    )
    bench.validate_args(parser, args)
    inputs = bench.load_tokens_inputs(args)
    assert inputs["context"].shape == (1, 8)
    assert [candidate.shape[1] for candidate in inputs["candidates"]] == [3, 2]
    spec = inputs["spec"]
    assert spec["context_tokens"] == 8
    assert spec["candidate_count"] == 2
    assert spec["candidate_lengths"] == [3, 2]
    assert spec["question_set_sha256"] == payload["sha256"]
    assert spec["selection"] == {"section": "headline", "name": "p400-k6-s16"}
    # Legacy digest is sha256 over concatenated token bytes.
    import hashlib

    import numpy as np

    concatenated = np.concatenate(
        [inputs["context"].numpy()] + [c.numpy() for c in inputs["candidates"]],
        axis=1,
    )
    assert (
        spec["legacy_concat_sha256"]
        == hashlib.sha256(concatenated.tobytes()).hexdigest()
    )


def test_bench_tokens_file_validation() -> None:
    parser = bench.build_parser()
    cases = [
        ["--tokens-file", "/tmp/nope.json", "--tokens-name", "x"],
        ["--tokens-name", "x"],
        [
            "--tokens-file",
            "/tmp/does-not-exist.json",
            "--tokens-section",
            "corpus",
            "--scale-contexts",
            "256,400",
        ],
        ["--baseline-key", "a.b"],
    ]
    for extra in cases:
        with pytest.raises(SystemExit):
            args = parser.parse_args(["--artifact", "/tmp/x", *extra])
            bench.validate_args(parser, args)


def test_load_golden_logits_with_baseline_key(tmp_path: Path) -> None:
    path = tmp_path / "reference.json"
    path.write_text(
        json.dumps(
            {
                "headline_logits": {
                    "p400-k6-s16": [-0.5, 0.25, 1.5],
                },
                "loader_guard": {"sha256": "x"},
            }
        ),
        encoding="utf-8",
    )
    values, metadata = bench.load_golden_logits(path, key="headline_logits.p400-k6-s16")
    assert values == pytest.approx([-0.5, 0.25, 1.5])
    assert metadata["key"] == "headline_logits.p400-k6-s16"
    assert metadata["sha256"]
    with pytest.raises(SystemExit):
        bench.load_golden_logits(path, key="headline_logits.missing")


def test_build_golden_payload_shape() -> None:
    receipt = {
        "provenance": {"artifact": {"model_id": "m"}},
        "scorer": {"device": "cpu"},
        "inputs": {"seed": 1, "context_tokens": 4, "candidate_count": 2},
        "command": {"argv": ["bench.py"]},
        "caps": {"threads": 6},
    }
    payload = bench.build_golden_payload(receipt, [-0.1, 0.2])
    assert payload["schema"] == "ouro-speed-golden/v1"
    assert payload["logits"] == [-0.1, 0.2]
    assert payload["tolerance"]["rank_top1_required"] is True
    assert payload["workload"]["candidate_count"] == 2


# ---------------------------------------------------------------------------
# Parity key paths / collection mode
# ---------------------------------------------------------------------------


def test_parity_load_logits_key(tmp_path: Path) -> None:
    path = tmp_path / "reference.json"
    path.write_text(
        json.dumps({"headline_logits": {"p400": [1.0, 2.0]}}), encoding="utf-8"
    )
    values, _ = parity.load_logits(path, key="headline_logits.p400")
    assert values == pytest.approx([1.0, 2.0])


def _collection_payloads(
    delta: float,
) -> tuple[dict, dict]:
    reference = {
        "headline_logits": {"p400": [0.5, -0.5, 1.0]},
        "corpus": [
            {"id": "c0", "logits": [0.1, 0.2]},
            {"id": "c1", "logits": [-0.3, 0.4]},
        ],
        "real": [{"id": "r0", "logits": [1.0, 2.0, 3.0]}],
    }
    other = {
        "headline": {"p400": [value + delta for value in [0.5, -0.5, 1.0]]},
        "corpus": [
            {"id": "c0", "logits": [0.1 + delta, 0.2]},
            {"id": "c1", "logits": [-0.3, 0.4]},
        ],
        "real": [{"id": "r0", "logits": [1.0 + delta, 2.0, 3.0]}],
    }
    return reference, other


def test_compare_collections_pass_and_fail() -> None:
    reference, other = _collection_payloads(1e-6)
    passed = parity.compare_collections(reference, other, atol=2e-5, rank_fail=True)
    assert passed["passed"] is True
    assert passed["entries_total"] == 4  # headline + 2 corpus + 1 real
    reference, other = _collection_payloads(1e-2)
    failed = parity.compare_collections(reference, other, atol=2e-5, rank_fail=True)
    assert failed["passed"] is False
    assert failed["entries_passed"] == 1  # only c1 unchanged
    assert failed["worst"]["max_abs_diff"] >= 1e-2


def test_parity_collections_cli_exit_codes(tmp_path: Path) -> None:
    reference, other = _collection_payloads(1e-6)
    a_path = tmp_path / "a.json"
    b_path = tmp_path / "b.json"
    a_path.write_text(json.dumps(reference), encoding="utf-8")
    b_path.write_text(json.dumps(other), encoding="utf-8")
    report_path = tmp_path / "collection.json"
    ok = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_DIR / "parity.py"),
            "--compare-collections",
            "--a",
            str(a_path),
            "--b",
            str(b_path),
            "--atol",
            "2e-5",
            "--fail-on-rank-change",
            "--json-out",
            str(report_path),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert ok.returncode == 0, ok.stderr
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "parity-collection"
    assert payload["passed"] is True
    reference, other = _collection_payloads(1e-2)
    a_path.write_text(json.dumps(reference), encoding="utf-8")
    b_path.write_text(json.dumps(other), encoding="utf-8")
    bad = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_DIR / "parity.py"),
            "--compare-collections",
            "--a",
            str(a_path),
            "--b",
            str(b_path),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert bad.returncode == 1


# ---------------------------------------------------------------------------
# FINAL-RESULTS table rendering
# ---------------------------------------------------------------------------


def test_make_results_table_renders(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    def write_receipt(name: str, *, device: str, e2e: float, batching: bool) -> None:
        receipt = {
            "status": "ok",
            "environment": {"device": device},
            "scorer": {"device": device},
            "provenance": {"unstable_loader": False},
            "golden_check": {"passed": True},
            "self_consistency": {"passed": True},
            "stages": {
                "load": {"seconds": 2.5},
                "prefill_stage1": {"median_s": e2e * 0.5},
                "candidate_scoring_from_cache": {"median_s": e2e * 0.4},
                "end_to_end": {"median_s": e2e},
            },
            "memory": {"peak_rss_gib": 6.0},
            "protocol": {"warmups": 2, "repeats": 5},
        }
        (artifacts / name).write_text(json.dumps(receipt), encoding="utf-8")

    write_receipt("cpu-default.json", device="cpu", e2e=3.4, batching=False)
    write_receipt("cpu-batching.json", device="cpu", e2e=1.7, batching=True)
    write_receipt("mps-default.json", device="mps", e2e=1.5, batching=False)
    manifest = {
        "schema": "ouro-speed-final-manifest/v1",
        "loader_sha256_expected": "f" * 64,
        "arms": [
            {
                "arm": "cpu-default",
                "workload": "p400-k6-s16",
                "device": "cpu",
                "candidate_batching": False,
                "receipt": "cpu-default.json",
            },
            {
                "arm": "cpu-batching",
                "workload": "p400-k6-s16",
                "device": "cpu",
                "candidate_batching": True,
                "receipt": "cpu-batching.json",
            },
            {
                "arm": "mps-default",
                "workload": "p400-k6-s16",
                "device": "mps",
                "candidate_batching": False,
                "receipt": "mps-default.json",
            },
        ],
        "parity_reports": [],
        "notes": ["test"],
    }
    markdown, summary = make_results_table.build_table(manifest, artifacts)
    assert "| cpu-default | p400-k6-s16 | cpu | no |" in markdown
    assert "2.00x" in markdown  # 3.4 / 1.7 speedup for batching
    assert "frozen loader" in markdown
    assert "ffffffff" in markdown
    assert summary["arms"] == 3
    assert summary["passed_arms"] == 3


# ---------------------------------------------------------------------------
# Build script refuses overwrite (no model load, instant)
# ---------------------------------------------------------------------------


def test_build_refuses_to_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "existing-artifact"
    target.mkdir()
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_DIR / "build_local_artifact.py"),
            "--out",
            str(target),
            "--no-lock",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 2
    assert "refusing to overwrite" in (completed.stdout + completed.stderr)
