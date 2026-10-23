"""Ouro decision scoring on Apple GPU (MPS), FP32 arm: reproduce and extend.

Measured on the canonical MPS-policy artifact
``/tmp/ouro-speed/artifact-last2-mps-fp32`` (``device_policy="mps-fp32"``), which
materialises FP32 weights on the Apple GPU. Everything is compared against the
CPU FP32 reference written by ``ref_cpu.py`` for the same frozen inputs.

Deliverables in this receipt:

* headline stage timings (prefill / candidate-from-cache / readout / end-to-end,
  >=5 reps after one warmup) at P400/K6/S16, P832/K10/S64 and a heterogeneous
  P512 candidate set of lengths 8..24;
* MPS ``current_allocated``/``driver_allocated`` memory after each stage and the
  process peak RSS;
* where the time goes (prefill vs candidate vs scalar readout);
* a ``torch.profiler`` top-op table for the candidate stage;
* FP32-vs-CPU parity (declared atol 2e-5, rank + top-1 must be identical) for
  the headline shapes, the 32-question synthetic corpus and the 32 real keme ARC
  dev rows;
* the Lead's earlier P400 numbers/inputs reproduced under the current loader, so
  the receipt says explicitly whether the loader refactor changed anything.

Run:
    .venv/bin/python scripts/ouro_speed/mps/probe_fp32.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mps_common as common  # noqa: E402

sys.path.insert(0, str(common.REPO_ROOT))

from wyat.loading.ouro import load_ouro  # noqa: E402

# Lead's original MPS probe (old loader, cpu-fp32 artifact) for reproduction.
LEAD_MPS_REPORT = Path("/tmp/ouro-speed/mps-probe-report.json")

HEADLINE_ORDER = ("p400-k6-s16", "p832-k10-s64", "p512-k9-hetero-8to24")


def main() -> None:
    with common.bench_lock(tag="mps-probe-fp32"):
        loader_at_start = common.loader_fingerprint()
        question_set_path = common.question_set_path()
        if not question_set_path.is_file():
            raise SystemExit(f"missing {question_set_path}; run ref_cpu.py first")
        questions = common.load_question_set()
        reference = common.load_reference()
        question_sha = questions["sha256"]
        if reference.get("question_set_sha256") != question_sha:
            raise SystemExit("CPU reference was built for a different question set")

        caps = common.configure(
            device="mps",
            fraction=0.6,
            threads=6,
            rss_cap_gib=14.0,
            log_path=Path("/tmp/ouro-speed/logs/mps-fp32-watchdog.jsonl"),
        )

        started = time.perf_counter()
        scorer = load_ouro(common.MPS_ARTIFACT, device="mps")
        load_seconds = time.perf_counter() - started
        memory_after_load = common.mps_memory()
        print(
            f"[mps] loaded {common.MPS_ARTIFACT} in {load_seconds:.2f}s "
            f"device={scorer.device} mem={json.dumps(memory_after_load)}",
            flush=True,
        )

        receipt: dict = {
            "device": str(scorer.device),
            "backend": "mps",
            "loader_at_start": loader_at_start,
            "question_set_sha256": question_sha,
            "caps_configured": caps,
            "load_seconds": round(load_seconds, 3),
            "memory_after_load": memory_after_load,
            "headline": {},
            "corpus": {},
            "real": {},
        }

        with common.torch.inference_mode():
            # 1. Headline stage timings.
            for name in HEADLINE_ORDER:
                question = questions["headline"][name]
                print(f"[mps] headline stage breakdown: {name}", flush=True)
                breakdown = common.stage_breakdown(
                    scorer.backbone,
                    scorer.head,
                    question,
                    "mps",
                    repeats=5,
                    warmup=1,
                )
                scores = common.run_question(
                    scorer.backbone, scorer.head, question, "mps"
                )
                logits = common.logits_list(scores)
                parity = common.compare_logits(
                    logits, reference["headline_logits"][name]
                )
                breakdown["logits"] = logits
                breakdown["parity_vs_cpu_fp32"] = parity
                breakdown["parity_vs_cpu_fp32"]["atol"] = common.DECLARED_ATOL
                breakdown["memory"] = common.mps_memory()
                receipt["headline"][name] = breakdown
                print(
                    f"[mps] {name}: prefill={breakdown['prefill']['median_s']:.3f}s "
                    f"candidate={breakdown['candidate_from_cache']['median_s']:.3f}s "
                    f"e2e={breakdown['end_to_end']['median_s']:.3f}s "
                    f"max|d|={parity['max_abs_diff']:.3g} rank_equal="
                    f"{parity['rank_equal']}",
                    flush=True,
                )

            # 2. Reproduction of the Lead's earlier MPS numbers (P400 shape).
            lead_check = None
            if LEAD_MPS_REPORT.is_file():
                lead = json.loads(LEAD_MPS_REPORT.read_text(encoding="utf-8"))
                lead_arm = lead.get("arms", {}).get("mps_fp32", {})
                if lead_arm:
                    ours_logits = receipt["headline"]["p400-k6-s16"]["logits"]
                    logit_compare = common.compare_logits(
                        ours_logits, lead_arm.get("logits", [])
                    )
                    lead_check = {
                        "report": str(LEAD_MPS_REPORT),
                        "lead_loader_seconds": lead.get("load_seconds"),
                        "lead_stages": {
                            stage: lead_arm.get(stage, {}).get("median_s")
                            for stage in (
                                "prefill",
                                "candidate_from_cache",
                                "end_to_end",
                            )
                        },
                        "ours_stages": {
                            stage: receipt["headline"]["p400-k6-s16"][stage]["median_s"]
                            for stage in (
                                "prefill",
                                "candidate_from_cache",
                                "end_to_end",
                            )
                        },
                        "max_abs_logit_diff_vs_lead": logit_compare["max_abs_diff"],
                        "rank_equal_vs_lead": logit_compare["rank_equal"],
                        "inputs_reproduced": bool(
                            logit_compare["max_abs_diff"] <= 2e-5
                        ),
                    }
                    print(
                        f"[mps] lead reproduction check: {json.dumps(lead_check)}",
                        flush=True,
                    )
            receipt["lead_reproduction_check"] = lead_check

            # 3. Candidate-stage profiler table (P832/K10/S64).
            print("[mps] profiling candidate stage (p832-k10-s64)", flush=True)
            profile = common.profile_candidate_stage(
                scorer.backbone,
                scorer.head,
                questions["headline"]["p832-k10-s64"],
                "mps",
                repeats=2,
                row_limit=20,
            )
            receipt["profiler_candidate_stage"] = profile
            print(profile.get("table", profile.get("error", "")), flush=True)

            # 4. Corpus: serial logits + per-question CPU parity.
            corpus_records = []
            corpus_parity = []
            corpus_seconds = []
            corpus_started = time.perf_counter()
            for index, question in enumerate(questions["corpus"]):
                common.mps_sync()
                start = time.perf_counter()
                scores = common.run_question(
                    scorer.backbone, scorer.head, question, "mps"
                )
                common.mps_sync()
                elapsed = time.perf_counter() - start
                corpus_seconds.append(round(elapsed, 4))
                logits = common.logits_list(scores)
                expected = reference["corpus"][index]
                if expected["id"] != question["id"]:
                    raise SystemExit("corpus order mismatch with CPU reference")
                comparison = common.compare_logits(logits, expected["logits"])
                comparison["id"] = question["id"]
                corpus_parity.append(comparison)
                corpus_records.append({"id": question["id"], "logits": logits})
                if (index + 1) % 8 == 0:
                    print(
                        f"[mps] corpus {index + 1}/{len(questions['corpus'])} "
                        f"e2e={elapsed:.3f}s",
                        flush=True,
                    )
            receipt["corpus"] = {
                "parity_vs_cpu_fp32": common.parity_summary(corpus_parity),
                "per_question": corpus_parity,
                "e2e_seconds": corpus_seconds,
                "e2e_median_s": round(
                    sorted(corpus_seconds)[len(corpus_seconds) // 2], 4
                ),
                "total_seconds": round(time.perf_counter() - corpus_started, 3),
            }
            print(
                f"[mps] corpus parity: {json.dumps(receipt['corpus']['parity_vs_cpu_fp32'])}",
                flush=True,
            )

            # 5. Real keme ARC dev rows: serial logits + CPU parity + decisions.
            real_records = []
            real_parity = []
            if questions["real"] is not None:
                for index, row in enumerate(questions["real"]):
                    scores = common.run_question(
                        scorer.backbone, scorer.head, row, "mps"
                    )
                    logits = common.logits_list(scores)
                    expected = reference["real"][index]
                    comparison = common.compare_logits(logits, expected["logits"])
                    comparison["id"] = row["id"]
                    real_parity.append(comparison)
                    real_records.append(
                        {
                            "id": row["id"],
                            "gold_index": row["gold_index"],
                            "logits": logits,
                        }
                    )
                gold = [row["gold_index"] for row in questions["real"]]
                receipt["real"] = {
                    "parity_vs_cpu_fp32": common.parity_summary(real_parity),
                    "per_row": real_parity,
                    "decisions_vs_cpu": common.decision_flips(
                        [record["logits"] for record in real_records],
                        [record["logits"] for record in reference["real"]],
                        gold,
                    ),
                }
                print(
                    f"[mps] real parity: {json.dumps(receipt['real']['parity_vs_cpu_fp32'])} "
                    f"decisions={json.dumps(receipt['real']['decisions_vs_cpu'])}",
                    flush=True,
                )

        receipt["loader_guard"] = common.loader_run_guard(loader_at_start)
        receipt["memory_end"] = common.mps_memory()
        receipt["totals"] = {
            "total_seconds": round(time.perf_counter() - started, 3),
            "peak_rss_gib": common.peak_rss_gib(),
        }
        receipt["verdict"] = {
            "mps_fp32_parity_atol": common.DECLARED_ATOL,
            "headline_within_atol": all(
                item["parity_vs_cpu_fp32"]["max_abs_diff"] <= common.DECLARED_ATOL
                for item in receipt["headline"].values()
            ),
            "corpus_within_atol": receipt["corpus"]["parity_vs_cpu_fp32"][
                "within_atol"
            ],
            "real_within_atol": (
                receipt["real"].get("parity_vs_cpu_fp32", {}).get("within_atol")
                if receipt["real"]
                else None
            ),
        }
        common.write_receipt("fp32-extended", receipt)
        common.write_logits(
            "mps-fp32-serial",
            {
                "arm": "mps-fp32-serial",
                "headline": {
                    name: receipt["headline"][name]["logits"] for name in HEADLINE_ORDER
                },
                "corpus": corpus_records,
                "real": real_records,
                "parity": {
                    "headline": {
                        name: receipt["headline"][name]["parity_vs_cpu_fp32"]
                        for name in HEADLINE_ORDER
                    },
                    "corpus": receipt["corpus"]["parity_vs_cpu_fp32"],
                    "real": receipt["real"].get("parity_vs_cpu_fp32"),
                },
            },
        )
        common.emit(
            {
                "device": receipt["device"],
                "load_seconds": receipt["load_seconds"],
                "headline": {
                    name: {
                        "prefill_s": receipt["headline"][name]["prefill"]["median_s"],
                        "candidate_s": receipt["headline"][name][
                            "candidate_from_cache"
                        ]["median_s"],
                        "e2e_s": receipt["headline"][name]["end_to_end"]["median_s"],
                        "max_abs_diff_vs_cpu": receipt["headline"][name][
                            "parity_vs_cpu_fp32"
                        ]["max_abs_diff"],
                    }
                    for name in HEADLINE_ORDER
                },
                "corpus_median_e2e_s": receipt["corpus"]["e2e_median_s"],
                "verdict": receipt["verdict"],
                "peak_rss_gib": receipt["totals"]["peak_rss_gib"],
            }
        )


if __name__ == "__main__":
    main()
