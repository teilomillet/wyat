"""Candidate batching on Apple GPU (MPS): grouped loader fast path vs padded prototype.

Three strategies are compared on identical inputs and one frozen MPS FP32 model:

1. ``serial`` - the production default: candidates scored one at a time;
2. ``grouped`` - the loader's opt-in ``candidate_batching=True`` fast path, which
   groups candidates by exact token length (no padding, exact positions);
3. ``padded`` - this repo-local prototype (``mps_batch.py``): pad all candidates
   of a question to ``S_max`` and run ONE batched forward per recurrent step with
   an exact additive causal mask. Chosen for heterogeneous candidate lengths,
   where ``grouped`` degenerates to K single-row passes.

Measured for every question: wall time, max/mean logit drift against the serial
arm and against the CPU FP32 reference, rank equality, top-1 equality, and (on
the 32 real keme ARC dev rows) decision flips. Headline shapes additionally get
5-rep medians and MPS current/driver memory.

Verdict question: does the MPS speedup justify a tolerance-gated fast path, and
which variant should the loader expose for Apple GPU?

Run:
    .venv/bin/python scripts/ouro_speed/mps/probe_batching.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mps_common as common  # noqa: E402

sys.path.insert(0, str(common.REPO_ROOT))

import mps_batch  # noqa: E402

from wyat.loading.ouro import (  # noqa: E402
    load_ouro,
    prefill_context,
    score_candidates_from_cache,
)

HEADLINE_ORDER = ("p400-k6-s16", "p832-k10-s64", "p512-k9-hetero-8to24")
STRATEGIES = ("serial", "grouped", "padded")


def score_serial(backbone, head, cache, candidates):
    return score_candidates_from_cache(
        backbone, head, cache, candidates, candidate_steps=2
    )


def score_grouped(backbone, head, cache, candidates):
    return score_candidates_from_cache(
        backbone,
        head,
        cache,
        candidates,
        candidate_steps=2,
        candidate_batching=True,
    )


def score_padded(backbone, head, cache, candidates):
    return mps_batch.score_candidates_batched_padded(
        backbone, head, cache, candidates, candidate_steps=2
    )


STRATEGY_FNS = {
    "serial": score_serial,
    "grouped": score_grouped,
    "padded": score_padded,
}


def timed_strategy(fn, repeats: int, warmup: int = 1) -> dict:
    return common.timed(fn, repeats=repeats, warmup=warmup, device="mps")


def one_shot(fn) -> tuple[list[float], float]:
    common.mps_sync()
    start = time.perf_counter()
    scores = fn()
    common.mps_sync()
    elapsed = time.perf_counter() - start
    return common.logits_list(scores), round(elapsed, 4)


def main() -> None:
    stable = common.await_loader_stable()
    loader_at_start = stable["loader"]
    with common.bench_lock(tag="mps-probe-batching"):
        caps = common.configure(
            device="mps",
            fraction=0.6,
            threads=6,
            rss_cap_gib=14.0,
            log_path=Path("/tmp/ouro-speed/logs/mps-batching-watchdog.jsonl"),
        )
        questions = common.load_question_set()
        reference = common.load_reference()
        if reference.get("question_set_sha256") != questions["sha256"]:
            raise SystemExit("CPU reference was built for a different question set")

        started = time.perf_counter()
        scorer = load_ouro(common.MPS_ARTIFACT, device="mps")
        load_seconds = time.perf_counter() - started
        print(
            f"[batch] loaded in {load_seconds:.2f}s device={scorer.device} "
            f"mem={json.dumps(common.mps_memory())}",
            flush=True,
        )
        backbone, head = scorer.backbone, scorer.head

        receipt: dict = {
            "device": str(scorer.device),
            "loader_guard_placeholder": None,
            "caps_configured": caps,
            "question_set_sha256": questions["sha256"],
            "load_seconds": round(load_seconds, 3),
            "memory_after_load": common.mps_memory(),
            "strategies": list(STRATEGIES),
            "headline": {},
            "corpus": {},
            "real": {},
        }

        with common.torch.inference_mode():
            # 1. Headline shapes: 5-rep medians + drift + memory per strategy.
            for name in HEADLINE_ORDER:
                question = questions["headline"][name]
                context, candidates = common.to_device(question, "mps")
                cache = prefill_context(
                    backbone, context, policy="last", context_steps=4
                )
                padded_length = max(int(item.shape[1]) for item in candidates)
                workspace_bytes = mps_batch.padded_workspace_bytes(
                    backbone,
                    int(context.shape[1]),
                    len(candidates),
                    padded_length,
                )
                entry: dict = {
                    "context_tokens": int(context.shape[1]),
                    "candidate_count": len(candidates),
                    "candidate_lengths": [int(item.shape[1]) for item in candidates],
                    "padded_workspace_bytes": workspace_bytes,
                    "arms": {},
                }
                serial_logits = None
                for strategy in STRATEGIES:

                    def fn(fn=STRATEGY_FNS[strategy]):
                        return fn(  # noqa: B023
                            backbone, head, cache, candidates
                        )

                    timing = timed_strategy(fn, repeats=5, warmup=1)
                    logits, _ = one_shot(fn)
                    arm = {
                        "timing": timing,
                        "logits": logits,
                        "memory": common.mps_memory(),
                    }
                    if strategy == "serial":
                        serial_logits = logits
                    else:
                        arm["drift_vs_serial"] = common.compare_logits(
                            logits, serial_logits
                        )
                    arm["drift_vs_cpu_fp32"] = common.compare_logits(
                        logits, reference["headline_logits"][name]
                    )
                    entry["arms"][strategy] = arm
                    print(
                        f"[batch] {name} {strategy}: "
                        f"candidate={timing['median_s']:.4f}s "
                        f"max|d serial|={(0.0 if strategy == 'serial' else arm['drift_vs_serial']['max_abs_diff']):.3g} "
                        f"max|d cpu|={arm['drift_vs_cpu_fp32']['max_abs_diff']:.3g} "
                        f"rank={arm['drift_vs_cpu_fp32']['rank_equal']}",
                        flush=True,
                    )
                serial_median = entry["arms"]["serial"]["timing"]["median_s"]
                for strategy in STRATEGIES[1:]:
                    entry["arms"][strategy]["speedup_vs_serial"] = round(
                        serial_median / entry["arms"][strategy]["timing"]["median_s"],
                        3,
                    )
                receipt["headline"][name] = entry
                common.empty_mps_cache()

            # 2. Corpus: 32 synthetic questions with varied P/K/S.
            corpus_records = {strategy: [] for strategy in STRATEGIES}
            corpus_totals = {strategy: 0.0 for strategy in STRATEGIES}
            corpus_rows = []
            for index, question in enumerate(questions["corpus"]):
                context, candidates = common.to_device(question, "mps")
                cache = prefill_context(
                    backbone, context, policy="last", context_steps=4
                )
                lengths = [int(item.shape[1]) for item in candidates]
                row: dict = {
                    "id": question["id"],
                    "context_tokens": int(context.shape[1]),
                    "candidate_count": len(candidates),
                    "candidate_lengths": lengths,
                    "heterogeneous": len(set(lengths)) > 1,
                    "timing_s": {},
                    "drift_vs_serial": {},
                    "drift_vs_cpu_fp32": {},
                }
                serial_logits = None
                expected = reference["corpus"][index]["logits"]
                for strategy in STRATEGIES:
                    logits, elapsed = one_shot(
                        lambda fn=STRATEGY_FNS[strategy]: fn(  # noqa: B023
                            backbone, head, cache, candidates
                        )
                    )
                    corpus_totals[strategy] += elapsed
                    corpus_records[strategy].append(
                        {"id": question["id"], "logits": logits}
                    )
                    row["timing_s"][strategy] = round(elapsed, 4)
                    if strategy == "serial":
                        serial_logits = logits
                    else:
                        row["drift_vs_serial"][strategy] = common.compare_logits(
                            logits, serial_logits
                        )
                    row["drift_vs_cpu_fp32"][strategy] = common.compare_logits(
                        logits, expected
                    )
                corpus_rows.append(row)
                if (index + 1) % 8 == 0:
                    print(
                        f"[batch] corpus {index + 1}/{len(questions['corpus'])} "
                        f"serial={corpus_totals['serial']:.2f}s "
                        f"grouped={corpus_totals['grouped']:.2f}s "
                        f"padded={corpus_totals['padded']:.2f}s",
                        flush=True,
                    )

            hetero_rows = [row for row in corpus_rows if row["heterogeneous"]]
            uniform_rows = [row for row in corpus_rows if not row["heterogeneous"]]

            def summarize(rows, strategy):
                times = [row["timing_s"][strategy] for row in rows]
                serial = [row["timing_s"]["serial"] for row in rows]
                return {
                    "questions": len(rows),
                    "total_s": round(sum(times), 3),
                    "serial_total_s": round(sum(serial), 3),
                    "speedup_vs_serial": (
                        round(sum(serial) / sum(times), 3) if sum(times) else None
                    ),
                    "median_question_s": (
                        round(statistics.median(times), 4) if times else None
                    ),
                }

            def summarize_drift(rows, strategy):
                comparisons = [row["drift_vs_cpu_fp32"][strategy] for row in rows]
                vs_serial = [
                    row["drift_vs_serial"][strategy]
                    for row in rows
                    if strategy in row["drift_vs_serial"]
                ]
                return {
                    "vs_cpu_fp32": common.parity_summary(comparisons),
                    "vs_serial": (
                        common.parity_summary(vs_serial) if vs_serial else None
                    ),
                }

            receipt["corpus"] = {
                "questions": len(corpus_rows),
                "heterogeneous_questions": len(hetero_rows),
                "totals": {
                    strategy: summarize(corpus_rows, strategy)
                    for strategy in STRATEGIES
                },
                "heterogeneous_totals": {
                    strategy: summarize(hetero_rows, strategy)
                    for strategy in STRATEGIES
                },
                "uniform_totals": {
                    strategy: summarize(uniform_rows, strategy)
                    for strategy in STRATEGIES
                },
                "drift": {
                    strategy: summarize_drift(corpus_rows, strategy)
                    for strategy in STRATEGIES
                },
                "per_question": corpus_rows,
            }
            print(
                f"[batch] corpus totals: {json.dumps({s: receipt['corpus']['totals'][s] for s in STRATEGIES})}",
                flush=True,
            )
            print(
                f"[batch] corpus hetero totals: {json.dumps({s: receipt['corpus']['heterogeneous_totals'][s] for s in STRATEGIES})}",
                flush=True,
            )

            # 3. Real keme ARC dev rows: decisions + flips vs CPU FP32 reference.
            real_records = {strategy: [] for strategy in STRATEGIES}
            real_rows = []
            if questions["real"] is not None:
                for index, row in enumerate(questions["real"]):
                    context, candidates = common.to_device(row, "mps")
                    cache = prefill_context(
                        backbone, context, policy="last", context_steps=4
                    )
                    expected = reference["real"][index]["logits"]
                    lengths = [int(item.shape[1]) for item in candidates]
                    entry: dict = {
                        "id": row["id"],
                        "gold_index": row["gold_index"],
                        "candidate_lengths": lengths,
                        "timing_s": {},
                        "drift_vs_cpu_fp32": {},
                    }
                    for strategy in STRATEGIES:
                        logits, elapsed = one_shot(
                            lambda fn=STRATEGY_FNS[strategy]: fn(  # noqa: B023
                                backbone, head, cache, candidates
                            )
                        )
                        entry["timing_s"][strategy] = round(elapsed, 4)
                        entry["drift_vs_cpu_fp32"][strategy] = common.compare_logits(
                            logits, expected
                        )
                        real_records[strategy].append(
                            {"id": row["id"], "logits": logits}
                        )
                    real_rows.append(entry)
                gold = [row["gold_index"] for row in questions["real"]]
                reference_logits = [item["logits"] for item in reference["real"]]
                receipt["real"] = {
                    "rows": len(real_rows),
                    "totals": {
                        strategy: {
                            "total_s": round(
                                sum(item["timing_s"][strategy] for item in real_rows), 4
                            ),
                            "median_row_s": round(
                                statistics.median(
                                    item["timing_s"][strategy] for item in real_rows
                                ),
                                4,
                            ),
                        }
                        for strategy in STRATEGIES
                    },
                    "drift": {
                        strategy: common.parity_summary(
                            [item["drift_vs_cpu_fp32"][strategy] for item in real_rows]
                        )
                        for strategy in STRATEGIES
                    },
                    "decisions_vs_cpu": {
                        strategy: common.decision_flips(
                            [item["logits"] for item in real_records[strategy]],
                            reference_logits,
                            gold,
                        )
                        for strategy in STRATEGIES
                    },
                    "per_row": real_rows,
                }
                print(
                    f"[batch] real decisions: {json.dumps(receipt['real']['decisions_vs_cpu'])}",
                    flush=True,
                )

        common.empty_mps_cache()
        receipt["memory_end"] = common.mps_memory()
        receipt["loader_guard"] = common.loader_run_guard(loader_at_start)
        receipt["totals"] = {
            "total_seconds": round(time.perf_counter() - started, 3),
            "peak_rss_gib": common.peak_rss_gib(),
        }

        corpus_totals_final = receipt["corpus"]["totals"]
        speedups = {
            strategy: corpus_totals_final[strategy]["speedup_vs_serial"]
            for strategy in STRATEGIES[1:]
        }
        hetero_speedups = {
            strategy: receipt["corpus"]["heterogeneous_totals"][strategy][
                "speedup_vs_serial"
            ]
            for strategy in STRATEGIES[1:]
        }
        drift_ok = all(
            receipt["corpus"]["drift"][strategy]["vs_cpu_fp32"]["within_atol"]
            and receipt["corpus"]["drift"][strategy]["vs_cpu_fp32"]["rank_stable"]
            and receipt["corpus"]["drift"][strategy]["vs_cpu_fp32"]["top1_stable"]
            for strategy in ("grouped", "padded")
        )
        real_flips = receipt.get("real", {}).get("decisions_vs_cpu", {})
        receipt["verdict"] = {
            "corpus_speedup_vs_serial": speedups,
            "heterogeneous_speedup_vs_serial": hetero_speedups,
            "drift_within_atol_and_rank_top1_stable": drift_ok,
            "declared_atol": common.DECLARED_ATOL,
            "real_decision_flips_vs_cpu": {
                strategy: real_flips.get(strategy, {}).get("decision_flips")
                for strategy in ("grouped", "padded", "serial")
            },
            "recommended_for_mps": None,
            "note": (
                "recommendation is assembled from the numbers above: grouped is "
                "the loader's existing opt-in fast path; padded is only worth it "
                "when candidate lengths are heterogeneous and the grouped pass "
                "would degenerate to single-row batches"
            ),
        }
        common.write_receipt("batching", receipt)
        common.write_logits(
            "mps-batching",
            {
                "arm": "mps-batching",
                "headline": {
                    name: {
                        strategy: receipt["headline"][name]["arms"][strategy]["logits"]
                        for strategy in STRATEGIES
                    }
                    for name in HEADLINE_ORDER
                },
                "corpus": corpus_records,
                "real": real_records,
            },
        )
        common.emit(
            {
                "corpus_speedups": speedups,
                "hetero_speedups": hetero_speedups,
                "drift_ok": drift_ok,
                "headline": {
                    name: {
                        strategy: receipt["headline"][name]["arms"][strategy]["timing"][
                            "median_s"
                        ]
                        for strategy in STRATEGIES
                    }
                    for name in HEADLINE_ORDER
                },
            }
        )


if __name__ == "__main__":
    main()
