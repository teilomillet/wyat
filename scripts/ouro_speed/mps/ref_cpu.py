"""CPU FP32 reference arm + frozen question set for the MPS probes.

Writes two receipts consumed by every MPS probe:

* ``artifacts/ouro-speed-20261021/mps/inputs/question-set.json``
  the exact token ids for the headline shapes, the 32-question synthetic corpus
  and the 32 real keme ARC dev rows (legacy ``prefix_ids``/``branches`` schema,
  mapped read-only to canonical shared context + candidate continuations);
* ``artifacts/ouro-speed-20261021/mps/reference-cpu-fp32.json``
  CPU FP32 logits for every question using the ``cpu-fp32`` artifact, plus a
  single-pass CPU timing at the headline shapes (comparison against the Lead's
  earlier baseline) and the load/cold-start measurement.

Only the CPU artifact is loaded here; no MPS allocator is touched. The process
still follows the shared caps: one heavy process at a time (bench lock),
``nice`` 10, <=6 torch threads, RSS watchdog.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mps_common as common  # noqa: E402  (sets MPS env vars; imports torch)

sys.path.insert(0, str(common.REPO_ROOT))

from wyat.loading.ouro import (  # noqa: E402
    load_ouro,
    prefill_context,
    score_candidates_from_cache,
)

LEAD_BASELINE = Path("/tmp/ouro-speed/baseline-cpu-p400-k6-s16.json")


def cpu_stage_timing(backbone, head, question: dict, *, repeats: int = 1) -> dict:
    """Single-pass CPU timing at one shape (comparison against Lead baseline)."""
    context, candidates = common.to_device(question, "cpu")

    def prefill_once():
        return prefill_context(backbone, context, policy="last", context_steps=4)

    prefill_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        prefill_once()
        prefill_times.append(time.perf_counter() - start)
    cache = prefill_once()

    candidate_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        score_candidates_from_cache(backbone, head, cache, candidates)
        candidate_times.append(time.perf_counter() - start)

    e2e_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fresh = prefill_context(backbone, context, policy="last", context_steps=4)
        score_candidates_from_cache(backbone, head, fresh, candidates)
        e2e_times.append(time.perf_counter() - start)

    def summarize(samples):
        samples = sorted(samples)
        return {
            "median_s": round(samples[len(samples) // 2], 4),
            "samples_s": [round(value, 4) for value in samples],
            "repeats": repeats,
        }

    return {
        "context_tokens": int(context.shape[1]),
        "candidate_count": len(candidates),
        "candidate_lengths": [int(item.shape[1]) for item in candidates],
        "threads": 6,
        "prefill": summarize(prefill_times),
        "candidate_scoring_from_cache": summarize(candidate_times),
        "end_to_end": summarize(e2e_times),
    }


def main() -> None:
    stable = common.await_loader_stable()
    loader_at_start = stable["loader"]
    with common.bench_lock(tag="mps-ref-cpu-fp32"):
        common.configure(device="cpu", threads=6, rss_cap_gib=14.0)
        questions = common.question_set()
        if questions["real"] is None:
            print(
                "[ref] keme dev-tokenized.json not reachable; real rows omitted",
                flush=True,
            )
        inputs_path = common.write_question_set(questions)
        question_sha = json.loads(inputs_path.read_text(encoding="utf-8"))["sha256"]
        print(f"[ref] wrote {inputs_path} sha256={question_sha}", flush=True)

        started = time.perf_counter()
        scorer = load_ouro(common.CPU_ARTIFACT, device="cpu")
        load_seconds = time.perf_counter() - started
        print(
            f"[ref] loaded cpu-fp32 artifact in {load_seconds:.2f}s "
            f"peak_rss={common.peak_rss_gib():.2f} GiB",
            flush=True,
        )

        reference: dict = {
            "schema": "ouro-speed-mps-cpu-reference/v1",
            "notice": common.PROFILING_NOTICE,
            "generated_at": common.now_iso(),
            "question_set_sha256": question_sha,
            "artifact": common.artifact_fingerprint(common.CPU_ARTIFACT),
            "loader": common.loader_fingerprint(),
            "host": common.host_fingerprint(),
            "cpu_caps": {
                "threads": 6,
                "niceness": 10,
                "device": "cpu",
            },
            "load_seconds": round(load_seconds, 3),
            "load_memory": {"peak_rss_gib": common.peak_rss_gib()},
            "headline_timing": {},
            "headline_logits": {},
            "corpus": [],
            "real": [],
        }

        with common.torch.inference_mode():
            check = None
            for name, question in questions["headline"].items():
                timing = cpu_stage_timing(scorer.backbone, scorer.head, question)
                scores = common.run_question(
                    scorer.backbone, scorer.head, question, "cpu"
                )
                logits = common.logits_list(scores)
                reference["headline_timing"][name] = timing
                reference["headline_logits"][name] = logits
                print(
                    f"[ref] headline {name}: e2e={timing['end_to_end']['median_s']:.3f}s "
                    f"logits={[round(value, 5) for value in logits]}",
                    flush=True,
                )
                if name == "p400-k6-s16" and LEAD_BASELINE.is_file():
                    baseline = json.loads(LEAD_BASELINE.read_text(encoding="utf-8"))
                    comparison = common.compare_logits(logits, baseline["logits"])
                    check = {
                        "lead_baseline": str(LEAD_BASELINE),
                        "max_abs_diff_vs_lead": comparison["max_abs_diff"],
                        "rank_equal_vs_lead": comparison["rank_equal"],
                        "reproduced_lead_inputs": bool(
                            comparison["max_abs_diff"] <= 1e-6
                        ),
                    }
                    print(f"[ref] lead baseline check: {json.dumps(check)}", flush=True)
            reference["lead_baseline_check"] = check

            corpus_times = []
            corpus_started = time.perf_counter()
            for index, question in enumerate(questions["corpus"]):
                start = time.perf_counter()
                scores = common.run_question(
                    scorer.backbone, scorer.head, question, "cpu"
                )
                elapsed = time.perf_counter() - start
                corpus_times.append(round(elapsed, 4))
                reference["corpus"].append(
                    {
                        "id": question["id"],
                        "context_tokens": len(question["context"]),
                        "candidate_lengths": [
                            len(item) for item in question["candidates"]
                        ],
                        "logits": common.logits_list(scores),
                        "e2e_s": round(elapsed, 4),
                    }
                )
                print(
                    f"[ref] corpus {index + 1}/{len(questions['corpus'])} "
                    f"{question['id']} e2e={elapsed:.3f}s",
                    flush=True,
                )
            reference["corpus_total_seconds"] = round(
                time.perf_counter() - corpus_started, 3
            )
            reference["corpus_e2e_seconds"] = corpus_times
            reference["corpus_e2e_median_s"] = round(
                sorted(corpus_times)[len(corpus_times) // 2], 4
            )

            if questions["real"] is not None:
                for index, row in enumerate(questions["real"]):
                    scores = common.run_question(
                        scorer.backbone, scorer.head, row, "cpu"
                    )
                    reference["real"].append(
                        {
                            "id": row["id"],
                            "suite": row["suite"],
                            "gold_index": row["gold_index"],
                            "context_tokens": len(row["context"]),
                            "candidate_lengths": [
                                len(item) for item in row["candidates"]
                            ],
                            "logits": common.logits_list(scores),
                        }
                    )
                    print(
                        f"[ref] real {index + 1}/{len(questions['real'])} {row['id']}",
                        flush=True,
                    )

        reference["loader_guard"] = common.loader_run_guard(loader_at_start)
        reference["total_seconds"] = round(time.perf_counter() - started, 3)
        reference["peak_rss_gib"] = common.peak_rss_gib()
        common.RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
        path = common.reference_path()
        path.write_text(json.dumps(reference, indent=1) + "\n", encoding="utf-8")
        print(f"[ref] wrote {path}", flush=True)
        common.write_receipt(
            "reference-cpu-fp32-run",
            {
                "question_set_sha256": question_sha,
                "question_set_path": str(inputs_path),
                "reference_path": str(path),
                "load_seconds": round(load_seconds, 3),
                "headline_timing": reference["headline_timing"],
                "headline_logits": reference["headline_logits"],
                "corpus_questions": len(reference["corpus"]),
                "real_questions": len(reference["real"]),
                "corpus_total_seconds": reference["corpus_total_seconds"],
                "total_seconds": reference["total_seconds"],
                "lead_baseline_check": check,
                "peak_rss_gib": common.peak_rss_gib(),
            },
        )


if __name__ == "__main__":
    main()
