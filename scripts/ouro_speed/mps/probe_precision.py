"""Precision arms for Ouro decision scoring on Apple GPU (MPS).

Arms compared on identical frozen inputs against the CPU FP32 reference:

* ``fp32`` - the currently shipped MPS behaviour: FP32 materialised weights,
  FP32 activations, FP32 LoRA and FP32 scalar head;
* ``autocast_mps_bf16`` - FP32 weights, ``torch.autocast("mps", dtype=bfloat16)``
  around the recurrent passes (matmul autocast only; the loader already disables
  autocast inside the step-LoRA update and the scalar head);
* ``bf16_base_weights_fp32_lora`` - manual ``.to(bfloat16)`` of every non-LoRA
  backbone parameter (attention/MLP linears, norms, embeddings) while the
  step-LoRA factors, the scalar head and the rotary ``inv_freq`` stay FP32.

For each arm: headline prefill/candidate/readout/e2e medians, MPS current/driver
memory, drift vs CPU FP32 (max/mean |Δlogit|, rank/top-1 stability) over the
32-question synthetic corpus, and how many argmax decisions flip on the 32 real
keme ARC dev rows (read-only legacy ``prefix_ids``/``branches`` data).

This answers the loader question: which precision arm (if any) is safe to ship
for MPS, and what declared tolerance it would need.

Run:
    .venv/bin/python scripts/ouro_speed/mps/probe_precision.py
"""

from __future__ import annotations

import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mps_common as common  # noqa: E402

sys.path.insert(0, str(common.REPO_ROOT))

from wyat.loading.ouro import (  # noqa: E402
    StepLoRALinear,
    load_ouro,
    prefill_context,
    run_candidate_steps,
    score_candidates_from_cache,
)

ARMS = ("fp32", "autocast_mps_bf16", "bf16_base_weights_fp32_lora")
HEADLINE_ORDER = ("p400-k6-s16", "p832-k10-s64")
CORPUS_LIMIT = 32


def arm_context(arm: str):
    """Autocast context for the autocast arm; no-op for the other arms."""
    if arm == "autocast_mps_bf16":
        return common.torch.autocast(device_type="mps", dtype=common.torch.bfloat16)
    return nullcontext()


def cast_base_weights_to_bf16(backbone: Any) -> dict[str, Any]:
    """Cast every non-LoRA FP32 backbone parameter to BF16 (LoRA/head stay FP32)."""
    lora_ids = set()
    for module in backbone.modules():
        if isinstance(module, StepLoRALinear):
            for parameter in list(module.lora_A) + list(module.lora_B):
                lora_ids.add(id(parameter))
    cast = 0
    kept = 0
    for _name, parameter in backbone.named_parameters():
        if id(parameter) in lora_ids:
            kept += 1
            continue
        if parameter.dtype == common.torch.float32:
            parameter.data = parameter.data.to(common.torch.bfloat16)
            cast += 1
    dtype_counts: dict[str, int] = {}
    for _name, parameter in backbone.named_parameters():
        key = f"{parameter.dtype}".replace("torch.", "")
        dtype_counts[key] = dtype_counts.get(key, 0) + 1
    return {
        "params_cast_to_bf16": cast,
        "lora_params_kept_fp32": kept,
        "parameter_dtype_counts": dtype_counts,
        "rotary_inv_freq_dtype": str(backbone.model.rotary_emb.inv_freq.dtype).replace(
            "torch.", ""
        ),
    }


def arm_stage_timing(
    backbone: Any,
    head: Any,
    question: dict[str, Any],
    arm: str,
    *,
    repeats: int = 2,
    warmup: int = 1,
) -> dict[str, Any]:
    """Stage timings for one arm; every stage runs under the arm's autocast ctx."""
    context, candidates = common.to_device(question, "mps")

    def prefill():
        return prefill_context(backbone, context, policy="last", context_steps=4)

    def candidate_stage():
        cache = prefill()
        return score_candidates_from_cache(
            backbone, head, cache, candidates, candidate_steps=2
        )

    def readout():
        cache = prefill()
        hidden = common.torch.cat(
            [
                run_candidate_steps(backbone, cache, candidate)[:, -1]
                for candidate in candidates
            ],
            dim=0,
        )
        return head(hidden)

    def end_to_end():
        cache = prefill()
        return score_candidates_from_cache(
            backbone, head, cache, candidates, candidate_steps=2
        )

    with arm_context(arm):
        prefill_t = common.timed(prefill, repeats=repeats, warmup=warmup)
        cache = prefill()
        candidate_t = common.timed(
            lambda: score_candidates_from_cache(
                backbone, head, cache, candidates, candidate_steps=2
            ),
            repeats=repeats,
            warmup=warmup,
        )
        readout_t = common.timed(readout, repeats=repeats, warmup=warmup)
        e2e_t = common.timed(end_to_end, repeats=repeats, warmup=warmup)
    return {
        "context_tokens": int(context.shape[1]),
        "candidate_count": len(candidates),
        "candidate_lengths": [int(item.shape[1]) for item in candidates],
        "prefill": prefill_t,
        "candidate_from_cache": candidate_t,
        "readout_head_only": readout_t,
        "end_to_end": e2e_t,
        "memory": common.mps_memory(),
    }


def score_arm(
    backbone: Any, head: Any, question: dict[str, Any], arm: str
) -> list[float]:
    with arm_context(arm), common.torch.inference_mode():
        return common.logits_list(common.run_question(backbone, head, question, "mps"))


def main() -> None:
    stable = common.await_loader_stable()
    loader_at_start = stable["loader"]
    with common.bench_lock(tag="mps-probe-precision"):
        caps = common.configure(
            device="mps",
            fraction=0.6,
            threads=6,
            rss_cap_gib=14.0,
            log_path=Path("/tmp/ouro-speed/logs/mps-precision-watchdog.jsonl"),
        )
        questions = common.load_question_set()
        reference = common.load_reference()
        if reference.get("question_set_sha256") != questions["sha256"]:
            raise SystemExit("CPU reference was built for a different question set")

        started = time.perf_counter()
        scorer = load_ouro(common.MPS_ARTIFACT, device="mps")
        load_seconds = time.perf_counter() - started
        backbone, head = scorer.backbone, scorer.head
        print(
            f"[prec] loaded in {load_seconds:.2f}s mem={json.dumps(common.mps_memory())}",
            flush=True,
        )

        receipt: dict = {
            "device": str(scorer.device),
            "caps_configured": caps,
            "question_set_sha256": questions["sha256"],
            "load_seconds": round(load_seconds, 3),
            "memory_after_load": common.mps_memory(),
            "arms_requested": list(ARMS),
            "arms": {},
            "not_safe_to_ship": [],
        }

        for arm in ARMS:
            print(f"[prec] ===== arm {arm} =====", flush=True)
            arm_entry: dict = {"arm": arm}
            try:
                if arm == "bf16_base_weights_fp32_lora":
                    arm_entry["cast"] = cast_base_weights_to_bf16(backbone)
                    common.empty_mps_cache()
                    arm_entry["memory_after_cast"] = common.mps_memory()
                    print(
                        f"[prec] cast: {json.dumps(arm_entry['cast'])}",
                        flush=True,
                    )
                # Smoke test first so a broken arm fails fast with a clean error.
                smoke = score_arm(
                    backbone, head, questions["headline"]["p400-k6-s16"], arm
                )
                arm_entry["smoke_logits"] = smoke
                print(f"[prec] {arm} smoke logits={smoke}", flush=True)

                arm_entry["headline"] = {}
                for name in HEADLINE_ORDER:
                    question = questions["headline"][name]
                    timing = arm_stage_timing(backbone, head, question, arm)
                    logits = score_arm(backbone, head, question, arm)
                    timing["logits"] = logits
                    timing["parity_vs_cpu_fp32"] = common.compare_logits(
                        logits, reference["headline_logits"][name]
                    )
                    arm_entry["headline"][name] = timing
                    print(
                        f"[prec] {arm} {name}: prefill={timing['prefill']['median_s']:.3f}s "
                        f"candidate={timing['candidate_from_cache']['median_s']:.3f}s "
                        f"e2e={timing['end_to_end']['median_s']:.3f}s "
                        f"max|d cpu|={timing['parity_vs_cpu_fp32']['max_abs_diff']:.3g}",
                        flush=True,
                    )

                corpus_records = []
                corpus_compare = []
                corpus_seconds = []
                for index, question in enumerate(questions["corpus"][:CORPUS_LIMIT]):
                    common.mps_sync()
                    t0 = time.perf_counter()
                    logits = score_arm(backbone, head, question, arm)
                    common.mps_sync()
                    corpus_seconds.append(round(time.perf_counter() - t0, 4))
                    expected = reference["corpus"][index]["logits"]
                    comparison = common.compare_logits(logits, expected)
                    comparison["id"] = question["id"]
                    corpus_compare.append(comparison)
                    corpus_records.append({"id": question["id"], "logits": logits})
                arm_entry["corpus"] = {
                    "questions": len(corpus_compare),
                    "parity_vs_cpu_fp32": common.parity_summary(corpus_compare),
                    "per_question": corpus_compare,
                    "e2e_seconds": corpus_seconds,
                    "e2e_median_s": round(
                        sorted(corpus_seconds)[len(corpus_seconds) // 2], 4
                    ),
                }
                print(
                    f"[prec] {arm} corpus parity: {json.dumps(arm_entry['corpus']['parity_vs_cpu_fp32'])}",
                    flush=True,
                )

                real_records = []
                if questions["real"] is not None:
                    for row in questions["real"]:
                        real_records.append(
                            {
                                "id": row["id"],
                                "gold_index": row["gold_index"],
                                "logits": score_arm(backbone, head, row, arm),
                            }
                        )
                    reference_logits = [item["logits"] for item in reference["real"]]
                    gold = [row["gold_index"] for row in questions["real"]]
                    arm_entry["real"] = {
                        "rows": len(real_records),
                        "decisions_vs_cpu": common.decision_flips(
                            [item["logits"] for item in real_records],
                            reference_logits,
                            gold,
                        ),
                        "per_row": [
                            {
                                "id": item["id"],
                                "gold_index": item["gold_index"],
                                "logits": item["logits"],
                            }
                            for item in real_records
                        ],
                    }
                    print(
                        f"[prec] {arm} real decisions: "
                        f"{json.dumps(arm_entry['real']['decisions_vs_cpu'])}",
                        flush=True,
                    )
                arm_entry["memory_end"] = common.mps_memory()
                arm_entry["status"] = "ok"
                arm_entry["_corpus_records"] = corpus_records
                arm_entry["_real_records"] = real_records
            except Exception as error:  # noqa: BLE001 - arms are probes
                arm_entry["status"] = "error"
                arm_entry["error"] = f"{type(error).__name__}: {error}"
                print(f"[prec] {arm} FAILED: {arm_entry['error']}", flush=True)
            receipt["arms"][arm] = arm_entry

        # Strip internal replay payloads from the receipt into their own files.
        arm_logits: dict = {}
        for arm in ARMS:
            entry = receipt["arms"][arm]
            arm_logits[arm] = {
                "status": entry.get("status"),
                "headline": {
                    name: entry["headline"][name]["logits"]
                    for name in HEADLINE_ORDER
                    if name in entry.get("headline", {})
                },
                "corpus": entry.pop("_corpus_records", []),
                "real": entry.pop("_real_records", []),
            }

        # Verdicts derived from the numbers above.
        fp32_arm = receipt["arms"]["fp32"]
        fp32_corpus_ok = (
            fp32_arm.get("status") == "ok"
            and fp32_arm["corpus"]["parity_vs_cpu_fp32"]["within_atol"]
            and fp32_arm["corpus"]["parity_vs_cpu_fp32"]["rank_stable"]
            and fp32_arm["corpus"]["parity_vs_cpu_fp32"]["top1_stable"]
        )
        receipt["verdict"] = {
            "fp32_parity_ok": bool(fp32_corpus_ok),
            "declared_atol": common.DECLARED_ATOL,
            "arms": {
                arm: {
                    "status": receipt["arms"][arm].get("status"),
                    "headline_max_abs_diff_vs_cpu": max(
                        (
                            receipt["arms"][arm]["headline"][name][
                                "parity_vs_cpu_fp32"
                            ]["max_abs_diff"]
                            for name in HEADLINE_ORDER
                            if name in receipt["arms"][arm].get("headline", {})
                        ),
                        default=None,
                    ),
                    "corpus_within_atol": (
                        receipt["arms"][arm]
                        .get("corpus", {})
                        .get("parity_vs_cpu_fp32", {})
                        .get("within_atol")
                    ),
                    "corpus_rank_stable": (
                        receipt["arms"][arm]
                        .get("corpus", {})
                        .get("parity_vs_cpu_fp32", {})
                        .get("rank_stable")
                    ),
                    "corpus_top1_stable": (
                        receipt["arms"][arm]
                        .get("corpus", {})
                        .get("parity_vs_cpu_fp32", {})
                        .get("top1_stable")
                    ),
                    "real_decision_flips": (
                        receipt["arms"][arm]
                        .get("real", {})
                        .get("decisions_vs_cpu", {})
                        .get("decision_flips")
                    ),
                    "e2e_p400_s": (
                        receipt["arms"][arm]
                        .get("headline", {})
                        .get("p400-k6-s16", {})
                        .get("end_to_end", {})
                        .get("median_s")
                    ),
                    "e2e_p832_s": (
                        receipt["arms"][arm]
                        .get("headline", {})
                        .get("p832-k10-s64", {})
                        .get("end_to_end", {})
                        .get("median_s")
                    ),
                }
                for arm in ARMS
            },
        }
        receipt["memory_end"] = common.mps_memory()
        receipt["loader_guard"] = common.loader_run_guard(loader_at_start)
        receipt["totals"] = {
            "total_seconds": round(time.perf_counter() - started, 3),
            "peak_rss_gib": common.peak_rss_gib(),
        }
        common.write_receipt("precision", receipt)
        common.write_logits(
            "mps-precision", {"arm": "mps-precision", "arms": arm_logits}
        )
        common.emit(receipt["verdict"])


if __name__ == "__main__":
    main()
