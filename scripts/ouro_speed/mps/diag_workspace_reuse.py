"""Diagnostic: candidate K/V workspace reuse can corrupt multi-candidate logits.

Evidence request from the MPS probe: the CPU FP32 reference computed with the
current (uncommitted) ``wyat/loading/ouro.py`` workspace refactor disagreed with
the Lead's committed baseline receipt for candidates after the first one, while
the frozen synthetic inputs are byte-identical (``input_sha256`` match).

Suspected mechanism (``_WorkspacePool.acquire`` / ``_KVWorkspace.update``):
``_KVWorkspace`` reuses ONE ``[1, heads, context + candidate, head_dim]`` buffer
pair for all decoder layers of a step. Within a pass that is fine (each layer
copies its own context prefix before writing its suffix), but the pool keeps
``prepared`` at ``len(layers)`` when the workspace is reused by the next
candidate. That candidate then skips the context copy for every layer, so layer
0 reads the context K/V left behind by the LAST layer of the previous candidate.
The first candidate of a pool is correct; later candidates of equal length are
not.

This script proves it without touching runtime source: compare
  1. current loader, all candidates in one call (pool shared across candidates);
  2. current loader, one fresh call per candidate (fresh pool -> correct);
  3. monkeypatched pool resetting ``prepared`` on reuse (one-line fix probe);
against the Lead's committed ``baseline-cpu-p400-k6-s16.json`` logits for
byte-identical inputs.

Run:
    .venv/bin/python scripts/ouro_speed/mps/diag_workspace_reuse.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mps_common as common  # noqa: E402

sys.path.insert(0, str(common.REPO_ROOT))

from wyat.loading import ouro as ouro_module  # noqa: E402
from wyat.loading.ouro import (  # noqa: E402
    load_ouro,
    prefill_context,
    run_candidate_steps,
    score_candidates_from_cache,
)

BASELINE = Path("/tmp/ouro-speed/baseline-cpu-p400-k6-s16.json")


def synthetic_inputs(
    seed: int = 20261021, context: int = 400, count: int = 6, length: int = 16
):
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.randint(0, 49152, (1, context), generator=generator)
    candidates = [
        torch.randint(0, 49152, (1, length), generator=generator) for _ in range(count)
    ]
    digest = torch.cat([ids, *candidates], dim=1).numpy().tobytes()
    return ids, candidates, hashlib.sha256(digest).hexdigest()


def main() -> None:
    stable = common.await_loader_stable()
    loader_at_start = stable["loader"]
    with common.bench_lock(tag="mps-diag-workspace-reuse"):
        common.configure(device="cpu", threads=6, rss_cap_gib=14.0)
        import torch

        baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
        context, candidates, input_sha = synthetic_inputs()
        scorer = load_ouro(common.CPU_ARTIFACT, device="cpu")
        backbone, head = scorer.backbone, scorer.head
        result: dict = {
            "schema": "ouro-speed-mps-diagnostic/v1",
            "notice": common.PROFILING_NOTICE,
            "generated_at": common.now_iso(),
            "loader": common.loader_fingerprint(),
            "artifact": common.artifact_fingerprint(common.CPU_ARTIFACT),
            "inputs": {
                "input_sha256": input_sha,
                "baseline_input_sha256": baseline["input_sha256"],
                "inputs_match_lead_baseline": input_sha == baseline["input_sha256"],
            },
            "baseline_logits": baseline["logits"],
            "arms": {},
        }
        if input_sha != baseline["input_sha256"]:
            raise SystemExit("synthetic input reproduction failed; aborting")

        with torch.inference_mode():
            cache = prefill_context(backbone, context, policy="last", context_steps=4)

            def record(name: str, scores) -> None:
                logits = common.logits_list(scores)
                comparison = common.compare_logits(logits, baseline["logits"])
                result["arms"][name] = {
                    "logits": logits,
                    "max_abs_diff_vs_baseline": comparison["max_abs_diff"],
                    "mean_abs_diff_vs_baseline": comparison["mean_abs_diff"],
                    "rank_equal_vs_baseline": comparison["rank_equal"],
                    "per_candidate_abs_diff": [
                        abs(float(left) - float(right))
                        for left, right in zip(logits, baseline["logits"], strict=False)
                    ],
                }
                print(
                    f"[diag] {name}: max|d|={comparison['max_abs_diff']:.6g} "
                    f"rank_equal={comparison['rank_equal']} "
                    f"per_candidate={[round(v, 6) for v in result['arms'][name]['per_candidate_abs_diff']]}",
                    flush=True,
                )

            # 1. Current loader, one call for all candidates (shared pool).
            record(
                "current_loader_shared_pool",
                score_candidates_from_cache(backbone, head, cache, candidates),
            )

            # 2. Current loader, one fresh call per candidate (fresh pool each).
            per_candidate = torch.cat(
                [
                    score_candidates_from_cache(backbone, head, cache, [candidate])
                    for candidate in candidates
                ],
                dim=0,
            )
            record("current_loader_fresh_pool_per_candidate", per_candidate)

            # 3. Explicit run_candidate_steps per candidate (fresh geometry+pool).
            explicit = torch.cat(
                [
                    run_candidate_steps(backbone, cache, candidate)[:, -1]
                    for candidate in candidates
                ],
                dim=0,
            )
            record("explicit_run_candidate_steps", head(explicit))

            # 4. One-line fix probe: reset `prepared` whenever a pooled workspace
            #    is reused, so each candidate re-copies its context prefix.
            original_acquire = ouro_module._WorkspacePool.acquire

            def patched_acquire(self, bank, step, batch, candidate_length):
                workspace = original_acquire(self, bank, step, batch, candidate_length)
                workspace.prepared = 0
                return workspace

            ouro_module._WorkspacePool.acquire = patched_acquire  # type: ignore[method-assign]
            try:
                record(
                    "current_loader_patched_prepared_reset",
                    score_candidates_from_cache(backbone, head, cache, candidates),
                )
            finally:
                ouro_module._WorkspacePool.acquire = original_acquire  # type: ignore[method-assign]

        result["loader_guard"] = common.loader_run_guard(loader_at_start)
        result["verdict"] = {
            "current_shared_pool_matches_baseline": result["arms"][
                "current_loader_shared_pool"
            ]["max_abs_diff_vs_baseline"]
            <= 1e-6,
            "fresh_pool_matches_baseline": result["arms"][
                "current_loader_fresh_pool_per_candidate"
            ]["max_abs_diff_vs_baseline"]
            <= 1e-6,
            "patched_pool_matches_baseline": result["arms"][
                "current_loader_patched_prepared_reset"
            ]["max_abs_diff_vs_baseline"]
            <= 1e-6,
            "suspected_bug": (
                "_WorkspacePool.acquire resets workspace.count but not "
                "workspace.prepared; the second and later equal-length candidates "
                "skip the per-layer context copy and read the previous layer's "
                "context K/V. Fix: also reset workspace.prepared = 0 in acquire."
            ),
        }
        print(f"[diag] verdict: {json.dumps(result['verdict'], indent=2)}", flush=True)
        common.write_receipt("diagnostic-workspace-reuse", result)
        common.emit(result["verdict"])


if __name__ == "__main__":
    main()
