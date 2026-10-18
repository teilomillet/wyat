# Expert use cases

The [basic examples](../README.md) show one decision each. These show how
decisions combine into systems you can run in production: several typed
questions per input, policies that abstain, agents with guards, measured
thresholds, bias checks and bulk jobs.

```sh
uv run python examples/use_cases/expert/01_support_triage.py --demo
uv run python examples/use_cases/expert/01_support_triage.py --model teilomillet/wyat-1-800m
```

`--demo` uses a word-overlap stand-in instead of a model. It is **not** a
model; it only makes the numbers vary so that thresholds, fallbacks and loops
visibly do something. Use `--model teilomillet/wyat-1-800m` to see real
decisions: the first run downloads and verifies the pinned release, then reuses
the verified cache. A local model directory also works. On Apple Silicon, add
`--device mps`: CPU is several times slower there.

| # | System | What it teaches |
| --- | --- | --- |
| 01 | [Support triage](01_support_triage.py) | Choice + Score + Noul for every ticket in one `predict_many` call; abstain on a small margin; one audit line per decision |
| 02 | [Hierarchical routing](02_hierarchical_routing.py) | Large taxonomies as two short menus; a fallback per stage; batching per stage |
| 03 | [RAG gates](03_rag_gates.py) | Score to grade passages and select context within a budget; "I don't know" path; groundedness check before showing an answer |
| 04 | [Agent loop](04_agent_loop.py) | Tools as JSON schemas; the model picks the step, code enforces allowlist, approvals, no repeats and a step limit |
| 05 | [Calibrate and set a threshold](05_calibrate_threshold.py) | `evaluate`, temperature fitting with a check for a degenerate fit, accuracy versus automation with `report.selective`, checking on held-back cases |
| 06 | [Option-order check](06_order_robustness.py) | Rotate options, average by ID, treat a changing winner as unsure (wyat-1 800M is order-invariant; other scorers may not be) |
| 07 | [LLM judge](07_llm_judge.py) | Rubric scores per criterion; A/B with swapped positions |
| 08 | [Document check](08_document_check.py) | Questions generated from a requirements table; rules name exactly what is missing |
| 09 | [Bulk jobs](09_batch_throughput.py) | Dedupe, chunked `decide_many(..., errors="return")`, refused inputs recorded instead of failing the job |

## Patterns to reuse

- **The model measures, code decides.** Every example ends in plain `if`
  rules you can read, test and change without retraining.
- **Abstaining is a feature.** A small `probability_margin`, a changing winner
  or a low-confidence check sends the case to a person (01, 02, 06).
- **Ask everything about one input in one call.** `predict` takes one input
  and many questions, and mixes Choice, Score and Noul (01, 07, 08).
- **Thresholds come from data.** Choose them on labelled cases, then confirm
  on cases you did not use to choose them (05).
- **Log the numbers, not just the answer.** Probabilities, model ID and policy
  version let you replay and re-tune later (01, 05).
- **Guard effects.** Allowlists, approvals for irreversible actions, step
  limits and repeat detection live in code, not in the model (04).

## Observed on wyat-1 800M

From running every example on the release (CPU and MPS, 2026-10-18). These
are observations on a handful of cases, not quality claims; measure on your
own labelled cases before relying on any of them.

- 03: the question that asks two things at once ("how long does shipping take
  and what does it cost?") keeps no passage, so it answers "I don't know" and
  the groundedness gate never runs. Ask one thing per question.
- 05: the tune cases reach the target without abstaining. The one error is
  confident, so a larger margin does not remove it.
- 06: the probabilities are identical in every option order.
- 07: the A/B comparison is close to a tie, while the rubric scores separate
  the two versions clearly. Prefer per-criterion scores to "which is better".
- "None of these" options win close calls: in 02, "statement for my landlord"
  goes to a human rather than to `account`. Naming each option and using the
  customer's words helps (basic example 01).
