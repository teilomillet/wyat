# Expert use cases

The [basic examples](../README.md) show one decision each. These show how
decisions combine into systems you can run in production: several typed
questions per input, policies that abstain, agents with guards, measured
thresholds, bias checks and bulk jobs.

```sh
uv run python examples/use_cases/expert/01_support_triage.py --demo
uv run python examples/use_cases/expert/01_support_triage.py --model teilomillet/wyat-1-600m
```

`--demo` uses a word-overlap stand-in instead of a model. It is **not** a
model; it only makes the numbers vary so that thresholds, fallbacks and loops
visibly do something. Use `--model` to see real decisions.

| # | System | What it teaches |
| --- | --- | --- |
| 01 | [Support triage](01_support_triage.py) | Choice + Score + Noul for every ticket in one `predict_many` call; abstain on a small margin; one audit line per decision |
| 02 | [Hierarchical routing](02_hierarchical_routing.py) | Large taxonomies as two short menus; a fallback per stage; batching per stage |
| 03 | [RAG gates](03_rag_gates.py) | Noul to select context within a budget; "I don't know" path; groundedness check before showing an answer |
| 04 | [Agent loop](04_agent_loop.py) | Tools as JSON schemas; the model picks the step, code enforces allowlist, approvals, no repeats and a step limit |
| 05 | [Calibrate and set a threshold](05_calibrate_threshold.py) | `evaluate`, temperature fitting, accuracy versus automation with `report.selective`, checking on held-back cases |
| 06 | [Option-order check](06_order_robustness.py) | Rotate options, average by ID, treat a changing winner as unsure |
| 07 | [LLM judge](07_llm_judge.py) | Rubric scores per criterion; A/B with swapped positions |
| 08 | [Document check](08_document_check.py) | Questions generated from a requirements table; rules name exactly what is missing |
| 09 | [Bulk jobs](09_batch_throughput.py) | Dedupe, chunked `decide_many(..., errors="return")`, refused inputs recorded instead of failing the job |

## Patterns to reuse

- **The model measures, code decides.** Every example ends in plain `if`
  rules you can read, test and change without retraining.
- **Abstaining is a feature.** A small `probability_margin`, a changing winner
  or a low-confidence check sends the case to a person (01, 02, 06).
- **Ask everything about one input in one call.** `predict` shares the input
  across questions and mixes Choice, Score and Noul (01, 07, 08).
- **Thresholds come from data.** Choose them on labelled cases, then confirm
  on cases you did not use to choose them (05).
- **Log the numbers, not just the answer.** Probabilities, model ID and policy
  version let you replay and re-tune later (01, 05).
- **Guard effects.** Allowlists, approvals for irreversible actions, step
  limits and repeat detection live in code, not in the model (04).

## Tested on a real model

Every example also ran on a trained Wyat model (CPU, 2026-10-14). The code
paths all worked. Some decisions were wrong, which is why examples 05 and 06
exist. Observations worth showing:

- In 03, grading passages kept only the right passage for both shipping
  questions and kept nothing for gift cards. It also abstained on the
  borderline receipt question, the safe direction for this gate.
- The groundedness gate caught a claim that contradicted the source (basic
  example 04) but passed an invented claim the source never mentions (03).
  Measure such a gate on your own cases before relying on it.
- Grades for unrelated passages can sit close to grades for weakly related
  ones, so a single cut does not transfer between datasets. Choose it as in
  05, on your own labelled cases.
