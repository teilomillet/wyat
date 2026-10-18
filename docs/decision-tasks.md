# Decision task meanings

wyat supports three structured decision tasks: **Choice**, **Score**, and
**Noul**. The default model is
[wyat-1-800m](https://huggingface.co/teilomillet/wyat-1-800m), whose
`wyat-option-fork-v1` format supports all three through `model.decide(request)`
and `model.predict(record)`. Loading its name downloads and verifies the pinned
release on first use. For a team-facing walkthrough, see
[Explaining decisions](explaining-decisions.md).

Load a supported artifact with `wyat.load("teilomillet/wyat-1-800m")`, then
call `model.decide(request)` or `model.predict(record)` for a multi-question
record, and `model.predict_many` for several records.
[Obtain a supported artifact](model-acquisition.md) and follow the
[Python API guide](api.md). The package contains no model weights.

| Artifact format | Tasks | Backends |
| --- | --- | --- |
| `wyat-option-fork-v1` (800M) | Choice, Score, Noul | PyTorch CPU/CUDA/MPS, MLX Apple Silicon |
| `wyat-native-full-v1` (legacy compatibility) | Choice, Score, Noul | PyTorch CPU/CUDA, MLX Apple Silicon |
| Single-choice candidate-head | Choice | PyTorch CPU/CUDA, MLX Apple Silicon |

## Task capabilities

| Application need | Current operation | Meaning |
| --- | --- | --- |
| Select an allowed action | `model.decide(request).selected` | Highest-ranked application ID; selection does not execute an action. |
| Inspect a candidate's raw score | `decision.scores_by_id[id]` | Finite raw logit; its absolute magnitude is not a rating or confidence. |
| Inspect relative alternatives | `decision.probabilities_by_id[id]` | Softmax weight among supplied candidates; not calibrated correctness. |
| Rank supplied choices | `decision.ranking` | IDs in descending raw score order; exact ties preserve input order. |
| Rate against an ordered rubric | `Request.score(...)` / `decision.expected_ordinal_value` | Supported on option-fork and native full-task artifacts (`0..N-1` rubric levels); `wyat.interpret` gives the same value on a `ScoreAnswer`. |
| Estimate yes/no probability | `Request.noul(...)` / `decision.probability_true` | Supported on option-fork and native full-task artifacts (`true`/`false` options); `wyat.interpret` gives the same value on a `NoulAnswer`. |

The single-choice candidate-head encoder supports `task="choice"` and rejects non-Choice
tasks rather than guessing. Native full-task (`wyat-native-full-v1`) and
option-fork (`wyat-option-fork-v1`) artifacts render each task (`choice`,
`score`, `noul`) with their respective trained task framing.

An application can include a review/abstain action in its supplied menu or apply
its own explicit policy using `decision.score_gap`,
`decision.probability_margin`, or `decision.normalized_entropy`. The library
does not silently impose a confidence threshold, invent another outcome, or hide
an operational failure as abstention.

## Distinguish selection, ratings and yes/no probabilities

Choice selects among supplied candidates. Its raw `scores` are logits, not
rubric ratings. Naming two Choice options `true` and `false` still produces a
Choice distribution; use `Request.noul(...)` (`task="noul"`) with a model that
supports Noul when a binary true/false task is intended.

An ordinal Score task (`Request.score(...)`, `task="score"`) uses an ordered
rubric (`"0"`, `"1"`, ..., `"N-1"`) and exposes `decision.expected_ordinal_value`
(the probability-weighted expected index, which can fall between rubric levels
and differs from selecting the single most likely level). A Noul task
(`Request.noul(...)`, `task="noul"`) exposes `decision.probability_true` (the
weight assigned to `"true"`); the application owns any Boolean threshold.

Evaluate the supported model on relevant labeled cases and retain operational
failures separately from semantic outcomes.

## See each task in use

The [use-case examples](../examples/use_cases/README.md) are short scripts that
run with `--demo` (no weights) or `--model` (a local model directory or
`teilomillet/wyat-1-800m`).

| Task | Typical use | Example |
| --- | --- | --- |
| Choice | Route a message; choose an agent's next step; pick a reply | [support routing](../examples/use_cases/01_support_routing.py), [agent next step](../examples/use_cases/02_agent_next_step.py) |
| Score | Sort by urgency; rerank search results | [ticket urgency](../examples/use_cases/03_ticket_urgency.py), [rerank](../examples/use_cases/05_rerank.py) |
| Noul | Check an answer against its source; flag signals | [answer check](../examples/use_cases/04_answer_check.py), [email triage](../examples/use_cases/08_email_triage.py) |
| Mixed, one record | Several questions about one document | [contract review](../examples/use_cases/07_contract_review.py) |

[Expert examples](../examples/use_cases/expert/README.md) combine the tasks
with abstention, calibration and guards.
