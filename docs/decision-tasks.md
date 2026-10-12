# Decision task meanings

Wyat's local Kai adapter supports **Choice**. A Choice returns an application
ID, raw candidate logits and relative probabilities. For a team-facing
walkthrough, see [Explaining decisions](explaining-decisions.md).

The supported path is `wyat.load(path)` followed by `model.decide(request)`.
[Obtain the supported artifact](model-acquisition.md) and follow the
[Python API guide](api.md). The package contains no model weights.

## Released Kai loader capabilities

| Application need | Current operation | Meaning |
| --- | --- | --- |
| Select an allowed action | `model.decide(request).selected` | Highest-ranked application ID; selection does not execute an action. |
| Inspect a candidate's raw score | `decision.scores_by_id[id]` | Finite raw logit; its absolute magnitude is not a rating or confidence. |
| Inspect relative alternatives | `decision.probabilities_by_id[id]` | Softmax weight among supplied candidates; not calibrated correctness. |
| Rank supplied choices | `decision.ranking` | IDs in descending raw score order; exact ties preserve input order. |
| Rate against an ordered rubric | Native Score is not yet admitted | Needs a Score input recipe, rubric interpretation, and model-bound postprocessing. |
| Estimate yes/no probability | Native Noul is not yet admitted | Needs its trained binary task recipe and an explicit true-probability interpretation. |

The generic custom `Scorer` may accept its own task strings, but `Model` still
interprets its output as candidate logits and a Choice distribution. It does
not supply a native Score or Noul interpretation by relabeling that result.
The current Kai encoder rejects non-Choice tasks rather than guessing.

An application can include a review/abstain action in its supplied menu or apply
its own explicit policy. The library does not silently impose a confidence
threshold, invent another outcome, or hide an operational failure as abstention.

## Distinguish selection, ratings and yes/no probabilities

Choice selects among supplied candidates. Its raw `scores` are logits, not
rubric ratings. Naming two options `true` and `false` still produces a Choice
distribution; it does not enable another trained task.

An ordinal Score task would require an ordered rubric and its own model-bound
interpretation. A probability-weighted expected index can fall between rubric
levels and differs from selecting the most likely level. A Noul task would
return the weight assigned to `true`; an application would own any Boolean
threshold. Neither task is exposed by the supported local Kai loader.

Changing `Request.task`, renaming logits or averaging arbitrary Choice
probabilities cannot establish those additional capabilities. Evaluate the
supported model on relevant labeled cases and retain operational failures
separately from semantic outcomes.
