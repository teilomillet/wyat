# Validate agent use and international understanding

Measure correct completion of the application task first. Valid output, successful
inference, a plausible explanation and a short script are useful observations but
do not each establish correct use. Keep setup, interpretation, learned decisions
and executed application outcomes as separate checks.

## Repeatable software checks

From the checkout:

```sh
uv run pytest -q tests/test_agent_contract.py tests/test_recovery.py \
  tests/examples/test_workflows.py
```

These fixed-logit checks retain exact IDs for French, NFC/NFD variants, CJK,
Arabic/RTL, emoji and mixed scripts. They exercise a second-position winner,
stable ties, JSON round-trip association, a misleading Score mean, Noul's
true-ID lookup, duplicate-ID rejection, action boundaries and failure-inclusive
policy denominators. Recovery tests preserve the v1 HTTP body and check its safe
reason header and CLI line provenance. No learned model or human is tested.

Translate instructions, evidence and rubric descriptions. Keep protocol fields and values
such as `query`, `context`, `type`, `criteria`, `score`, `noul`, `true` and `false`
unchanged. Application IDs remain exact, including distinct NFC/NFD spellings.
The configured tokenizer may normalize input text internally. Do not translate
or replace a trained renderer's prefix/suffix to make it appear language-neutral;
that would change the model recipe and needs its own training/evaluation evidence.

## Fresh-agent integration trials

Before a confirmation trial, freeze the wheel hash, documentation packet, supported
artifact/revision, model/runtime settings, exact prompts, allowed assistance,
environment, time caps, expected behavior and grader. Assign fresh-context agents
equivalent tasks and randomize documentation variants when comparing a change.
Use one independent task assignment per agent as the analysis unit. Retain the
agent/runtime identity when available; common prompts or model settings are
shared assumptions, not independent model replications.

Require each agent to preserve its first complete program before execution,
commands, errors, repairs, missing output and final interpretation. Enforce reading
restrictions with an isolated filesystem when available; otherwise report that
they were instructions only. Grade behavior and meaning against the frozen task,
not an unstated output format. Replay independently from an installed wheel
outside the checkout. Preserve failed setup separately from fixture completion.

Include these cases in the supplied task packet:

- Choice IDs `review, ship, hold` with logits `0, 2, 2`: select `ship`, retain input
  order at the tie, associate values by ID and explain that no action ran.
- Score levels `low, medium, high` with logits `0, -1000, 0`: expected index 1.0
  does not make the middle level likely; retain the full distribution.
- Noul criteria ordered `true, false` with logits `0, log(4)`: look up `true`
  by ID and apply the stated policy to the returned value without display rounding.
- A complete local setup and dispatch task, including a missing runtime/artifact
  or input rejection. A fixed scorer cannot satisfy trained-inference setup.
- A failed prediction or action that remains an explicit failure, rather than
  receiving a fabricated decision or semantic-review label.

All assigned trials, including timeout, abandonment, assistance and missing output,
remain in the completion denominator. Report each task cell. Record installation,
acquisition and warm task time separately, including failed attempts at their
declared caps. A later repaired script does not replace its first-attempt record.
Do not infer a speedup from a small unmatched cohort or agent self-ratings.

## Independently labeled language-quality trials

Freeze language/source strata, real supported artifacts, model policy and complete
menus before confirmatory inference. Use a representative cohort independently
labeled by competent speakers, with rubric examples and adjudicated ambiguities.
Keep translations, paraphrases and repeated menu orders bound to their original
source owner so they do not masquerade as independent cases. Separate language
comprehension from task difficulty with matched scenarios when practical, while
retaining genuinely local-language cases that translation cannot replace.

Measure instruction interpretation, rubric meaning, actual predictions and recovery
separately for each language and task. Choice needs correctness and retained failure
coverage; Score needs rubric-specific error and distribution checks; Noul needs
probabilistic error and a separate empirical calibration analysis. Freeze application
thresholds and their error/review costs, or fit them on a separate development set
before untouched confirmation. Unsupported tasks remain unsupported; binary Choice
does not become native Noul because its descriptions are translated.

Reject a multilingual-quality claim if a predeclared language/task threshold fails,
coverage is missing, labels lack independent review or confirmation data were used
to tune the recipe. Keep all failed/missing/invalid/unsupported outcomes visible.
Report owner-grouped uncertainty and each stratum, rather than hiding a weak
language behind an aggregate. Exact Unicode transport cannot substitute for these
observations.

Human understanding requires a separate human study with its own consent,
participants and frozen task protocol. Agent trials cannot supply its completion
rates. The [ergonomics standard](ergonomics.md) retains the prospective human
targets; the existing agent study and this software suite do not establish them.
