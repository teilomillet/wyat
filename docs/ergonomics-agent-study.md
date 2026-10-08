# Agent integration study — 8 October 2026

The product objective is correct unaided completion first, then lower discovery
and completion time, then fewer necessary concepts. This study tested the public
Python contract using local fixed-logit fixtures. It does not establish human
usability, trained-model quality, or comparative superiority over other libraries.

## Frozen method

Twelve fresh-context agents used a baseline wheel and copied public docs. Twelve
other fresh-context agents used an updated wheel and docs. Each phase assigned
two agents to each of six families: construct a multilingual request, associate
scores with IDs, batch different menus, recover from errors, evaluate with a
failed case, and distinguish supported tasks from raw scores. Equivalent ID/text
variants, expectations, and the 600-second cap were frozen before baseline.

Agents could use supplied public docs, examples, and installed-package help.
They could not inspect source, tests, research, or sibling trials. This restriction
was instructed rather than enforced by filesystem isolation. Each agent retained
its first complete executable program, errors, repairs, final output, and sampled
UTC timestamps. A separate grader reviewed task semantics and replayed scripts
against the frozen expectations from outside the checkout.

Time starts at the agent's sampled pre-documentation timestamp and ends at its
sampled validated-completion timestamp. Queue/dispatch and report-writing time
are excluded. Individual discovery-command times were often unrecorded or
bounded; these intervals are not inference latency. Dependencies were prepared,
and no weights or hosted APIs were used. Exact runtime model identifiers were
not exposed; spawn requests inherited settings without overrides.

## Observations

The independent baseline replay passed all twelve first-program artifacts. One
baseline evaluator discovery command guessed an unavailable top-level import
before finding `wyat.evaluation`. Thus artifact success does not mean that every
command or entire journey succeeded without error.

Independent replay passed all twelve confirmation first-program artifacts and
final programs. Original first execution was confirmed for eleven; one shell
capture failure left its original Python exit status unknown. Both cohorts
completed within their reported caps. Two grader assumptions were corrected
against the unchanged task wording: different declared menu orders and keyed
rather than array-shaped raw scores were permitted. The same corrected checker
replayed both cohorts. Original strict-checker results (baseline 12/12,
confirmation 9/12) and the correction rationale remain preserved.

Mean sampled completion time was 71.5 seconds at baseline and 59.6 seconds at
confirmation. Mean discovery time was 70.6 and 53.6 seconds respectively.
Family means show why an aggregate is insufficient:

| Task family | Baseline seconds | Confirmation seconds |
| --- | ---: | ---: |
| construct | 59.3 | 59.0 |
| inspect | 84.1 | 49.0 |
| batch | 61.2 | 52.0 |
| recovery | 85.6 | 71.7 |
| evaluate | 66.9 | 74.6 |
| capabilities | 72.0 | 51.5 |

Evaluation was slower in the confirmation cohort despite the new guide. The
small, later, unmatched cohort and partially captured timing do not establish
a causal speedup or a reliable T95. Baseline artifact success was already at
the observed ceiling, so completion improvement cannot be claimed. The final
result is contract-audited conformance under a corrected checker, rather than
confirmation under a wholly unchanged verifier.

## Resulting changes

- `Request` accepts ordered candidate sequences and stores a tuple snapshot.
  Ordinary application lists need no tuple conversion, while later list mutation
  cannot change the request.
- `Decision.candidate_ids`, `scores_by_id`, and `probabilities_by_id` associate
  values directly with exact application IDs. They preserve the four stored
  dataclass fields and existing JSON format. Ranking and raw score arrays still
  have different orders; the properties recover association through the stable
  ranking contract and work without retaining a request.
- No-weight examples show a second-position winner, Unicode result lookup, and
  Python evaluation with correct, wrong, and failed calls. Failure remains in
  both the report and accuracy denominator.
- CLI JSONL uses strict UTF-8 on real standard streams; exact IDs are preserved
  without normalization. Invalid case-ID Unicode fails at construction.
  Already configured streams and custom injected streams remain usable.
- Malformed scorer cardinality uses public `ModelOutputError`, including custom
  containers whose reported length differs from actual iteration. Backend and
  iterator execution exceptions retain their original identity.

An agent can now express the integration as:

```python
request = Request(query=instruction, context=message, candidates=candidates)
decision = model.decide(request)
chosen_id = decision.selected
raw_score = decision.scores_by_id[chosen_id]
probability = decision.probabilities_by_id[chosen_id]
```

The user reasoning is a hypothesis: an application already owns its IDs and
records, so preserving those IDs and exposing keyed values fits its next step.
Selection returns an ID; the application owns execution. Agent self-ratings
provide diagnostic impressions, not human preference measurements.

## Competitors and task meanings

The canonical [TypeSafe Python SDK](https://docs.typesafe.ai/sdk/python) and
[Laya](https://github.com/NandhaKishorM/laya) expose typed decision tasks and
ID-aware distributions. Their hosted/local lifetimes and interpretation paths
differ. Source review identified useful integration patterns; no peer latency,
calibration, language quality, or direct usability comparison was executed.

The [task guide](decision-tasks.md) separates raw candidate logits from rubric
Score and binary Noul. Score means an expected ordinal index over ordered levels;
Noul means P(true), with any Boolean threshold owned by the application. The
released local Kai loader still admits Choice only. The native research typed
recipe has different exact serialization and artifact admission requirements.
It cannot be enabled by changing a task string or copying released-model offsets.

The Bitter Lesson informs the direction toward general mechanisms and repeated
external feedback. It does not prove that a particular Python interface is best.
No language-specific prompt heuristics or automatic model-quality claims were
introduced by these Unicode mechanics checks.

## Verification and limits

Connected core/loading/CLI/evaluation/server/example checks passed **228 tests**.
Scoped Ruff lint/format and strict mypy passed. Independent review checked
129,444 candidate-order/tie perturbations and found cardinality and stream-lifetime
issues, which were repaired and independently rechecked. A no-dependency wheel
installation outside the checkout passed core, CLI-help, and all three examples
without optional runtime imports.

One third-party Starlette deprecation warning remains. Native Windows behavior,
fresh inference installation, real-model multilingual quality, and human
naturalness remain unmeasured. Baseline optional-server metadata was inaccurate;
confirmation also had an extra development site-packages path. Those packages
were unused by the core/evaluation tasks, but environments were not perfectly
matched. One confirmation shell wrapper lost the original Python exit status;
a subsequent successful replay cannot recreate that missing observation.

The frozen study and detailed reports are retained under
`artifacts/ergonomics-agent-study-20261008/` in this checkout: task freeze,
expectations, wheel/doc/source identities, code/output/history, independent
replays, timing audit, and review resolutions. These ignored research artifacts
are not distributed in the wheel. Existing unrelated work was preserved.
