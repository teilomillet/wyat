# Wyat ergonomics standard

Make a correct decision easy to express, understand, and recover when it fails.
The preferred public path is **load once → describe a request → decide → use an
application ID**. This is a design choice supported by the current application
tasks, open to revision when comparative user evidence supports a better one.

The Python package remains `wyat`. Ergonomic work does not select a new model
architecture or establish trained-model quality.

## Objective function

Among designs that preserve the hard contracts, maximize, in order:

1. **Q: correct unassisted task completion**, including understanding the result
   and recovering from the specified failure.
2. **Lower T95: the 95th percentile of task time**, including failed attempts at
   their declared time cap.
3. **Lower K: concepts a caller must learn** to complete those tasks correctly.

In notation: `maximize (Q, -T95, -K)` lexicographically. Fewer lines, methods, or
dependencies are useful diagnostics, not substitutes for successful use.
Hidden preparation, conversions, ID mapping, and postprocessing still count.

Q is the equally weighted mean of completion rates for the assigned task/persona
cells below. Report each cell and each persona separately; an aggregate gain
cannot compensate for a supported regression in a task or persona. A correct
task must pass both its executable and interpretation expectations. Incorrect
interpretations, assistance, abandonment, timeouts, and missing outputs stay in
the assigned-trial denominator.

The prospective product target is **Q ≥ 90% and ≥ 80% in every assigned cell**.
These thresholds are choices for a future human study, not observed human scores.
Human Q, T95, and K remain unmeasured. The [agent study](ergonomics-agent-study.md)
provides bounded fixture integration evidence, not a comparative human usability win.

Before confirmation, freeze participants, versions, artifacts, task wording,
documentation, caps, environment, randomization, and analysis. Use counterbalanced
design order and fresh equivalent task variants to limit practice effects. Treat
the participant, rather than repeated attempts, as the independent unit.
Predeclare confidence intervals and practically meaningful differences; do not
select a winner from a tiny noisy advantage. At equivalent supported Q, compare
time; at equivalent supported time, compare conceptual burden. Inconclusive
evidence leaves the simpler established behavior in place.

## Hard contracts

- Preserve application IDs, candidate descriptions, request/candidate order,
  stable exact ties, and documented scalar/batch behavior.
- Retain finite raw-logit interpretation and relative, uncalibrated softmax
  probabilities. A selected ID identifies an action; it does not execute it.
- Keep local loading explicit. Importing the package does not load weights or
  optional runtimes. Do not add implicit downloads or custom model-code execution.
- Reject unsupported input explicitly. Do not truncate, retry, split, repair,
  skip, or invent successful output to hide a failure.
- Preserve the wire format, failure-inclusive evaluation, and explicit resource,
  concurrency, cancellation, and model-lifetime ownership.
- Preserve existing user work, model contracts, and privacy. Diagnostics must
  not expose request text merely to improve recoverability.

## Seven design rules

1. **Design from a complete user task.** Write the consumer example before adding
   an interface. Include preparation, result use, and a likely failure.
2. **Teach one recommended path.** Keep `load`, `Request`, `decide`, and
   `selected` as the ordinary path. Additional forms must solve an observed unmet
   task and share the same validation and interpretation.
3. **Use types where meaning needs protection.** Keep IDs distinct from
   descriptions, validated immutable inputs, and the small structural `Scorer`
   protocol. Avoid registries, inheritance frameworks, and scalar wrappers
   without a present caller need.
4. **Make effects and owners apparent.** Load outside request handlers. Ordinary
   attribute access performs no inference or I/O. Advanced runtime settings do
   not enter the first-use lesson. Direct model concurrency belongs to its
   scorer; the HTTP server supplies its documented admission gate.
5. **Make failures actionable.** A caller can distinguish rejected input from
   invalid scorer output. A diagnostic identifies the boundary and a permitted
   next step, retaining causes without concealing unrelated backend failures.
6. **Write for the next reader.** Use ordinary Python, meaningful names, clear
   intermediate values, early validation, focused functions, and comments that
   explain constraints. Keep the current Ruff and strict mypy tooling rather
   than starting a whole-repository style migration.
7. **Keep examples executable and claims bounded.** Test the contract and its
   connected boundaries. Fixtures verify mechanics; users establish usability;
   independent labeled cohorts establish model quality.

The maintenance test is simple: can a reader explain the information, effects,
and failures of one decision without traversing unrelated modules?

## Evaluation tasks

These task families are frozen for the next exploration. Confirmation must freeze
their exact prompts, fixtures, preparation, and expected artifacts separately.

| Task | Observable success | Personas |
| --- | --- | --- |
| U1 Explore without weights | Run the scorer example and identify it as a fixed-logit demonstration, not a trained decision. | Beginner |
| U2 First local decision | Install the selected runtime, obtain the declared supported artifact, load once, and return an allowed ID. | Beginner, integrator |
| U3 Use the selected action | Keep ID and description distinct; dispatch using the ID; explain that inference executes no action. | Beginner, integrator |
| U4 Batch reuse | Convert scalar use to two requests with different menus; preserve association and reuse the model. | All |
| U5 Interpret results | Recover scores by original candidate IDs when the winner is second; explain probability and tie semantics. | All |
| U6 Repair input | Correct duplicate IDs and a model-limit rejection using diagnostics and public docs; retain the first failure. | All |
| U7 Repair installation | Identify a missing runtime and install the correct extra in the application environment. | Beginner, integrator |
| U8 Integrate a scorer | Return finite raw logits in both orders; catch broken output separately from rejected input and backend execution failure. | Scorer author |
| U9 Serve the same request | Preserve Python/HTTP meanings; explain one worker's model lifetime and overlapping-request behavior. | Integrator |
| U10 Evaluate a failure | Retain failed cases in the denominator and locate their recorded errors. | Integrator |

Report first-use tasks separately from integration tasks. Use controlled logits
for interpretation tasks so an incorrect model prediction cannot be mistaken
for API difficulty. U2 is an actual inference/setup task; U1 cannot stand in for
it. Pin each model's artifact and capability before comparison. Record setup
time separately from warm task time and include both in the complete journey.

Measure K with a published concept ledger: count distinct domain concepts,
required configuration decisions, order/identity conversions, and recovery
branches across the complete journey. Aliases and hidden configuration do not
remove a concept. Independent annotators resolve disagreements before scoring;
the count remains a proxy for cognitive burden, not a human cognition measure.

For future human trials, the initial proposed cap is 30 minutes for installation
and artifact acquisition in U2, and 10 minutes for each other task. Predeclare
dependency/network availability and record infrastructure failures separately;
an unavailable environment supplies no ergonomic pass. Change these exploratory
caps only before confirmation, recording the reason. Keep original attempts,
edits, documentation visits, assistance, timestamps, and final outputs.

## What comparable libraries teach

The following is source inspection on 8 October 2026, not execution or a measured
leaderboard. Peers share some tasks but differ in model semantics and capabilities.
Compare common tasks and report capability coverage separately. Record versions
and source snapshots before any comparative trial.

| Reference | Useful convention | Wyat choice and tradeoff |
| --- | --- | --- |
| [Sentence Transformers CrossEncoder](https://www.sbert.net/docs/package_reference/cross_encoder/model.html) | `rank(query, documents)` returns sorted entries with positional `corpus_id` and score; explicit local-only loading is available. | Named candidates require more input construction but carry application IDs directly to selection. Test the whole dispatch journey, including the peer's ID mapping. |
| [scikit-learn estimators](https://scikit-learn.org/stable/developers/develop.html) and [probability columns](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression.predict_proba) | Predictable operations and explicit association of output columns with classes. | Preserve operation consistency and request/candidate association. A fixed class vocabulary or training API would not fit the current runtime. |
| [Transformers zero-shot classification](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/zero_shot_classification.py) | A short supplied-label call returns labels and scores together in ranked order. | Keep application ID/text distinct and one score interpretation. NLI classification and candidate-logit decisions are different model contracts. |
| [Aurelio Semantic Router](https://docs.aurelio.ai/docs/semantic-router/user-guide/guides/semantic-router) | A router call produces a named route after route examples, an encoder, and index setup. | Count complete setup, remote effects, and unmatched-route behavior. This embedding/index system is not Kai inference. |
| [Decision 2.0 Kai native interface](https://huggingface.co/vllm-sr/Decision-2.0-Kai-0.6B) | Named questions distinguish state, instructions, and criteria IDs; the interface covers additional task types. | The current adapter deliberately accepts only choice and local full artifacts. A broader peer interface does not authorize reinterpretation or expansion. |

The distinctive standard is **express a decision once, preserve application
identity, and keep its effects and interpretation predictable across supported
paths**. This is a product direction, not a claim that the concept is novel.

## How the named guidelines apply

[Jane Street's Effective ML](https://blog.janestreet.com/effective-ml/) supports
reader-first code, uniform interfaces, useful types, explicit errors, restrained
type machinery, and pragmatic purity.
[Design for the call site](https://dev.realworldocaml.org/files-modules-and-programs.html#design-for-the-call-site)
supports evaluating interfaces through consumer examples. These are OCaml design
lessons transferred to Python, not a company-wide Python style standard.

[Google's Python guide](https://google.github.io/styleguide/pyguide.html) supplies
public annotations and semantic documentation, careful exceptions, simple
control flow, and restraint with power features.
[Google's review guidance](https://google.github.io/eng-practices/review/reviewer/looking-for.html)
adds design, complexity, naming, documentation, and present-use review. Wyat
deliberately documents validation failures and retains its existing formatter
and tools; this is an adaptation rather than complete Google-style compliance.

[Google's Go guide](https://google.github.io/styleguide/go/guide.html) prioritizes
clarity and simplicity before concision; consistency resolves otherwise equal
choices. It permits internal complexity when that simplifies actual usage.
[Go review guidance](https://go.dev/wiki/CodeReviewComments) supports synchronous
lifetimes, useful errors, small interfaces, and examples. Keep idiomatic Python
exceptions and keyword arguments; do not transplant Go syntax or concurrency
machinery. [Effective Go](https://go.dev/doc/effective_go) is useful for durable
principles but explicitly is not actively updated.

## Simulated user reasoning

A beginner likely asks: “What instruction, evidence, and allowed actions do I
give it?” Named request fields and candidate IDs answer that question. The
instruction is `query`; the evidence is `context`. A small example must teach
those meanings. This hypothesis favors the current request shape, while U2–U6
can reject it if construction or interpretation proves unnecessarily difficult.

An integrator likely asks: “What stays loaded, and what happens when requests
overlap?” Visible loading and admission ownership answer that question. A
one-line facade that reloads or secretly caches a model would move complexity
into lifetimes, resource use, and invalidation.

A scorer author likely asks: “What do I return, and who is responsible if it is
wrong?” One batch method, a finite-logit contract, and a public output exception
answer that question. A broad plugin framework adds no demonstrated value here.

These are theory-of-mind hypotheses from agent walkthroughs. They expose risks;
they do not supply human completion rates or prove preference.

## First increment and remaining hypotheses

The first increment exposes the existing `ModelOutputError` publicly, makes
malformed output shape/numeric conversion use that boundary, and supplies missing
runtime recovery for consuming applications as well as contributors. The API
guide documents catch ordering and preserves backend failure propagation.

Keep these questions open until their task evidence warrants a change:

- Would accepting ordered lists and storing a tuple improve U2/U6 while retaining
  immutable request ownership? The second increment now accepts ordered sequences
  and snapshots them into tuples, supported by compatibility tests and agent tasks.
- Would ID-aware score access reduce U5 mistakes enough to justify added output
  structure? The second increment supplies derived IDs and keyed dictionaries,
  preserving the existing stored fields and wire format. Never zip ranked IDs
  with input-ordered scores.
- Does first-use acquisition need an independently verified download guide and
  a matching distribution release? The [pinned acquisition guide](model-acquisition.md)
  now supplies the full-Kai route and integrity check; the installation guide also
  supports a transferable built wheel. A matching registry release remains separate.
- Can safe structured model-limit diagnostics improve HTTP recovery without
  leaking request content? Model-owned reasons now cross HTTP in an allowlisted
  response header while preserving the v1 status/body. CLI failures retain physical
  line/batch provenance. [Recovery checks](../tests/test_recovery.py) exercise both;
  measured agent recovery improvement remains unknown.
- Direct construction of `Decision` can currently produce inconsistent output
  values, and an evaluation case ID containing invalid Unicode can fail later
  during serialization. The second increment validates case IDs at construction.
  Manual Decision construction remains compatible; new keyed properties check
  structure and finite numbers without validating an entire probability distribution.

## First-increment verification receipt

On 8 October 2026, the connected check passed **112 tests**:

```sh
.venv/bin/python -m pytest -q tests/test_core.py tests/test_loading.py \
  tests/test_cli.py tests/test_evaluation.py tests/test_server.py tests/server
```

Scoped Ruff lint/format and mypy checks passed for the changed production code
and tests. The scorer example and CLI help ran successfully. A fresh wheel was
installed with no dependencies into a new environment outside the checkout;
the installed core exercised scalar/batch inference, ID/score association,
public error access, and lazy imports. This is an offline fixture check, not
an inference-extra installation or trained-model check.

Independent review found a sized-but-noniterable output leaking `TypeError`.
The revised increment guards iterator creation, preserves its cause, and passed
an independent **49-test** core/loading recheck. Exceptions from scorer execution
or advancing custom output iterators propagate unchanged. The connected test
run reports one third-party FastAPI/Starlette deprecation warning.

The source checkout contains ongoing unrelated work. It was preserved; this
receipt covers the listed paths and tests, not the full checkout or a release.
Human studies, trained-model inference, acquisition, and peer execution remain
separate work. No comparative human ergonomics score is established.

## Agent-facing implementation follow-up

The [complete workflows](agent-workflows.md) connect routing, prioritization,
human review and failure-inclusive policy evaluation through the existing API.
The [validation protocol](agent-validation.md) separates repeatable interpretation
and Unicode mechanics from fresh-agent use, learned language quality and human
understanding. The acquisition guide records a trained installed-wheel CPU smoke
check; it does not retroactively expand the fixture study above into model-quality
evidence. No new default threshold, implicit action or automatic model repair was
introduced.

## Second increment: actual agent integration

The [agent-study receipt](ergonomics-agent-study.md) records twelve frozen
baseline tasks and twelve fresh confirmation tasks across six families. This
increment accepts list inputs without mutable request ownership, supplies
ID-aware result access without changing serialization, adds no-weight result
and Python-evaluation examples, and makes CLI JSONL strict UTF-8. It preserves
raw-logit semantics, failed-case denominators, exact Unicode IDs, explicit
loading, and optional-runtime import boundaries.

The product objective remains success first, then time, then required concepts.
For this small agent sample, report family-weighted completion and capped mean
time cost rather than infer a reliable T95. Guided fixtures and agent
self-ratings do not measure human naturalness, installation, trained language
quality, or superiority over competitors.

The [task meanings](decision-tasks.md) distinguish candidate logits from rubric
Score and binary Noul. Adding a task string or renaming softmax values cannot
admit a new model recipe. Training and benchmarking conformance must bind
inputs, exact rendering, the model artifact, and task-specific interpretation.
