# Decision task meanings

For a team-facing walkthrough of the calculation, application policy and limits
of an explanation, see [Explaining decisions](explaining-decisions.md).

Wyat currently loads **Choice models** through its local Kai adapter. A Choice
returns an application ID, raw candidate logits, and relative probabilities.
The raw `scores` are not rubric ratings, and a two-option Choice does not become
the native Noul task merely by naming its options yes and no.

Choose the execution path before choosing an import:

| Path | Entry point and owner | What it establishes |
| --- | --- | --- |
| Ordinary local inference | Public `wyat.load(path)` → `Model.decide(Request)` | Runs the admitted full Kai artifact for Choice. [Acquire it](model-acquisition.md). |
| No-weight typed readout | Public `wyat.tasks.public_questions` and `interpret` | Parses questions and interprets supplied logits for Choice, Score and Noul; performs no trained inference. |
| Private source-pinned adapter research | `scripts/training_optimization/native.py` authenticates an archived executor whose class provides `NativeModel.from_adapter` | Runs the separately bound research adapter/initializer recipe; not a `wyat.tasks` import or ordinary SDK loader. |
| Private experimental full-native module | `wyat.loading.native.load_native`, with its own `NativeModel` class and `wyat-native-full-v1` manifest | A separate package-internal full-artifact path, not the archived `from_adapter` class or an extension of `wyat.load`. Artifact admission and learned quality need their own receipts. |

The two private research paths describe the development checkout and are not
included in the ordinary distribution or enabled by the public loader.

Do not guess `from wyat.tasks import NativeModel`; that class is not exported
there. The ordinary integration path needs only the first row. The
[complete workflows](agent-workflows.md) clearly label their fixed scorers and
pure typed interpreters.

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

## Native typed research contract

`wyat.tasks` supplies one pure parser, exact segmented encoder, and typed result
interpreter for `wyat-native-typed-segmented-independent-questions-v1`. It supports
the accepted research record's Choice, Score and Noul meanings without loading
weights. It does not enable those tasks in `wyat.load()` or provide a packaged
native-adapter loader. The source-pinned executor described in the table uses
`NativeModel.from_adapter(adapter, initializer)`; actual checkpoint conformance
is a separate launcher check. The private checkout's full-native class has a
different loading interface and must not be substituted by matching its name.

The record contains only `state` and an insertion-ordered `questions` mapping.
Each question has `type`, optional `instructions` (defaulting to its exact ID),
and task-specific `criteria`. State is verbatim text or canonical finite JSON;
references, supervision, source/review metadata and labels remain external.

| Task | Criteria | Result's principal value |
| --- | --- | --- |
| Choice | Ordered IDs mapped to nonempty descriptions | `ChoiceAnswer.choice`: first maximum ID; `ranking` keeps input order at ties. |
| Score | Ordered list of 2–255 rubric descriptions | `ScoreAnswer.expected_ordinal_value`: `sum(i*p_i)`, with generated IDs `"0"` onward. |
| Noul | Omitted/null or exact `false`/`true` mapping | `NoulAnswer.probability_true`: lookup by `"true"`, regardless of binary order. |

All tasks admit 2–255 options. Default Noul descriptions are exactly `False.`
then `True.`. Custom binary ordering stays intact. Results retain exact option
IDs, FP32 readout `logits`, and complete probabilities, with detached
`logits_by_id` and `probabilities_by_id` dictionaries. Principal values derive
from those snapshots rather than duplicate independently writable state.

```python
from wyat.tasks import ScoreAnswer, interpret, public_questions

record = {
    "state": {"invoice": "overdue"},
    "questions": {
        "urgency": {
            "type": "score",
            "instructions": "Rate urgency using the ordered rubric.",
            "criteria": ["Can wait.", "This week.", "Today."],
        }
    },
}
_, questions = public_questions(record)
answer = interpret(questions[0], (0.0, 1.0, 2.0))  # fixed logits, not inference
assert isinstance(answer, ScoreAnswer)
print(answer.expected_ordinal_value)
```

Run [the complete no-weight example](../examples/tasks/main.py) with
`uv run python examples/tasks/main.py`. `encode_record(tokenizer, record)`
encodes each question independently using the exact accepted State / Task type /
Question ID / Instructions prefix, separate complete option segments, and native
suffix. The tokenizer receives `add_special_tokens=False`. Complete rows and
segments exceeding their explicit limits fail without truncation; the executor
owns padded batch limits.

Input identities remain exact; tokenizer normalization remains its configured
policy. The intended tokenizer normalizes NFC internally, so exact Unicode
application IDs do not imply unchanged codepoints in tokenized text. Strict JSON
transports must reject duplicate keys before creating dictionaries.

The readout reproduces the native decoder's FP32 logit conversion before ties
and selection. Overflow fails as `ModelOutputError`. Stable stdlib softmax uses
double precision; reference probabilities and expectations compare to native
CPU FP32 with absolute/relative tolerance `1e-6`. Option order and selected IDs
must match exactly. These fresh weights inherit no released-Kai Score offsets
or empirical calibration claim.

The canonical model-bound contract and label-free actual-tokenizer fixtures are
retained in `artifacts/typed-task-conformance-20261008/`. They bind the immutable
training recipe and local initializer identities to the production interpretation
owner. The launcher and typed benchmark consumers must use that owner and retain
unsupported/missing/failed outcomes; artifact loading and learned task quality
remain separate evidence.

## What the direct competitors expose

[TypeSafe's official Python SDK](https://docs.typesafe.ai/sdk/python) exposes
named Choice, Score, and Noul questions and typed answers. Choice probabilities
are keyed by alternative ID; Score probabilities are keyed by rubric level.
[Its Score task](https://docs.typesafe.ai/primitives/score) returns the
probability-weighted expected level, while
[Noul](https://docs.typesafe.ai/primitives/noul) returns true/yes probability.
The inspected SDK is pinned to
[`f078f1e`](https://github.com/typesafe-ai/typesafe-sdk-python/tree/f078f1e208a0d885154dc758344ae4fce77ac168).
It is a hosted client with authentication, network, and retry obligations.

[Canonical Laya](https://github.com/NandhaKishorM/laya/tree/3cf26cbcb18725dbc2d127bb8bb2c4c43243ae63)
provides local typed decisions and schema projection. Its ordinal Score and
schema projection can consume the same distribution differently: expected level
and winning level are not interchangeable. Its processed probabilities also do
not satisfy Wyat's raw-logit `Scorer` contract by renaming the values.

These are inspected integration contracts, not comparative model-quality,
multilingual-competence, speed, or calibration measurements. A SDK's `confidence`
field need not mean probability of correctness. TypeSafe's
[documented confidence](https://docs.typesafe.ai/confidence) is computed from the
answer distribution; adopting that formula would not calibrate Wyat's scores.

## Standard for admitting another task

A task needs a distinct validated input and a result whose principal value has
one clear meaning. Keep the complete distribution inspectable. Do not combine
unrelated meanings behind a generic `score` field or automatically threshold a
probability into a Boolean.

For an ordered rubric with levels `0..L-1`, native Kai Score reports
`sum(i * p_i)`. It can fall between levels. It is neither the winning level nor
a physical-unit regression; equal index spacing is part of that readout. Noul
reports `p_true`; the application owns any Boolean threshold and its error costs.

A released-Kai extension must admit its actual task recipes. The pinned native
[Kai release](https://huggingface.co/vllm-sr/Decision-2.0-Kai-0.6B/tree/cd49ea3813fd8ba0928a9a23ef6c9a0f2f0cd764)
changes the rendered task type and uses model-bound offsets for five-level
Score. Removing the current Choice guard or taking an expectation over arbitrary
Choice logits would not reproduce that native contract.

Before exposing it:

1. Bind the task, exact renderer/tokenization, permitted levels/keys, and
   output interpretation to an admitted artifact recipe.
2. Validate and fingerprint every consumed offset/calibration file with its
   model/cardinality binding. Preserve raw and adjusted values distinctly.
3. Give Python, CLI, and HTTP one canonical interpretation owner, with explicit
   failures and no truncation or retries hidden in the decision layer.
4. Compare tokens, spans, logits, probabilities, and typed answers with pinned
   primary reference fixtures, including ties, Unicode, malformed metadata,
   Noul defaults, and different rubric sizes. Check each supported backend.
5. Evaluate task quality separately on independent labeled cohorts. Score and
   Noul need their own error/probability metrics; Choice accuracy alone cannot
   establish them. Retain failures and language/cardinality strata.

Exact contract conformance is an engineering target. Perfect semantic decisions
are not established by valid types, successful inference, or a well-designed API.
The [training/deployment contract](training-contracts.md) governs admission.
