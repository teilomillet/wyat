# Explaining a decision to your team

An explanation should answer three questions: **what did the model return,
how was that value calculated, and why does our application act on it?**
The first two have an exact computational answer. The third depends on your
question, rubric, validation evidence and application policy.

```text
state + question + allowed outcomes
    → model logits → typed result and full distribution
    → team policy → application action or review
```

Keep the model result, the policy decision and the executed action distinct.
The [complete workflows](agent-workflows.md) exercise these boundaries, including
application actions, review resolution and separate operational failures. The
policy-evaluation example reports explicit denominators and compares unrounded
probabilities; its cutoffs are illustrative application choices.
Selecting `refund` does not issue a refund. A number alone does not establish
that an action is justified.
The native recipe answers each question independently. A billing answer does
not authorize a refund or establish consistency between the answers; the
application must express any such dependency in its policy.

## Three meanings, three plain explanations

| Task | Meaning | What to tell a teammate |
| --- | --- | --- |
| Choice | Select the ID with the highest logit among the supplied options. | “`refund` ranked first among `wait` and `refund`. Its relative weight was 0.881.” |
| Score | Average the zero-based rubric indices using their probabilities. | “The expected rubric index was 1.167 on levels 0–2. Here are the weights on each level.” |
| Noul | Return the probability assigned to the option with ID `true`. | “The model assigned 0.750 to ‘true’ for this exact yes/no question. Our policy sends this result for review.” |

These interfaces give each task one clear purpose. Choice returns an exact
application ID, so callers can use a supplied outcome without parsing prose.
Score summarizes the weights on an ordered rubric; the complete distribution
keeps distinctions that the average loses. Noul exposes the true weight so the
team can choose a policy reflecting its error costs instead of receiving a
hidden library threshold. These are design reasons, not evidence of accuracy.

Write instructions, evidence and outcome descriptions in your team's language.
Keep Python/JSON field names and task values such as `query`, `context`, `type`,
`criteria`, `choice`, `score` and `noul` unchanged. Noul's option IDs stay `true`
and `false`; translate their descriptions. Application IDs keep their exact
spelling between input and result, including distinct NFC/NFD spellings.

These numbers come from the fixed-logit example below, not a trained model's
assessment. **Noul is a yes/no task.** It is distinct from Python `None`, JSON
`null`, an absent answer or an automatic abstention. In the input contract,
omitted/null Noul *criteria* request the default false/true descriptions; that
does not make the resulting answer null.

## Why these numbers?

`wyat.tasks.interpret(question, logits)` takes a finite FP32 snapshot of each
logit and computes stable softmax:

```text
p[i] = exp(logit[i] - max(logits)) / sum(exp(logit[j] - max(logits)))
```

The complete distribution sums to approximately one. These are relative model
weights among the supplied outcomes, not evidence of calibrated correctness.
Raw logits have no rubric units, and changing the question, descriptions or menu
can change the distribution.

**Choice:** with logits `[0, 2]` for `[wait, refund]`, the weights are approximately
`[0.119, 0.881]`. `refund` has the largest logit, so it is selected. Exact ties
preserve supplied option order. Native FP32 conversion can turn very close
values into an exact tie; record the resulting logits and original order.
This explains the selection calculation. It does not establish a semantic
reason such as “refund because the invoice is duplicated.”

**Score:** use an explicit ordered rubric, for example:

| Index | Description | Example weight |
| --- | --- | --- |
| 0 | Can wait. | 1/6 |
| 1 | This week. | 1/2 |
| 2 | Today. | 1/3 |

The expected ordinal value is `0 × 1/6 + 1 × 1/2 + 2 × 1/3 = 7/6`, approximately
`1.167`. Its unit is **rubric index**, not days, confidence or a selected level.
The most likely level here is 1. The readout uses evenly spaced numerical
indices; it does not prove equal differences between the descriptions.
Avoid assigning a new interpolated description to the fractional value.

Always show the distribution alongside this summary. For logits `[2, -4, 2]`,
the weights are approximately `[0.499381, 0.001238, 0.499381]`. The expected
index is exactly `1.0`, although the middle level has almost no weight.
“The model chose the middle level” would be misleading.

**Noul:** for the question “Does the message concern billing?”, logits
`[0, log(3)]` on `[false, true]` produce approximately `[0.25, 0.75]`.
`probability_true` is the weight attached to the ID `true`, even when the
criteria order is reversed. It is not a boolean decision. An illustrative
application policy might accept true at `p_true >= 0.90`, accept false at
`p_true <= 0.10`, and send everything else for review. Under that policy,
`0.75` means review. Those cutoffs and inclusive comparisons are the team's
choice, not Wyat defaults or empirically justified thresholds.

These arithmetic explanations use the canonical `interpret()` path. Directly
constructed answer values validate supplied logits and probabilities without
requiring that the probabilities equal softmax of the logits. Establish their
producer before claiming that relationship. Released Kai's generic `Decision`
also represents Choice; its raw `scores` must not be relabeled as native Score.
See [the task contract](decision-tasks.md) for the separate interfaces.

## What justifies using the result?

Define the question and options so teammates can agree on what a correct answer
would mean. For Score, provide concrete examples of each rubric level and a
consistent low-to-high ordering. For Noul, say exactly what makes the proposition
true; use a separate question for a separate proposition.

Then evaluate the actual model on relevant examples, languages and menus.
Report task quality and failure cases. If the application treats `0.75` as a
real-world frequency, test calibration for that use. Choose policy thresholds
against observed errors and their costs; evaluate the review path too.
The example's synthetic logits establish none of these empirical claims.

Retain a small decision note that lets a teammate reconstruct the result:

- The exact input, or a retrievable access-controlled reference to it.
- The question, ordered option IDs and descriptions, and rubric where applicable.
- The model/artifact identity, task recipe and readout version.
- The readout logits, complete distribution and principal value with its unit.
- The application policy and version, its decision, and any action actually taken.

A hash can establish input identity, but cannot explain the input's meaning by
itself. Keep sensitive inputs behind their existing access boundary rather than
copying them into an unrestricted log.

When presenting evidence from the input, identify it as supplied evidence.
Neither that evidence nor a plausible generated rationale proves which facts
caused the model's result. Causal attribution requires its own investigation.
For a teammate asking “why?”, start with the exact calculation and policy,
then state the relevant validation evidence and what remains unknown.

## Run the worked example

From the repository, with Python 3.12 or newer and Wyat installed:

```sh
python examples/explanations/main.py
```

The [example source](../examples/explanations/main.py) prints a plain JSON
decision note for Choice, Score and Noul, plus the Score counterexample.
It uses fixed logits, loads no weights and executes no application actions.
It demonstrates the pure typed readout contract; it does not establish loader
capability, trained-model quality or measured human usability.
