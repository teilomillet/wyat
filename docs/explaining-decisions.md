# Explaining a decision to your team

An explanation should answer three questions: **what did the model return,
how was that value calculated, and why does our application act on it?**
The first two have an exact computational answer. The third depends on your
question, validation evidence and application policy.

```text
state + question + allowed outcomes
    → model logits → decision and full distribution
    → team policy → application action or review
```

Keep the model result, the policy decision and the executed action distinct.
The [complete workflows](agent-workflows.md) exercise these boundaries, including
application actions and separate operational failures.
Selecting `refund` does not issue a refund. A number alone does not establish
that an action is justified.

## Explain the returned Choice

`decision.selected` is the candidate ID with the highest raw logit. Exact ties
preserve input order. Scores and probabilities follow the original candidate
order; use `decision.scores_by_id` and `decision.probabilities_by_id` when
looking up a specific ID.

For decisions returned by `Model`, the probabilities are stable softmax weights:

```text
p[i] = exp(logit[i] - max(logits)) / sum(exp(logit[j] - max(logits)))
```

With fixed logits `[0, 2]` for `[wait, refund]`, the weights are approximately
`[0.119, 0.881]`. A teammate can say: “`refund` ranked first among these supplied
options, with relative weight 0.881.” That explains the calculation. It does not
establish a semantic reason such as “refund because the invoice is duplicated.”

These probabilities describe the supplied menu. They are not calibrated
probabilities that the selected action is correct. Changing the question,
evidence, option descriptions or menu can change the distribution.

Raw candidate scores are not rubric ratings. The supported local Kai loader
accepts Choice; [task meanings](decision-tasks.md) explain the distinction from
ordinal ratings and yes/no probability tasks.

Write instructions, evidence and descriptions in your team's language. Keep
Python/JSON field names and `task="choice"` unchanged. Application IDs preserve
their exact spelling between input and result; tokenizer normalization remains
part of the model's configuration.

## What justifies using the result?

Define the question and options so teammates can agree on what a correct answer
would mean. Use a separate request for a separate decision.

Then evaluate the actual model on relevant examples, languages and menus.
Report task quality and failure cases. If the application treats `0.75` as a
real-world frequency, test calibration for that use. Choose policy thresholds
against observed errors and their costs; evaluate the review path too.
The example's synthetic logits establish none of these empirical claims.

Retain a small decision note that lets a teammate reconstruct the result:

- The exact input, or a retrievable access-controlled reference to it.
- The question and ordered option IDs and descriptions.
- The model/artifact identity, task recipe and readout version.
- The raw logits, complete distribution and selected ID.
- The application policy and version, its decision, and any action actually taken.

A hash can establish input identity, but cannot explain the input's meaning by
itself. Keep sensitive inputs behind their existing access boundary rather than
copying them into an unrestricted log.

When presenting evidence from the input, identify it as supplied evidence.
Neither that evidence nor a plausible generated rationale proves which facts
caused the model's result. Causal attribution requires its own investigation.
For a teammate asking “why?”, start with the exact calculation and policy,
then state the relevant validation evidence and what remains unknown.

## Run a result example

With Wyat installed, run the [result example](../examples/results/main.py):

```sh
python examples/results/main.py
```

It uses fixed logits, loads no weights and executes no application actions.
It demonstrates candidate ID lookup and result ordering. Use
[Python evaluation](api.md#evaluate-labeled-cases-in-python) to assess the model
on cases relevant to your application.
