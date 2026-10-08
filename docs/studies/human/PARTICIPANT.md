# Participant packet

Use a supplied Python application directory outside Wyat's source checkout.
The researcher supplies the exact wheel path, runtime choice, supported artifact
path, frozen public docs and `fixtures.py`. Use those docs while working; save
your first complete program before execution, then retain errors and repairs.
Do not use an AI assistant, implementation/tests, web search or another person's
solution. All records are fictional. No task needs a real application action.

For tasks other than U2, U7 and U9, the researcher supplies a working base-wheel
environment. `fixtures.make_model(mode)` returns a fixed-logit demonstration
model. It is not trained inference. `fixtures.make_scorer(mode)` returns a
scorer with a public `calls` counter; retain this object if counting calls. These helpers
do not perform external actions. Finish the program and the explanation within
the card's cap. Ask the researcher to read the next assigned card.

## U1 — Explore without weights (10 minutes)

Run the scorer example from the supplied public examples. Save its output.
Explain what `query`, `context` and each candidate ID mean. What supplied the
numbers, what evidence does this run provide, and did it change a booking?

## U2 — First local decision (30 minutes)

In your empty application project, install the exact supplied wheel using the
declared runtime. Obtain or locate and verify the declared full local artifact
according to the supplied acquisition condition. Load it once. Use two allowed
IDs `cancel` and `create` with descriptions “Cancel the booking” and “Create a
booking”, instruction “Choose the requested action” and evidence “Cancel my
booking”. Run one decision and save its complete result plus the exact selected
ID. Print the corresponding application operation name from this mapping:
`cancel → cancel_booking`, `create → create_booking`. Do not run a real action.

Explain which installation, acquisition, verification, loading and inference
steps occurred; identify the runtime and artifact. Is an allowed result evidence
that the model is accurate or that a booking changed? What would you do if this
artifact were absent? Do not use a fixture or different artifact to hide failure.

## U3 — Dispatch by identity (10 minutes)

Use `fixtures.make_model("rank")` and candidates, in this order:
`review: Send to a reviewer`, `ship: Send to shipping`, `hold: Keep on hold`.
Instruction: “Choose the allowed next action”; evidence: “Fictional package”.
The fixture returns logits `[0, 2, 2]`. Record the result and implement a toy
in-memory dispatch table using the exact selected ID. Append the chosen ID once
to a local effects list, then print it. Explain what caused that append, what
selection alone did, how the exact tie is handled and what changes if the same
results arrive from an unauthenticated external producer.

## U4 — Batch reuse (10 minutes)

Retain one `fixtures.make_scorer("last")` and construct one `wyat.Model` from it,
with two independent requests:
first IDs `[wait, refund]`, second IDs `[keep, review, close]`. Give each ID an
appropriate description, instruction “Choose the permitted action” and fictional
evidence. In one batch call, print each complete result with its request index
and exact candidate IDs; print the scorer call count. Explain how you preserved
request/result and candidate/value association and where loading belongs in a
request handler. No action should run.

## U5 — Explain three result types (10 minutes)

Make a JSON-serializable note for each part using public APIs, then explain it
to a teammate. These are supplied logits, not a trained model's judgments.

1. Choice: ordinary request IDs `[review, ship, hold]`, logits `[0, 2, 2]` from
   `fixtures.make_model("rank")`. Retain original-order scores and probabilities,
   keyed access by ID, selected ID and ranking. Round-trip the note through JSON.
   Explain the tie, the relationship between ranking and score order, what a
   probability means and whether any action happened.
2. Native typed Score: rubric `[Can wait, This week, Today]`, logits
   `[0, -1000, 0]`. Use the pure `wyat.tasks` interpretation path. Show the full
   distribution and the expected value with its unit. Explain whether the middle
   description was selected or is likely; can the mean alone justify a priority?
3. Native typed Noul: “Does the message concern billing?”, criteria inserted in
   order `true: Concerns billing`, `false: Does not concern billing`, logits
   `[0, 1.3862943611198906]`. Show the full distribution and true probability.
   Apply this application rule to the unrounded result: true at `p_true >= .8`,
   false at `p_true <= .2`, otherwise review. Explain the chosen policy outcome,
   who owns the cutoffs and whether this probability is calibrated. Explain how
   this task differs from Python `None`, JSON `null` and a missing answer. What
   do omitted or null Noul criteria mean? Does ordinary Kai loading supply native
   Score and Noul inference?

## U6 — Recover without concealing failure (10 minutes)

Retain and repair both attempts:

1. Build a request with two candidates using the same ID `wait`. Save the first
   exception. Correct the IDs to `wait` and `cancel` without changing their
   descriptions. Use `fixtures.make_model("last")` and print the result.
2. Use `fixtures.make_model("length")` with instruction “Choose”, IDs
   `[wait, cancel]`, and context `"Cancel booking." + " " * 90`. Save the
   rejection and its reason. This fixture accepts context of at most 80
   characters. Repair only the extra whitespace in this fictional evidence,
   preserving its meaning, then retry explicitly and retain both records.

Explain validation versus model-limit rejection, who chose the repair, and why
an SDK should not silently truncate or fabricate a successful decision. No real
model's context limit is established by this fixture.

## U7 — Repair the runtime (10 minutes)

Use the researcher-supplied separate environment, which has the base wheel but
no Torch runtime, and the complete verified local Choice artifact. Attempt
`python -c 'import torch'` in that environment and save the error. Use the
installation docs to install the dependency extra for the declared Torch CPU
runtime into this same application environment. Verify the import, then load
the artifact and produce one allowed Choice result. Preserve the first failure.
Explain which extra fixed which boundary, why changing another environment or
using a fixed scorer would not complete this task, and why installing the package
alone does not provide weights. Do not silently select another backend.

## U8 — Supply and distinguish scorer behavior (10 minutes)

Implement a public scorer returning finite raw logits in request/candidate order.
Exercise two differently sized menus in a batch and print their results. Then
exercise `fixtures.make_model("malformed")`, `fixtures.make_model("length")`
with the over-limit U6 context and `fixtures.make_model("backend")` on a valid
request. Record each exception type and distinguish broken output, rejected
input and the original backend execution failure without replacing them with
semantic decisions. Explain who owns batching, resource limits and execution.

## U9 — Serve the same meaning (10 minutes)

In the supplied server-enabled environment, make a factory serving
`fixtures.make_model("slow")` through the public HTTP integration. Run one
worker on localhost. Send a Choice request for `[wait, cancel]`, retain its
response, and compare the selection and score associations with Python. Send
two overlapping requests and retain both responses. Explain initialization per
worker, admission behavior, what additional workers would load, and whether the
result executed an application action. Stop your server when finished.

## U10 — Retain a failed prediction (10 minutes)

Use the existing `wyat.evaluation` evaluator with `fixtures.make_model("evaluation")`.
Give each request IDs `[wait, cancel]` and instruction “Choose”. Freeze these
three fictional labeled cases before execution:

| Case ID | Context | Accepted ID |
| --- | --- | --- |
| case-1 | first | wait |
| case-2 | second | cancel |
| case-3 | failure | cancel |

Save the full report and its individual statuses/errors. Explain accuracy and
prediction coverage with numerator and denominator; explain how the failed case
is retained and how an operational error differs from an uncertain semantic
answer or a model-selected review action. Do not implement a new evaluator.
