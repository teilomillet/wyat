# Researcher protocol

Status: prospective; no human trials conducted. This is a single-design
acceptance study, not a comparison showing that Wyat is best or faster.
This packet's instruction language is English. Its rates cannot establish
human understanding in another language. A localized study needs speaker-reviewed
instructions and documentation, its own language stratum and a new freeze;
preserve protocol keys and exact application IDs during translation.

## Prepare once

1. Select an exact locally built wheel, Python version, OS, uv version and
   supported full local Choice artifact. Follow the frozen installation and
   acquisition documents. Independently verify the artifact before participant
   sessions. Keep acquisition conditions explicit: either fresh download or a
   verified supplied local copy. Do not combine their times as if equivalent.
2. Prepare an application directory outside the checkout for each participant.
   The participant can read only their currently revealed task card, `fixtures.py`, the exact wheel
   and a frozen copy of these public docs: install, model-acquisition, api,
   decision-tasks, explaining-decisions and agent-workflows. Include
   the acquisition verifier and public workflow examples linked by those docs.
   Do not expose this protocol, observer answers, tests, private artifacts or
   earlier participant programs. The freeze tool separates the public cards;
   reveal one at a time, keeping later cards with the observer. Record whether filesystem isolation is enforced
   or merely instructed. Do not use private training data.
3. Prepare a separate base-wheel-only environment for U7, with no Torch runtime,
   and a complete verified artifact available. Confirm `import torch` fails
   there before assigning U7. Do not remove dependencies from the U2 environment.
   Prepare server dependencies for U9 in the integrator environment. Pin the
   platform, dependency versions and lockfile before enrollment.
4. Copy `CONFIG-TEMPLATE.json` to `study-config.json` and fill the actual
   identities, assignment roster, environment and assistance rules. Resolve
   every null before freezing. Freeze
   using `python freeze.py --config study-config.json --out /absolute/new/session`.
   The tool hashes specified files and copies the public packet; it does not
   install packages, download or load models. Include the complete documentation
   packet and environment locks in `files_to_pin`. Hash all nine artifact files
   under `artifact_files`. Keep the generated private `FREEZE.json` with the
   researcher. Give participants the prepared public workspace, not that file.
5. A researcher independently smoke-tests the assigned setup and supplied
   fixture examples in equivalent disposable environments. Record this as
   preparation, never a participant success. If the actual artifact/runtime is
   unavailable, keep U2 blocked; fixture completion cannot replace it.

The template deliberately has no wheel, artifact or environment identity.
Prepare and independently verify those exact inputs before filling it. Freeze the
actual public docs, examples, environment locks and preparation evidence for this
study; a historical development receipt cannot qualify a new study. The helper
pins the researcher-specified files and prepares cards. It does not copy the wheel,
artifact, docs or locks into participant workspaces: the researcher supplies
those exact frozen files separately. No human observations are included here.

Pilot: the ready initial proposal uses six participants per persona, 18 total,
and the same assignments. Pilot findings are exploratory, never an acceptance
rate claim. Any pilot-driven repair creates a new freeze and is disclosed. Pilot
participants do not join confirmation. Optional confirmation proposal: at least 30 enrolled and
assigned people per persona, 90 total, with no optional stopping after a good
result. Enrollment is a proposal, not an assumed available resource or a release
prerequisite; nobody is recruited by this packet. Participants know basic Python; beginners have not used Wyat, integrators
have shipped a Python application, scorer authors have implemented batch model
interfaces. Use `enrollment-template.csv` to record only anonymous ID, persona,
prior Wyat exposure (yes/no), eligibility and assignment; prior exposure
does not qualify for the fresh confirmation cohort.

## Assignments and order

| Persona | Assigned tasks | Odd ID order | Even ID order |
| --- | --- | --- | --- |
| Beginner B001–B030 | U1–U7 | U1,U2,U3,U4,U5,U6,U7 | U1,U2,U6,U5,U4,U3,U7 |
| Integrator I001–I030 | U2–U7,U9,U10 | U2,U3,U4,U5,U6,U7,U9,U10 | U2,U6,U5,U4,U3,U7,U10,U9 |
| Scorer author S001–S030 | U4,U5,U6,U8 | U8,U4,U6,U5 | U8,U5,U6,U4 |

Use these exact 19 task/persona cells in this study. Increasing enrollment uses
the same assignment rule and must be declared before confirmation begins.
Orders balance some within-study practice effects; they do not create a
comparative design or independent observations from repeated tasks. Split long
sessions at task boundaries if required, recording the predeclared schedule.
The per-task clocks stop between tasks. Do not omit later tasks after a failure.

## Conduct a session

Read only: “Use the supplied documentation and fictional inputs to complete
each task. Please tell me what you are trying to do. I cannot provide technical
help during a task. Save your first complete program before running it. You can
stop at any time. Your work is evaluated against the stated task.”

No agent assistance, search engines, implementation/test sources, grader access
or other participant solutions. Allow only the frozen public docs, standard
Python documentation supplied in the manifest and ordinary package-manager
dependency resolution. Those choices are study restrictions, not product rules.
Reading documentation is allowed and timed. The observer may repeat task wording
verbatim and resolve an environment outage without suggesting a solution.
Technical hints count as assistance and make that task an unassisted failure,
even if the eventual answer is correct. Record what was said.

Start a monotonic clock when the card is revealed. U2 has a 1,800-second cap,
including installation and acquisition; every other task has a 600-second cap.
Record separately U2 install, acquisition/verification, model-load and warm
decision phases, all within its total. Record the first complete program before
its first run, then stdout/stderr, edits and the final program. Retain each
error and attempt. Immediately after the first run, ask the card's explanation
questions without providing corrections and retain that first explanation.
If the participant repairs the program, collect its final explanation too.
The explanation is part of the same clock and pass.
Stop at the cap. A failed or abandoned task receives the declared cap for T95,
while the actual observed duration is retained separately. A task not started
after assignment remains missing and receives its cap.

Use `observations-template.csv`, one row per assigned task, and the observer
key. Store anonymous IDs and fictional-program artifacts only. Use relative
paths to retained programs/output and concise error categories; scrub home paths,
credentials or unrelated personal content before retaining terminal excerpts.
Do not collect a name, employer, private input or unrestricted recording. Leave
unavailable values blank; blank is unknown, not zero. After each task's clock
ends, ask “Which step took unexpected effort, if any?” and retain the answer
as an optional qualitative note. It is not a correctness grade or a measured
preference over another library. Never replace the first
attempt with the final repaired program. No human rate can be entered now.

CSV conventions: pass fields are `1` or `0`; blank means unavailable. `status`
is one of `completed`, `timeout`, `abandoned`, `missing`, `infrastructure_failure`.
`assistance_count` is an integer count of technical hints, including prohibited
tool assistance. Durations are seconds on the monotonic clock; retain the raw
duration even when a failed task is charged its cap. U5 subtask fields are blank
for other tasks. Preserve each observer's execution/interpretation grades and
their adjudicated result rather than overwriting a disagreement. Store detailed
grading in the referenced explanation/output record if a CSV cell is too small.

## Grade and analyze as frozen

`unassisted_pass = execution_pass AND interpretation_pass AND status=completed
AND assistance_count=0`. U5 requires **all** Choice, Score and Noul subtasks;
also report each subtask's rate diagnostically. Two observers independently
grade the preserved output and explanation before seeing aggregates. Preserve
their original grades and disagreements; adjudicate against the frozen key
without changing it. Missing evidence fails the relevant expectation. Code
being plausible is not evidence that it ran.

For each of the 19 cells, rate = unassisted passes / assigned people. All timeout,
abandonment, missing output, assisted repairs and infrastructure failures remain
in the denominator. Report infrastructure failures as an additional category;
do not silently exclude them or start fresh after an observed failure. Before
assignment an unavailable environment delays the entire affected cell rather
than supplying a pass. After assignment it is an observed failed attempt.

Q is the equally weighted mean of the 19 cell rates, not a pooled rate that
overweights participants assigned more tasks. Report persona means separately.
The observed-target gate is Q ≥ .90 and every cell ≥ .80. Label it explicitly
as a sample result; it alone does not establish a population rate. For each cell,
report a 95% Wilson interval on the assigned-person denominator:

```
z = 1.959963984540054; p = successes/n
center = (p + z*z/(2*n)) / (1 + z*z/n)
half = z*sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
interval = [center-half, center+half]
```

For uncertainty in Q, resample complete participants with replacement within
each persona, recompute all 19 cell rates and Q, 5,000 times with seed 20261008;
report percentile 2.5/97.5 intervals. Keep a person's tasks together. If all
participants in a persona have identical outcomes, its nonparametric bootstrap
has no variation; report that boundary limitation rather than interpreting a
collapsed interval as certainty. Do not make a population-target claim from a
degenerate Q interval. These nominal intervals assume reasonably independent
recruited people; shared teams or
recruitment dependence must be disclosed. Do not call the population target
supported unless the Q interval lower bound is ≥ .90 and every cell's Wilson
lower bound is ≥ .80. This stricter interpretation can remain inconclusive even
when the observed-target gate passes. No claim of formal joint confidence or
best-library superiority follows from these intervals.

T95: report nearest-rank (`ceil(.95*n)`) elapsed seconds per cell, charging each
unsuccessful task its cap, plus U2 phase distributions and the complete journey.
Do not trade lower correctness for speed. A single-design study establishes no
speedup. K: two independent annotators complete `concept-ledger-template.csv`
from the complete task paths; count each necessary concept once per task and
once for the union of its persona journey, retain disagreements and resolve
before reporting counts. K is a documented proxy, not measured human cognition.

Include a result table with all 19 cells, passes/assigned, failure categories,
Wilson intervals, T95, U5 subtask rates, persona means and Q interval. Show first
attempt and eventual completion separately. Keep any qualitative comments as
attributed anonymous observations, not numerical preference evidence. Report
enrollment, exclusions made before assignment and all post-assignment losses.
If a threshold fails or evidence is missing, retain that finding and redesign
only through a new protocol/version. Do not revise success after seeing results.

Re-running after implementation changes requires new participants and a new
freeze. Comparing versions requires predeclared equivalent fresh task variants,
counterbalanced design order, a practically meaningful difference and an analysis
of participant dependence. This acceptance packet supplies no such comparison.
