# Complete decision workflows

Use a result through the whole application path:

```text
application records → validated input → model/readout → complete distribution
    → explicit application policy → action or review → retained outcome/failure
```

The examples below use fixed logits and in-memory effects. They run without
weights, optional inference dependencies, network access or action credentials.
They teach application boundaries; they do not establish trained-model quality,
calibration, durable transactions or actual human review effectiveness. For a
trained Choice, complete [installation](install.md) and
[artifact acquisition](model-acquisition.md) first.

## Routing

Run [routing.py](../examples/workflows/routing.py):

```sh
uv run python examples/workflows/routing.py
```

The application prepares a request, reuses one fixture model, looks up the exact
selected ID in its allowed queues and records a toy queue effect. Selection alone
changes nothing. The example retains duplicate-ID preparation rejection, a
backend failure, malformed output and a model-selected review separately.
It blocks an unknown ID from a directly constructed, untrusted replay result.
Imported results also need producer and input identity checks; an ID allowlist
alone does not authenticate them.

One simulated action fails before its effect. A simulated operator explicitly
retries that retained action, and toy in-memory duplicate suppression prevents a
second write. This is not a general retry policy: an external action may succeed
before its acknowledgement fails and needs its own reconciliation contract.

## Prioritization

Run [prioritization.py](../examples/workflows/prioritization.py):

```sh
uv run python examples/workflows/prioritization.py
```

The pure Score interpreter produces an expected **zero-based rubric index**.
The example declares an application policy that sorts by that index, retains
full distributions, preserves input order at equal expected values and keeps an
invalid readout visible. Its split distribution has expectation 1.0 while assigning
very little probability to the middle level. Rounding it into a likely category
would lose that distinction.

This policy treats rubric steps as equally spaced for prioritization. A team
must assess whether that utility assumption fits its actual work. No native
Score weights are loaded by this example or enabled in `wyat.load`.

## Human review

Run [human_review.py](../examples/workflows/human_review.py):

```sh
uv run python examples/workflows/human_review.py
```

The pure Noul interpreter returns the weight attached to `true`. A named,
illustrative application policy accepts, rejects or sends a valid uncertain
result to a review record. A separate simulated reviewer resolves that record
before a toy effect occurs. Invalid logits instead produce an operational
failure record; no semantic decision is invented for them.

Real integration must record reviewer identity, unresolved work and eventual
outcome under its existing privacy and action-authorization policy. The fixture
does not establish reviewer reliability or queue capacity.

## Evaluate the policy

Run [policy_evaluation.py](../examples/workflows/policy_evaluation.py):

```sh
uv run python examples/workflows/policy_evaluation.py
```

The example freezes 12 synthetic labeled cases before execution. Labels remain
outside the model-facing question. Cutoffs are required application arguments,
not SDK defaults; the displayed `.8/.2` band is illustrative. The report includes
the full inventory, every status, false accepts/rejects, unresolved labels,
prediction coverage, autonomous coverage, uncertainty reviews, operational
failures and accept precision. Every ratio includes its numerator and denominator.
An empty denominator is unknown (`null`), not zero error.

For this fixture, 8/12 readouts are valid, 5/12 decisions are autonomous, 3/12 need
uncertainty review and 4/12 are operational failures. False accepts are 2/6 labeled
negatives; false rejects are 1/6 labeled positives. Unresolved cases remain in
those population denominators and are counted separately. Low false-reject rate
alone cannot imply useful acceptance when much of the population is unresolved.

Compare unrounded returned probabilities. The boundary fixture returns
`0.7999998480215554`, which displays as `0.800000` but remains below `0.8`.
Changing the menu can also change a Choice's relative probability without making
it a calibrated probability of correctness.

For actual threshold selection, freeze the policy, model/revision, menu, input
cohort, label meaning, failure treatment, error costs and review capacity. Use
representative independently labeled development data for policy fitting, then
untouched confirmation data for evaluation. Report language/source strata and
eventual review outcomes. Calibration needs a separate assessment of estimates
against observed outcomes; the [calibration research](https://proceedings.mlr.press/v70/guo17a.html)
explains this distinction. No empirical calibration result follows from these
hand-picked fixtures. Existing [Choice evaluation](api.md#evaluate-labeled-cases-in-python)
remains the published Choice prediction-level owner. Native typed inference
requires a separately admitted model recipe and evaluation.

## Run from an installed wheel

Wheels contain Wyat's runtime, not these example files. Copy a selected script
into your application, install the exact wheel as described in
[installation](install.md), then run it from that directory. The scripts need
only `wyat` and the Python standard library. They do not depend on checkout paths
or private research sources.

The focused [workflow tests](../tests/examples/test_workflows.py) assert action,
failure, distribution and denominator expectations. The
[agent contract fixtures](../tests/test_agent_contract.py) cover stable ties,
Unicode IDs, the misleading Score mean and Noul's true-ID lookup. The
[recovery checks](../tests/test_recovery.py) exercise line provenance and safe
HTTP rejection reasons. These are repeatable software checks; fresh-agent trials,
human usability and learned language quality remain distinct evidence.
