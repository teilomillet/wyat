# Validate decisions on real, independently labeled inputs

This packet freezes the measurement method and provides blank preparation forms.
It does not contain a real cohort, predictions, calibrated probabilities or an
approved deployment threshold. Fill a **new version** of the run manifest before
confirmation. A null prerequisite is unknown and cannot pass a gate.

Typed benchmark execution is pending separate benchmark integration. This
publication contains preparation forms, not that evaluator or an executable
benchmark CLI. Reuse the benchmark owner's reviewed integration once available;
do not build a second evaluator. Human library
usability belongs to the separate human acceptance protocol. Correct Unicode
transport and fixed-logit arithmetic do not establish multilingual model quality.

## Preparation and ownership

1. The application owner declares supported languages, actual use population,
   task families, complete menus/rubrics and costs of mistakes and review.
2. The data owner supplies permitted sources and binds each translation,
   paraphrase, repeated menu order and derived question to its source ancestry
   owner. Split **owners**, not rows, between development and untouched
   confirmation. Retain the original source and target languages privately.
3. Two independently competent speakers label every question, without model
   predictions. Record language competence and independent first labels, then
   have a third reviewer adjudicate disagreements against frozen rubric examples.
   Annotators must not see the other annotator's first label. Unresolvable cases
   remain in the source inventory with a declared disposition before inference;
   never remove a case because the model got it wrong.
4. Use `annotations.blank.csv` and `cohort.blank.json` as preparation forms.
   They deliberately are **not** executable benchmark inputs: blank answers
   must not be silently converted into false, zero or an empty acceptable set.
   After the separate benchmark integration is available, the data owner
   materializes the final accepted labels into its reviewed typed-case schema.
   These preparation forms do not imply that schema is currently installed.
5. Freeze `run-manifest.blank.json` in a new version after all null fields are
   resolved. Bind exact source, reviewed contract successor, installed wheel,
   weights, tokenizer, recipe, runtime, device, data, labels, complete question
   and option order, predictions and baseline identities. Any later model,
   threshold, calibration or recipe change needs fresh owner-held confirmation.

Do not repurpose protected TRAIN, DEV or reserved datasets. Their access and
evaluation owners retain authority. This packet admits no artifact, releases no
private labels and starts no model, training or inference job.

## Execution prerequisites

The independently reviewed typed benchmark integration, its supported CLI,
schema and source/contract identities must be supplied by the benchmark owner
before this study can run. This packet does not advertise benchmark commands
available in the current publication. `run-manifest.blank.json` records that
prerequisite explicitly; all blank fields remain unknown.

The existing [policy illustration](../../../examples/workflows/policy_evaluation.py)
can be read or exercised separately without trained weights. It supplies
fictional arithmetic, not model-quality observations. Its accounting remains
the application-policy reference below.

Once the benchmark integration and authorized data/runtime preparation exist,
retain complete public request records and saved raw predictions. Successful
predictions must have complete raw logits, exact `option_ids` order and a
reviewed canonical recipe. Missing, invalid, failed and unsupported questions
remain in every relevant denominator; never normalize the menu, fabricate an
answer, retry a failure out of the report or reinterpret binary Choice as Noul.
A later retry is a separate recovery observation linked to its original attempt.
The scorer must not load a model.

## Frozen measurement rules

| Task | Principal result | Required full-denominator measure |
| --- | --- | --- |
| Choice | Exact selected ID; first input-order option wins ties | Acceptable-ID accuracy / error |
| Score | Expected zero-based ordinal index and full distribution | MAE divided by `L-1`; ranked probability score divided by `L-1` |
| Noul | `true` probability by ID | Binary Brier loss |

Each failed question gets worst bounded loss 1, an evaluation penalty rather
than a model probability. Every task reports statuses, coverage, valid-only
likelihood and complete failure-inclusive losses. Score's index spacing is an
evaluation convention, not a claim that rubric levels are physical equal units.
Multiple acceptable Choice IDs are not a fabricated uniform target distribution.
Do not combine the three tasks into a pooled quality headline.

Report every frozen language × task × family × menu-width cell, total assigned
owners/questions, and paired owner-bootstrap intervals (2000 resamples, seed 17)
from the reviewed benchmark integration after it is available. The same source owner must be resampled together.
Repeated questions/translated variants add observations, not independent owners.
Cells with too few owners retain unknown uncertainty; they cannot claim a
supported language. Owner-bootstrap uncertainty does not cover annotation error,
model seeds or source-distribution shift.

Numeric quality/error/cost limits and minimum owner counts must be set by the
application and data owners **before** confirmation in the manifest. They remain
null in this blank packet because no application costs, cohort or justified
limits were supplied. Any failing, missing or insufficiently supported required
cell rejects the intended claim; an aggregate cannot rescue it. When comparison
is intended, freeze the baseline and required paired gain in advance.

## Application policy is a separate decision

Reuse the accounting in `examples/workflows/policy_evaluation.py`, with the
application's independently labeled Noul population and frozen cutoffs. Its
twelve illustrative fixtures are not a threshold-fitting cohort. First preserve
each row's original status and canonical `probability_true`; apply `accept >=
accept_min`, `reject <= reject_max`, otherwise uncertainty review. Compare the
unrounded canonical value. A displayed 0.800000 can still be below 0.8.

Always report these counts and fractions separately, including numerator and
denominator: false accepts / all negatives; false rejects / all positives;
uncertainty reviews / all questions; operational failures / all questions;
autonomous actions / all questions; erroneous autonomous actions / autonomous
actions; true accepts / accepts. Report unresolved positives and negatives, too.
An empty denominator yields null, not zero or a pass. Entirely empty inventories
are rejected. Operational referral is not semantic uncertainty review and earns
no correctness credit. Human resolution and executed effects require separate
observations; pending review has not succeeded.

Fit thresholds only on owner-separated development data using declared error,
review and operating costs. Freeze them before one confirmation evaluation;
publish the development search ledger and retain all failed attempts privately.
No optimum is claimed by the fixed-logit example.

## Probability evidence

Brier loss measures probabilistic error; it alone does not establish calibration.
For valid Noul rows, a registered secondary analysis may fill
`reliability.blank.csv`: ten fixed intervals `[0,.1), …, [.9,1]`, per language
and question family, with bin counts, distinct owners, mean predicted `true`
probability and empirical true fraction. Empty bins have null means. Disclose
valid-prediction coverage and all operational failures beside that table; do not
invent probabilities for unsuccessful questions. No table has been filled here.

Register that analysis with the existing evaluation owner before confirmation;
the planned typed analysis provides Brier and likelihood; reliability
diagrams or ECE require separately registered support from that owner. This packet introduces no substitute evaluator or calibration
implementation. Reliability diagrams are diagnostic and depend on binning and
sample support. Temperature scaling is a possible development-only calibration
experiment, not an automatic default or proof: it changes the probability path
and requires its own frozen artifact/readout qualification and confirmation.
See [Guo et al., ICML 2017](https://proceedings.mlr.press/v70/guo17a.html).

Actual model quality, multilingual transfer, calibration, selected thresholds,
human review outcomes and deployment suitability are **unknown** until the
corresponding admitted, independently labeled observations exist.
