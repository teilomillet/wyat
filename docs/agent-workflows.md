# Complete decision workflows

For a plain-language introduction, [watch a tiny coding agent](embedded.md)
ask Wyat what to do, run the chosen function, and check what happened.

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
python examples/workflows/routing.py
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

## Evaluate the decisions

Use [Python evaluation](api.md#evaluate-labeled-cases-in-python) with labeled
cases and the same loaded model. Keep labels outside the model-facing request.
The report retains failed calls in the accuracy denominator and reports coverage
separately. Run the [fixed-scorer example](../examples/evaluation/main.py):

```sh
python examples/evaluation/main.py
```

This example includes a correct prediction, a wrong prediction and a failed
call. Its synthetic logits demonstrate report behavior; they do not measure a
trained model's quality. Evaluate an application's routing policy and action
failures separately from the model's choice accuracy.

## Run from an installed wheel

Wheels contain Wyat's runtime, not these example files. Copy a selected script
into your application, install Wyat as described in [installation](install.md),
then run it from that directory. The routing and evaluation scripts need only
`wyat` and the Python standard library.

A production integration owns durable action records, duplicate suppression,
retries, review outcomes and privacy. The in-memory examples illustrate those
boundaries without implementing a durable transaction system.
