# Decisions inside real application workflows

Ten small examples show where a Wyat decision fits between your application's
input and its next step. Each file contains a reusable function accepting a
loaded `wyat.Model`, a fictional input in `run(model)`, and a command-line entry
point. Load once in your application and call the function for each item.

The examples cover tasks related to the measured strengths discussed below.
They are original teaching fixtures, not questions copied from JDI or DMBench.

## Run an example

Use Python **3.12 or newer** and [uv](https://docs.astral.sh/uv/getting-started/installation/).
Run the commands from the root of a source checkout containing this folder.
Installed wheels provide `wyat`, but do not include the checkout's `examples/`.

```sh
uv sync --extra inference
uv run python -m examples.use_cases.benchmark_workflows.07_application_conditions \
  --model teilomillet/wyat-1-800m
```

The first model load downloads and verifies about **1.82 GB** of weights.
Later loads reuse the verified cache, and inference works offline. To download
ahead of time, run `uv run wyat download`; you can also pass a local model
directory. See [model acquisition](../../../docs/model-acquisition.md) and
[installation in your own project](../../../docs/install.md).

On Apple Silicon, use `--device mps` with PyTorch, or install the `mlx` extra
and pass `--backend mlx`. All examples accept the same `--model`, `--backend`
and `--device` flags as the other use cases. A local model directory also works.
`--demo` uses the existing equal-score stand-in: it shows the plumbing and
does not demonstrate model quality.

To run another example, change the last module name in the command:

| Example | Input → decision → application use |
| --- | --- |
| [01 Group statement](01_group_statement.py) | Opinions and proposed statements → choose a draft → collect votes on that draft. |
| [02 Phishing triage](02_phishing_triage.py) | Message and usual domain → probability of phishing → select a review queue. |
| [03 Review tags](03_review_tags.py) | Customer review → independent aspect/sentiment flags → return a row for an analytics table. |
| [04 Product relevance](04_product_relevance.py) | Search query and product → exact/substitute/complement/irrelevant → place it in the correct shop section. |
| [05 Code output review](05_code_output_review.py) | Code, expression and proposed outputs → choose an output → prepare a test for verification. |
| [06 Chess move](06_chess_move.py) | Position and all legal moves → select a move → validate and apply it to a local board. |
| [07 Application conditions](07_application_conditions.py) | Supplied rules and an application → failed/unknown/satisfied prerequisites → request missing evidence or prepare review. |
| [08 Service handoff](08_service_handoff.py) | Ticket, payment record and routing rules → is this team's remit established? → prepare its handoff or request routing review. |
| [09 Updated request](09_updated_request.py) | Request history and proposals → score against current requirements → rank proposals for fulfillment review. |
| [10 Incident priority](10_incident_priority.py) | Operational impact and a rubric → priority level → prepare the appropriate incident queue. |

Chess additionally uses `python-chess` to generate and enforce every legal move:

```sh
uv run --with python-chess python \
  -m examples.use_cases.benchmark_workflows.06_chess_move \
  --model teilomillet/wyat-1-800m
```

## Integrate the pattern

### Sample data and your own files

The [sample-inputs.json](sample-inputs.json) file contains the complete input
arguments for all ten examples. It is a data-only view of the fixtures embedded
in the scripts; tests check that both paths produce identical model requests.
It contains fictional teaching records, not the full JDI or DMBench datasets.

Three files serve different purposes:

| File | Contains | Sent to the model? |
| --- | --- | --- |
| [sample-inputs.json](sample-inputs.json) | Opinions, messages, records, rules and proposed choices. | Only the selected example's input. |
| [evaluate.py](evaluate.py) | Authored expected application results. | No. These belong to the evaluator. |
| [recorded-run.json](recorded-run.json) | Actual model outputs, observed failures, and run identity. | No. These are results from the earlier run. |

Run the bundled prerequisite data through the file-input entry point:

```sh
uv run python -m examples.use_cases.benchmark_workflows.run \
  application_conditions \
  --input examples/use_cases/benchmark_workflows/sample-inputs.json
```

The default is the real 800M model. Add `--demo` for a quick check without
weights; output is explicitly marked `"mode": "demo_equal_scores"`. Add
`--backend mlx` or `--device mps` on Apple Silicon. The chess workflow also
needs `uv run --with python-chess`.

To try your own case, create `my-application.json` containing just the selected
function's input fields, for example:

```json
{
  "application": "Lina est majeure et sa cotisation est payée. Le dossier ne permet pas encore de vérifier son initiation à la sécurité.",
  "rules": "L'accès exige la majorité, une cotisation à jour et une initiation à la sécurité attestée."
}
```

```sh
uv run python -m examples.use_cases.benchmark_workflows.run \
  application_conditions --input my-application.json > decision.json
```

For a different workflow, change the name and use its fields from the sample
file. `--help` lists all ten names. Missing, extra or incorrectly typed fields
are rejected before loading or downloading weights. JSON input contains data
only: the code-output example does not execute code supplied in that file.

### Read a result

The **recorded built-in prerequisite example** returned this shortened result:

```json
{
  "level": 1,
  "next_step": "request_missing_evidence",
  "decision": {
    "selected": "1",
    "probabilities": [0.038883, 0.947159, 0.013958]
  }
}
```

`level: 1` means no requirement is established as failed, but evidence is
missing. `next_step` is the application policy's recommendation. The three
probabilities follow **rubric order 0, 1, 2**, rather than ranked order, and
are not calibrated guarantees. Your own input may produce a different result.
The full output retains the model ID, raw scores and ranking for inspection.

### Put the function in your application

Copy the reusable function **and the constants it uses**, such as `RUBRIC`,
`NEXT_STEPS`, `RELATIONS` or `SECTIONS`. Keep its `import wyat` and standard
library imports; omit `run(model)` and the command-line block. The
`examples.use_cases.common` import belongs to checkout-specific CLI scaffolding,
so do not copy that import into your application.

In your own uv project, install the runtime with `uv add wyat --extra inference`.
Here is a complete small worker you can save as `worker.py`; it reads the custom
JSON above and prints an application record:

```python
import json
from pathlib import Path

import wyat

model = wyat.load()  # once, when the worker starts
rubric = [
    "A mandatory condition is established as unmet.",
    "No condition is known to fail, but some evidence is missing.",
    "Every mandatory condition is established as satisfied.",
]


def prepare_application(application, rules):
    decision = model.decide(
        wyat.Request.score(
            "What do the available records establish about eligibility?",
            rubric,
            context={"application": application, "rules": rules},
        )
    )
    next_step = {
        "0": "review_unmet_condition",
        "1": "request_missing_evidence",
        "2": "ready_for_final_review",
    }[decision.selected]
    return {"next_step": next_step, "decision": decision.to_dict()}


inputs = json.loads(Path("my-application.json").read_text(encoding="utf-8"))
print(json.dumps(prepare_application(**inputs), ensure_ascii=False, indent=2))
```

Run it from your application directory with `uv run python worker.py`.
For a long-running worker, keep the loaded `model` across items and call
`prepare_application(**inputs)` for each record. The other nine examples expose
the same pattern: a loaded model plus ordinary Python function arguments.

Your worker can put that returned record in its normal work queue. The example
functions return dictionaries or update an in-memory chess board; they need no
email, ticketing, payment, or shop credentials. Your application owns the actual
queue write and any action taken afterward.

Three result types answer different application questions:

- **Choice:** use `selected` to look up an application-owned ID. The product
  example maps the chosen relation to a shop section.
- **Noul:** use `probability_true` for one exact proposition. The handoff example
  sends an uncertain or rejected destination to review; it does not assume that
  rejecting one team establishes another team's responsibility.
- **Score:** `selected` is the most likely rubric level, while
  `expected_ordinal_value` averages the ordered levels. The incident example
  uses a modal level for a discrete queue; the proposal example uses expected
  values to rank candidates under the same rubric and request history.

Probabilities are model outputs, not calibrated guarantees. The 0.8 and 0.5
thresholds in these scripts are explicit illustrative application policies.
Choose them using labeled examples from your workflow. Model errors propagate
to the caller, so the normal worker retry or review policy can handle them.

## What each fixture is meant to demonstrate

These are authored expectations, fixed before the recorded model run. They are
kept outside the model-facing input in [evaluate.py](evaluate.py).

| Example | Authored expectation |
| --- | --- |
| Group statement | Propose night-time car restrictions with continued bicycle and emergency access. This is a constructed fit to stated preferences; actual group preference needs votes. |
| Phishing triage | Flag the urgent password/SMS-code request on a separate domain for review. |
| Review tags | Positive food, negative service; neither positive service nor negative ambience is stated. |
| Product relevance | The camera battery belongs in accessories, rather than replacing the requested camera. |
| Code output | Propose `ba:fed:hg`; independently verified using the trusted fixture function. Incoming code is never executed by this example. |
| Chess move | `h5f7` is the unique mate in one, independently checked with `python-chess`. The model receives 43 legal UCI moves without check/mate notation hints. |
| Application conditions | Missing safety-training evidence calls for more evidence, rather than establishing a failed prerequisite. |
| Service handoff | The validated merchant record establishes the network's billing remit. |
| Updated request | Prefer fourteen vegetarian meals over ten, preserving the unchanged pickup time. This checks ordering, not exact rubric calibration. |
| Incident priority | One whole team blocked with no workaround maps to level 3. |

Run the small diagnostic, loading the model once:

```sh
uv run --with python-chess python \
  -m examples.use_cases.benchmark_workflows.evaluate \
  --model teilomillet/wyat-1-800m > workflow-results.json
```

The JSON retains every example's returned record, expected application result,
source hash, and any failure. Optional-dependency and inference failures remain
in the ten-example denominator. Progress and the demo banner go to stderr,
leaving stdout as JSON. A trained-model run exits with status 1 if any check
fails; the complete JSON is still written. Demo runs have `quality_valid: false`
and are only plumbing checks.

### Recorded first run

The [recorded output](recorded-run.json) is the first CPU run on 18 October 2026
using the pinned public 800M weights. Inputs and expectations were frozen before
that run. All ten workflows executed; **seven matched their authored application
expectation**. Inputs and labels were not tuned after observing the results.

| Fixture | Observed result |
| --- | --- |
| Group statement | Selected `night_access`, then prepared collection of group votes. |
| Phishing triage | Selected the suspected-phishing review queue. |
| Review tags | Found positive food and negative service, but also falsely flagged negative ambience: `P(true) = 0.579`. |
| Product relevance | Put the battery in the accessories section. |
| Code output | Proposed `hg:fed:ba` with probability 0.899; the actual output is `ba:fed:hg`. The proposed test needs verification. |
| Chess move | Played legal `c4d3`, missing the available mate `h5f7`. Legality checks alone cannot establish move quality. |
| Application conditions | Requested missing evidence. |
| Service handoff | Prepared handoff to network billing. |
| Updated request | Ranked the updated quantity above the old quantity. |
| Incident priority | Prepared the team-incident queue. |

This ten-fixture result describes these particular examples and application
policies. It is not an estimate of accuracy in a deployed workflow. The receipt
preserves all outcomes, model/weight identity, source hashes, and runtime details.

## Benchmark connection

The following are previously measured comparisons of **wyat-1-800m versus
D3-Lite**. They motivate the examples; the new fixtures do not reproduce those
benchmarks or establish a fresh head-to-head result. JDI figures come from
[Wyat's public submission](https://huggingface.co/datasets/teilomillet/wyat-1-800m-decision-index-0.3/blob/a0b54dfbd3a899af9af6c1bfc97cd3655151ce16/runs/wyat-1-800m-public03/scores.json)
and the [public board data](https://multimodalart-jev-decision-index.static.hf.space/data/index.json).

| Related example | Evaluation and metric | Wyat | D3-Lite | Population |
| --- | --- | ---: | ---: | --- |
| Group statement | JDI Habermas Machine, preference accuracy | 42.18% | 37.95% | 1,676 |
| Phishing triage | JDI PhishNChips, accuracy | 57.45% | 54.85% | 2,000 |
| Review tags | JDI ACOS, per-review F1 on a 0–100 scale | 8.05 | 6.45 | 400 reviews / 1,565 requests |
| Product relevance | JDI Amazon ESCI, macro-F1 on a 0–100 scale | 42.72 | 41.28 | 5,000 |
| Code output | JDI CRUXEval, accuracy | 34.56% | 33.68% | 570 |
| Chess move | JDI ChessBench, accuracy | 8.60% | 8.34% | 5,000 |
| Application conditions | DMBench French v4 Conditions, modal Score accuracy | 30/40 | 24/40 | 20 paired families |
| Service handoff | DMBench French v4 Routing, Noul accuracy | 33/40 | 31/40 | 20 paired families |
| Updated request | DMBench French v4 Updating, expected-score pair ordering | 10/12 | 8/12 | 12 pairs |
| Incident priority | DMBench French v4 Escalation, normalized expected-score MAE ↓ | 0.1211 | 0.1249 | 40 questions / 20 paired families |

DMBench is an authored development panel. The escalation lead is on average
scoring error, not modal accuracy. These are observed task-level margins, without
a demonstrated significance claim; D3-Lite leads the overall results. In
particular, chess is close to chance, CRUXEval is below its chance baseline, and
ACOS has a low absolute F1. Those three examples teach integration with
independent checks, rather than demonstrating a strong domain solver. JDI
private evaluation and maintainer latency eligibility remain pending.

For the other application patterns, see the [basic](../README.md),
[expert](../expert/README.md), and [integration](../integrations/README.md) examples.
