# Use a loaded Wyat model with DMBench

Wyat owns model loading and inference. DMBench owns benchmark imports, case
schemas, provenance, groups, expected answers and quality scoring. Wyat's
`benchmark_pair` is an engine performance and parity experiment; use DMBench
for general benchmark quality evaluation.

Install both projects in the same environment. DMBench is optional; importing
Wyat or its adapter does not import DMBench. The adapter imports DMBench when
`as_dmbench` is called and reuses the already-loaded model for every request.

```python
from dmbench import evaluate
from dmbench.io import load_cases

import wyat
from wyat.interop.dmbench import as_dmbench, as_dmbench_batch

model = wyat.load("/local/kai-bundle", device="cpu")
cases = load_cases("/local/dmbench-dataset")
report = evaluate(
    cases,
    as_dmbench(model),
    batch=as_dmbench_batch(model),
    batch_size=8,
)
```

The default adapter recipe, `dmbench-verbatim-choice-v1`, places DMBench's
`Request.text` verbatim in the Wyat query and
leaves context empty. Each supplied choice becomes a candidate with the exact
same ID, description and order. DMBench task names identify benchmark panels;
the Wyat request uses `task="choice"` to identify its inference contract.
Source IDs, groups, splits and expected answers are never placed in the model
input.

`as_dmbench` preserves the scalar callable boundary. `as_dmbench_batch` calls
the loaded model's native `decide_many` once per ordered chunk; neither helper
spawns concurrent model calls. They share the same request and prediction
conversions. Empty tuples return empty tuples without inference. A choiceless
member rejects its whole chunk before the model is called. DMBench retains
chunk failures and records the true batch wall time; it does not fabricate
per-item timing by dividing the batch duration. Omit `batch` and `batch_size`
for scalar evaluation.

The default recipe adds no serializer. The selected Wyat backend alone owns
its prompt recipe. A benchmark text may contain JSON, embedded instructions,
context or choices; the default recipe does not parse, strip or reconstruct
that text. Benchmark authors select other projections explicitly in their
model factory.

The opt-in `dmbench-native-choice-v1` recipe accepts only the explicit
`jevbench/choice` task and DMBench's native `question.type="choice"` schema:

```python
recipe = "dmbench-native-choice-v1"
report = evaluate(
    cases,
    as_dmbench(model, recipe=recipe),
    batch=as_dmbench_batch(model, recipe=recipe),
    batch_size=8,
)
```

It delegates schema admission to DMBench's actual
`dmbench.adapters.native.native_request`. That helper checks exact
state/question/labels roles, task/type agreement, label/menu order and
criterion/menu text agreement. Wyat puts `question.instructions` verbatim
in the query. A string state becomes context verbatim; object or conversation
list state becomes context through DMBench's `canonical_json`, preserving
typed JSON values and list order. Public choices retain their exact IDs,
text and order. The envelope and criteria are not appended again to the
query. No heuristic envelope detection or independent parser is introduced.

Recipe selection is validated when constructing the callable. Scalar and
batch projections use the same conversion, and the entire public chunk is
validated before calling the loaded model. A missing or unknown native role,
criterion mismatch, unsupported task or task/type mismatch rejects the chunk.
JevBench `noul` and ordinal `score` remain unsupported by this choice recipe.
DMBench's native validator is imported only when the native path is used;
module import and the default path retain their existing optional-dependency
behavior. Recipe identity belongs in the model factory/evaluation provenance;
the selected recipe is never inferred from benchmark text.

The result answer is the selected original choice ID. Optional probabilities
are the loaded model's native choice softmax values mapped to those IDs; they
are relative scores, not calibrated correctness estimates. Token counts and
forward-pass counts remain unknown, so usage is omitted. Inference exceptions
propagate to DMBench, whose evaluator retains failed cases in its denominator.

A choice head requires supplied choices. Open answers, JevBench ordinal scores
and the frozen synthetic judgment label map cannot be represented through
this adapter; they fail explicitly. No new menu, ordinal interpretation,
shortlist, retry or truncation is invented. The model artifact's candidate and
context limits remain enforced.

The original integration was exercised against DMBench 0.2.0 source with
ten pytest tests, including a Hypothesis check that arbitrary public text is
preserved. Its actual evaluator scored a choice result and retained both a
model failure and an unsupported task in the original denominator. Explicit
batch evaluation retained order, grouping and gold isolation across a partial
final chunk. Source
file SHA-256 identities and the tested contract are recorded in the
[integration receipt](../benchmarks/real/dmbench-integration.json).

Twenty additional native-recipe checks use the actual peer role fixtures and
evaluator. They cover text whitespace/Unicode, canonical object/list state,
exact menu order, gold isolation, empty chunks and whole-chunk rejection of
malformed roles. An actual `WyatEncoder` with a character-tokenizer fixture
configured with `max_options=64` and `max_length=512` rejects 65 candidates
and oversized context without loading weights. These checks establish the
configured fixture's validation; they do not establish the released artifact's
candidate-count or token-length limits. The adapter preserves whichever limits
the loaded model declares. The original ten tests still pass. The
separate [native integration receipt](../benchmarks/real/dmbench-native-integration.json)
binds the adapter, peer helper/API/fixtures and tests by hash. This is
interface evidence; no native-recipe model-quality or performance result
has been measured here. Original pilot hashes and results remain unchanged.

An earlier full-model pilot with the default `dmbench-verbatim-choice-v1`
recipe exercised the actual DMBench evaluator through the downloaded full
Kai artifact on MLX GPU, in FP32. It used the first
four fixed training-source BANKING77 development queries, preserving their
choice IDs, descriptions, order and independently supplied labels. Scalar and
two-request batch conditions both answered all four correctly, with no
inference errors. The original assay's separate classification instruction
was omitted because the default recipe forwards public query text verbatim
into an empty context. The
[native pilot receipt](../benchmarks/real/dmbench-native-pilot.json) retains
every prediction, probabilities, actual DMBench timing, model identity and
source/version hashes. This single execution per condition has no warmup or
repetition protocol and supports no speed or broad model-quality claim.

Reproduce the bounded pilot with both projects installed or DMBench on
`PYTHONPATH`:

```sh
PYTHONPATH=/path/to/dmbench/src python -m benchmarks.real.dmbench.pilot \
  /local/kai-bundle
```

Run optional integration checks against an installed DMBench or a source
checkout:

```sh
PYTHONPATH=/path/to/dmbench/src pytest \
  tests/test_dmbench.py tests/interop/dmbench/test_native.py
```

Without DMBench, these optional tests are skipped; a skip is not a verified
integration result.
