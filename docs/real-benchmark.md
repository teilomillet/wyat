# Measuring an engine change

Use the same local model artifact and labeled request cohort for both engines.
Freeze the candidate configuration and quality tolerances before inference.
Change one setting at a time: batching, token caching, attention implementation,
precision, device or runtime. Keep the baseline description precise; a speedup
over singleton Wyat is not a speedup over the upstream server.

The public `wyat benchmark` command accepts cases on stdin and emits the complete
JSON receipt on stdout; see the [CLI example](../README.md#measure-an-engine-change).
The final [CPU/MLX receipt](../benchmarks/real/runtime-cpu-mlx77.json) uses the
entire 77-case cohort, batch eight, cache zero and eight CPU threads. Its paired
median is 2.02x, with 1.85x–2.16x paired 5th–95th percentiles. All rankings and
decisions agree. This compares FP32 CPU and Metal runtimes on the M4 Pro;
the repeated ratios are measurement variation, not confidence intervals.

```python
import json
from pathlib import Path

import wyat
from wyat.benchmark import BenchmarkEngine, BenchmarkProtocol, benchmark_pair
from wyat.evaluation import case_from_dict

path = Path("benchmarks/real/banking77-train-dev-menu6.jsonl")
cases = tuple(case_from_dict(json.loads(line)) for line in path.read_text().splitlines())
model = wyat.load("/local/kai-bundle", device="cpu")
report = benchmark_pair(
    BenchmarkEngine("singleton", model, batch_size=1,
                    conditions={"device": "cpu", "optimization": "none"}),
    BenchmarkEngine("batch4", model, batch_size=4,
                    conditions={"device": "cpu", "optimization": "batching"}),
    cases[:16],
    protocol=BenchmarkProtocol(repeats=5, warmup=1),
)
Path("measurement.json").write_text(
    json.dumps(report.to_dict(), indent=2, allow_nan=False) + "\n"
)
```

The example compares batch scheduling on the same loaded model. Separate loaded
engines are needed to compare distinct runtime options. Running independently
loaded models in one process shares host and device memory: process RSS is not
per-engine memory usage. For a deployable memory comparison, run each arm in an
isolated process and record loading, idle resident memory and peak memory with
the same model and cohort. OS high-water RSS is cumulative; subtracting two
high-water readings does not measure allocated model memory.

## Predeclared development protocol

The first bounded full-checkpoint assay uses the first 16 cases of the committed
77-row cohort. Its choice is made from case order, before predictions. Each
arm receives every case in the same order, and the canonical input hash binds
IDs, queries, contexts, candidates, task and expected labels. The full cohort
is available for broader development runs. Any subset has its own hash and IDs;
its performance and accuracy cannot be reported as covering all 77 intents.

One complete warmup cohort is excluded from timing statistics; its failures
remain in the receipt and prevent acceptance. At least three measured repetitions
alternate baseline/candidate and candidate/baseline order. Use five or more
repetitions for retained results. Run engines serially, close unrelated heavy
workloads, and record hardware, model identity, software versions, precision,
attention mode, cache size and CPU threads in `conditions`. Repetition intervals
describe measurement variability; they do not quantify population accuracy or
remove thermal, shared-process or cache-state effects.

Native Torch and MLX scorers materialize host floats before returning. An engine
with deferred work must supply `BenchmarkEngine.synchronize`, called before
each cycle and after each batch. Wall-clock cycles include request packing,
tokenization, model execution, decision interpretation and result bookkeeping.
Loading, warmups, HTTP and JSON encoding are excluded. This is an offline engine
assay; HTTP admission, contention and streaming need separate load tests.

Cycle p50/p95 cover the whole declared cohort. Attempted and successful request
throughputs are separate, so failed requests cannot appear as useful work.
Batch residence latency assigns the full synchronous batch time to each member;
it is not per-request isolated service time, queue latency or throughput divided
by batch size. Every measured repetition retains predictions, logits,
probabilities, rankings, labels, timings and exceptions in original case order.
A failed batch retains each member as a failed prediction.

Default numerical gates are declared before execution: each corresponding logit
must satisfy absolute tolerance `1e-3` or relative tolerance `1e-3`, and every
probability must differ by at most `1e-4`. Every predicted top choice and full
ranking must agree, every request must succeed, and labeled accuracy must not
decrease. Exact ties follow caller order. A near-tie flip still fails selection
agreement even when its numerical differences pass. Repetitions repeat the
same labeled examples; they are not additional independent observations.

A faster candidate with a failed quality gate is rejected. Any precision-specific
relaxed tolerance belongs to a new protocol declared before running; it cannot
be chosen after observing drift simply to make that candidate pass. A passed
parity gate preserves the baseline's measured behavior; it does not establish
that the baseline model is useful. Record labeled accuracy separately.

`speed.median_speedup_positive` reports only whether the paired median exceeds
one. It is separate from the quality `passed` flag and does not establish a
supported performance improvement. Read the complete paired variation, total
successful throughput and p95 latency together before adopting a change. A
near-unit median gain with slower overall throughput or worse tail latency
does not justify a performance claim.

## Source and scope

The committed assay uses actual annotated queries from the official
[PolyAI BANKING77 source](https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data)
at revision `57ec275d8078af65b7731c2a98be812d844a6d6b`. Only `train.csv`
was downloaded. Its SHA-256 is
`b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b`.
Neither the official test split nor reserved Kayak confirmation data is read.

There is one deterministic unique normalized utterance per intent. Six-choice
menus contain the original labeled intent and five deterministic sampled
negatives, with deterministic positive placement. Candidate descriptions
replace underscores in the original intent name with spaces. This is a
changed task and a runtime development assay, **not the official 77-way
BANKING77 score**. Labels come from the dataset rather than from Kai or an LLM.
Source label errors and prior checkpoint exposure remain unknown.

The [dataset manifest](../benchmarks/real/banking77-train-dev-menu6.manifest.json)
records source row IDs, text hashes, normalized utterance groups, label identity,
positive position and the generated byte hash. The source is licensed
[CC-BY-4.0](../benchmarks/real/BANKING77-LICENSE). Attribution: PolyAI;
Casanueva et al. (2020),
[Efficient Intent Detection with Dual Sentence Encoders](https://arxiv.org/abs/2003.04807).

Reproduce from the downloaded pinned training CSV:

```sh
python -m benchmarks.prepare_banking77 /path/to/train.csv
```

An optional `--menu-size 77` generates the corresponding full-intent menus.
An artifact's declared candidate limit may reject those menus; do not increase
that limit simply to run this assay. Any compatible workload needs its own
token-length and resource receipt; it does not become a held-out official score
by using all 77 labels.
