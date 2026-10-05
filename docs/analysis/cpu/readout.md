# Exploratory CPU final readouts

This experiment asks whether skipping unused final-block query, output and MLP
rows improves the already confirmed retained-prefix CPU scorer. Production code,
loader defaults and server behavior remain unchanged. The temporary scorer is
owned by `benchmarks/analysis/torch/readout/scorer.py`; it is removed or replaced
only after a separately reviewed experiment establishes a useful boundary.

One FP32 CPU Kai base owns the weights. Retained and selected resources build
independent eager snapshots of the same exact 25-token prefix. Each call encodes
complete original requests, admits the entire batch before removing the prefix,
and forks fresh stock caches. Mixed, mismatched or prefix-overlapping readouts
fall back through the original production tensor owner without encoding again.
The existing fixed weights/config/device/encoder lifetime, no backbone hooks,
disabled CPU autocast, default RoPE and stock SDPA requirements still apply.

The selected path captures absolute suffix positions, full early causal mask and
full suffix RoPE before any cache update. It executes the first 27 stock blocks.
The last block retains every suffix K/V row and the complete prefix K/V, while
query/output/MLP/final normalization operate on six candidate rows and one query
row. Ordered duplicates and padded candidate slots are preserved. Its explicit
mask permits each key only at or before the readout's original absolute position.
This real-arithmetic equivalence does not establish identical FP32 rounding or
native kernel dispatch.

## Fixed screen

After independent tiny block/core tests and source review, the owned launcher is:

```sh
python -m benchmarks.analysis.torch.readout.run .models/Decision-2.0-Kai-0.6B \
  --output benchmarks/analysis/results/torch-readout-first8.json
```

One worker has a 180-second cap. The runner admits the original ordered 77 inputs,
all nine original model asset hashes, dataset/reference identities and archived
historical confirmations before selecting the canonical first eight. B8, FP32,
eight intraop threads, max length 512 and zero token cache are fixed. Attention,
precision and native HF eligibility flags are observed without modification.
The original thread count is restored before successful completion is saved.

O is ordinary production scoring, R final production retained-prefix scoring,
and S selected final readouts. Warm O/R/S, then ordinary O/R/S/S/R/O, then separate
intrusive R/S profiles: 11 cohorts, 88 outcomes. Every output must preserve exact
IDs, gold labels, complete ranks, selections and accuracy, with original logit
absolute/relative tolerance 1e-3 and probability absolute tolerance 1e-4. All
ordinary arms are also compared pairwise. No tolerance changes are permitted.
Each raw cohort is persisted first, then checked against the reference and
available arm anchors. The first drift, rank, failure or timing violation stops
further calls, preserving partial history and the pending cohort list.

Ordinary timing is the existing complete-cohort metric, including encoding,
packing, fresh cache construction, model execution and host decision output.
Setup, source hashes, journal serialization and profiler reduction are outside
that metric. Two observations per arm are exploratory, with no speed gate.
`accepted` and `performance_passed` remain false even when quality/dispatch pass.

The separate native profiles must observe 28 full-query CPU flash calls for R
and 27 full-query plus one seven-query call for S, retaining eight K/V heads and
all 192 key/value rows. The selected mask has shape [8,1,7,192]. Eligibility flags
alone do not prove this dispatch; missing/incompatible evidence fails the gate.
Profiles are intrusive and their grouped operator times do not establish the
last layer's share of ordinary latency.

Every completed raw cohort is journaled before validation. Invalid clock values
retain null plus their original representation; partial profiler failures retain
produced predictions, and unexecuted cohorts remain pending.
The journal has one immutable raw entry per produced cohort. Main rows add
reference and anchor reports afterward; their original raw fields must agree.
Main/journal/source archive paths are exclusive. Current runtime, benchmark,
tests, docs and relevant
installed dependency source are frozen and checked outside each body and at the
end. Historical identities come from executed archives, independently of later
legitimate production changes. These observations do not attest every native
binary/imported object. All raw receipts and failed attempts remain immutable.

Extended K/V retention remains unchanged; this is not a memory ceiling. A useful
first-eight result would require a separately declared full77 R-relative B8/B1
confirmation with the original four speed gates before any production proposal.
No held-out quality, online/server capacity, GPU or 20x claim follows from this
screen. Under assumed uniform layer costs, eliminating an entire last block
would illustrate 28/27 speedup; no CPU profile establishes that uniformity, and
this candidate still computes full K/V.
