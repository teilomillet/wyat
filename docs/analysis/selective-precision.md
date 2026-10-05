# Selective precision declaration archive

The canonical prospective protocol and observed result are now in
[precision.md](precision.md).

The exact declaration frozen before the real CPU scan remains in
[the immutable executed source archive](../../benchmarks/analysis/results/mlx-mlp-fp16-eligibility.sources-960d68e2ea757b13.json),
with declaration SHA256
`f960d2d6b70c71fe4a686bcf648798842459b4ba04c053c6e3a589d5854025e1`.
Its six arms and exact-storage guard were preserved; all six failed eligibility,
so no full model or inference variant was executed.
