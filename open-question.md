# Decisions needed for production Wyat inference

These questions do not block the local Kai reference backend or the generic
logit-scorer API. They block selecting and validating the eventual Wyat model
contract and claiming a production performance or quality target.

1. **Which checkpoint and export contract will be promoted from training?**
   Kayak's released CLM, its joint models, and CLM Lab's validity model differ.
   The export must name the family/version, rendering and segmented tokenization,
   pooling, head shape, candidate limits, score interpretation/calibration,
   adapter base revision, and hashes. See `docs/training-contracts.md`.
2. **Which deployment hardware and request distribution define “20× vanilla”?**
   Freeze the device, runtime, precision, CPU threads, contexts, candidate counts,
   batch sizes/concurrency, repeated-candidate rate, warm/cold conditions, baseline
   implementation, and p50/p95/p99 or throughput objective. A tokenization-only
   improvement does not establish a full-engine speedup.
3. **What labeled development and reserved confirmation cohort gates quality?**
   Specify accepted metrics and tolerances for logits, ranking disagreements,
   task accuracy, abstention, and calibration. Real Kai CPU/MLX parity is now
   measured on a frozen BANKING77 training-source development assay. Prior model
   exposure is unknown, so it cannot gate held-out Wyat model quality. DMBench
   owns benchmark definitions and provenance; Wyat supplies inference adapters.
4. **Which externally exposed deployment boundary is authorized?**
   The supplied server runs locally. Production hosting, authentication/TLS,
   resource budgets, and model access should follow the chosen deployment.
5. **Who approves a registry release, and are repository controls available?**
   The release workflow is disabled until a maintainer configures PyPI trusted
   publishing, the protected `pypi` environment and its reviewers, then enables
   publication. Verify protection-rule availability for the private repository's
   plan before enabling it. A new, unused version is needed because the existing
   PyPI `0.0.1` differs from this implementation. These administrative choices
   do not block local tests, builds, or CI preparation; see `docs/releasing.md`.
   PyPI distribution is public; confirm that audience for the source archive,
   including its retained research evidence, before enabling registry publication.
No answer has been invented for these choices. Work can continue mechanically
on the reference backend and its evidence while training contracts develop.
