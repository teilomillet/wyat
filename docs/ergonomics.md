# Wyat ergonomics standard

Make a correct decision easy to express, understand, and recover when it fails.
The preferred public path is **load once → describe a request → decide → use an
application ID**. This is a design choice supported by the current application
tasks, open to revision when comparative user evidence supports a better one.

The Python package remains `wyat`. Ergonomic work does not select a new model
architecture or establish trained-model quality.

## Hard contracts

- Preserve application IDs, candidate descriptions, request/candidate order,
  stable exact ties, and documented scalar/batch behavior.
- Retain finite raw-logit interpretation and relative, uncalibrated softmax
  probabilities. A selected ID identifies an action; it does not execute it.
- Keep local loading explicit. Importing the package does not load weights or
  optional runtimes. Do not add implicit downloads or custom model-code execution.
- Reject unsupported input explicitly. Do not truncate, retry, split, repair,
  skip, or invent successful output to hide a failure.
- Preserve the wire format, failure-inclusive evaluation, and explicit resource,
  concurrency, cancellation, and model-lifetime ownership.
- Preserve existing user work, model contracts, and privacy. Diagnostics must
  not expose request text merely to improve recoverability.

## Seven design rules

1. **Design from a complete user task.** Write the consumer example before adding
   an interface. Include preparation, result use, and a likely failure.
2. **Teach one recommended path.** Keep `load`, `Request`, `decide`, and
   `selected` as the ordinary path. Additional forms must solve an observed unmet
   task and share the same validation and interpretation.
3. **Use types where meaning needs protection.** Keep IDs distinct from
   descriptions, validated immutable inputs, and the small structural `Scorer`
   protocol. Avoid registries, inheritance frameworks, and scalar wrappers
   without a present caller need.
4. **Make effects and owners apparent.** Load outside request handlers. Ordinary
   attribute access performs no inference or I/O. Advanced runtime settings do
   not enter the first-use lesson. Direct model concurrency belongs to its
   scorer; the HTTP server supplies its documented admission gate.
5. **Make failures actionable.** A caller can distinguish rejected input from
   invalid scorer output. A diagnostic identifies the boundary and a permitted
   next step, retaining causes without concealing unrelated backend failures.
6. **Write for the next reader.** Use ordinary Python, meaningful names, clear
   intermediate values, early validation, focused functions, and comments that
   explain constraints. Keep the current Ruff and strict mypy tooling rather
   than starting a whole-repository style migration.
7. **Keep examples executable and claims bounded.** Test the contract and its
   connected boundaries. Fixtures verify mechanics; users establish usability;
   independent labeled cohorts establish model quality.

The maintenance test is simple: can a reader explain the information, effects,
and failures of one decision without traversing unrelated modules?

## Complete application tasks

The documentation and examples should let callers:

- Explore a fixed scorer without downloading weights.
- Install a runtime, obtain a supported artifact and make a local decision.
- Use an exact selected ID through an application's permitted action map.
- Reuse one model across scalar and batch calls while preserving request order.
- Read input-ordered scores by candidate ID and understand relative probabilities.
- Correct invalid inputs and missing dependencies using actionable errors.
- Integrate a custom scorer with finite raw logits and explicit output failures.
- Serve the same request over HTTP with explicit admission and model ownership.
- Evaluate labeled cases while retaining failed calls in the denominator.

Prefer one recommended path: [install](install.md), [obtain a model](model-acquisition.md),
then [load, describe and decide](api.md). The [routing example](agent-workflows.md)
connects a choice to an explicit application effect. These examples demonstrate
API mechanics; they do not establish human usability or trained-model quality.

When assessing an API change, check complete task correctness first, then time
and the concepts a caller must learn. Hidden setup, conversions and recovery
still count toward the complete task. Keep a simpler established interface
unless evidence shows that a change improves supported use.
