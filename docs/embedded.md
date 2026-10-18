# Put wyat inside a coding agent

Imagine a tiny coding helper. Someone tells it:

> "My program says 2 + 3 is -1. Please fix it."

The helper has a few buttons it can press: **look at the code**, **apply a fix**,
**run a test**, or **ask for help**. wyat chooses a button. The helper presses it.

```text
The agent tells wyat what happened and which buttons are available.
                         ↓
wyat returns a button's name, such as "fix".
                         ↓
The agent runs the function attached to that name.
                         ↓
The agent remembers the result and asks again.
```

## Try the story

In your application's uv project, install wyat:

```sh
uv add wyat
```

Use Python 3.12 or newer. See [installation](install.md) if you need to create
your uv project first.

Copy [the example](../examples/embedded/main.py) into your application as
`agent_demo.py`. Run:

```sh
uv run python agent_demo.py --demo
```

Practice mode uses hand-written scores. It needs no model download, so everyone
can see the same lesson: **inspect → uncertain choice → ask a larger chooser → fix → test**.

There are two practice choosers. The small one tries first. If its top two scores
are close, the router asks the larger one **before running an action**.

Here is the important part of its output:

```text
Turn 1
Agent knows: The user says add(2, 3) returns -1. It should return 5.
Agent offers: inspect, ask
Small wyat picks: inspect. Score gap: 2.00.
Router: gap at least 1.00; keep the small model's choice.
Final choice: inspect
Agent runs: inspect_code()
Result: I looked at the code. It says return a - b.

Turn 2
Agent knows: I looked at the code. It says return a - b.
Agent offers: inspect, fix, test, ask
Small wyat picks: test. Score gap: 0.10.
Router: gap below 1.00; ask the larger model.
Larger model picks: fix.
Final choice: fix
Agent runs: apply_known_fix()
Result: I changed return a - b to return a + b. No test run yet.

Turn 3
Agent knows: I changed return a - b to return a + b. No test run yet.
Agent offers: inspect, test, ask
Small wyat picks: test. Score gap: 2.00.
Router: gap at least 1.00; keep the small model's choice.
Final choice: test
Agent runs: run_test()
Result: I ran add(2, 3). Expected: 5. Got: 5. PASS!

Agent stopped: passed.
```

The example really changes its tiny Python function and runs that check in
memory. It does not edit your files. The example supplies the one-line fix;
the chooser selects when to apply it. Both practice choosers use rules written
for this story; neither is a trained model.

## This demo's default: ask the bigger model on a close call

Think of two answers in a race. If one is far ahead, keep it. If they are almost
next to each other, ask the bigger model for another decision.

The router checks `small.score_gap`, which subtracts the second-best score from
the best score:

```text
Clear choice: best = 2.0, second = 0.0 → gap = 2.0 → use the small model
Close call:   best = 0.1, second = 0.0 → gap = 0.1 → ask the larger model
Tie:         best = 0.0, second = 0.0 → gap = 0.0 → ask the larger model
```

The demo's default cutoff is **1.0**: a smaller gap sends the same request and
choices to the larger model. Only the final decision is acted on. Change the
cutoff with `--min-score-gap 0.5`, for example, or pass optional
`--min-probability-margin` and `--max-normalized-entropy` gates when you have
calibrated a specific model on held-out requests.

### Fit temperature and routing thresholds per model on held-out data

A score gap, probability margin, or entropy cutoff is a routing signal, not a
model-independent guarantee of correctness:

- **Temperature belongs to a specific trained model**: Use `report.calibrate()`
  in [wyat.evaluation](api.md#evaluate-labeled-cases-in-python) on held-out
  calibration cases for the exact model you deploy, pass the fitted value with
  `--temperature` (small model only), and evaluate fallback coverage and accuracy on separate test cases.
  Do not reuse a temperature or threshold fit on an earlier checkpoint or
  in-sample validation set.
- **Fallback rate is a product decision**: Escalating 10%, 40%, or 60% of
  decisions to a larger model or human reviewer depends on your latency, cost,
  and false-positive budget. Choose `--min-score-gap`,
  `--min-probability-margin`, or `--max-normalized-entropy` (`decision.score_gap`,
  `decision.probability_margin`, and `decision.normalized_entropy`) by measuring
  that trade-off on requests representative of your application. A large gap can
  still accompany a wrong answer.

## The three lines that connect them

Inside the example, the agent builds a `wyat.Request` containing its latest
observation and allowed choices. Then:

```python
decision = choose_action(request, small_model, larger_model)
action = actions[decision.selected]   # Find the matching agent function.
result = action()                     # The agent does the work.
```

`choose_action` calls `small_model.decide(request)`, checks the score gap, and
calls `larger_model.decide(request)` on a close call. `actions` is an ordinary
Python dictionary:

```python
actions = {
    "inspect": agent.inspect_code,
    "fix": agent.apply_known_fix,
    "test": agent.run_test,
    "ask": agent.ask_for_help,
}
```

So when wyat returns `"test"`, the agent runs `agent.run_test()`. Choosing and
doing are two separate steps. Your program owns the functions, their arguments,
and which choices are available. In this story, the fix button appears only
after the agent has looked at the code.

## Use a trained chooser

Install the inference runtime, then [obtain and verify local wyat weights](model-acquisition.md):

```sh
uv add wyat --extra inference
uv run python agent_demo.py --model teilomillet/wyat-1-800m --larger-model /path/to/larger-model
```

The 800M artifact uses PyTorch, including CPU on Apple Silicon. Its named
download is pending; use the prepared local model directory until it is pinned.

Supply both models; the example requires a fallback rather than silently
disabling it. Both must work with `wyat.load`. The larger fallback is a model
directory you provide.
An arbitrary chat-model directory cannot be passed to that loader. Another model
runtime can be connected in Python through a custom [scorer](api.md).

The example loads each model once. The agent and its functions stay the same.
A trained chooser may pick a different sequence, ask for help, or repeat a step.
The agent stops after six turns if it has not finished. A failed check is printed
as a failure. The bigger model can still make a mistake. If it fails, the run stops
before acting; it does not use the uncertain small-model choice as a substitute.
This example escalates
once; your application owns any further review of the larger model's answer.

This demonstrates connecting the parts. It does not establish how well a trained
model handles real coding tasks; the tiny check covers only `add(2, 3)`.

## A router works the same way

A router sends something to the right destination. Think of it as choosing which
door a message should go through.

| Coding agent | Support router |
| --- | --- |
| Input: what happened while fixing a bug | Input: a customer's message |
| Choices: inspect, fix, test, ask | Choices: billing, technical, human help |
| wyat returns `"test"` | wyat returns `"billing"` |
| The agent runs its test function | Your program sends the message to billing |

The same `Request → decide → selected ID → application function` connection
works in both. See [application workflows](agent-workflows.md) for a runnable
support-routing example and recovery details.
