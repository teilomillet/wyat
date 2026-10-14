# Use cases

Short, copyable examples of a Wyat decision model inside an application. Each
script is one idea and fits on one screen.

```sh
uv run python examples/use_cases/01_support_routing.py --demo
uv run python examples/use_cases/01_support_routing.py --model teilomillet/wyat-1-600m
```

`--demo` runs without weights: every option gets the same score, so you can
read the code and see the API work, but the answers mean nothing. Use
`--model` with a local model to see real decisions.

## The idea

The model never invents an answer. You give it a situation and the options;
it scores every option. Your code decides what to do with the result.

```
situation + your options  ->  model scores them  ->  your code acts
```

## Three task types

| Type | Question it answers | You read | Use it for |
| --- | --- | --- | --- |
| **Choice** | Which of these options? | `decision.selected`, `decision.ranking` | routing, next action, best draft |
| **Noul** | Yes or no, how likely? | `decision.probability_true` | checks, flags, guardrails |
| **Score** | Where on this ordered scale? | `decision.expected_ordinal_value` | urgency, relevance, risk |

```python
wyat.Request.choice("Which queue?", {"billing": "...", "other": "..."}, context=msg)
wyat.Request.noul("Is the answer supported by the source?", context=text)
wyat.Request.score("How urgent?", ["Can wait.", "Today.", "Now."], context=msg)
```

## The examples

| # | Use case | Type |
| --- | --- | --- |
| 01 | [Route a support message](01_support_routing.py) | Choice |
| 02 | [Agent: act, ask, or decline](02_agent_next_step.py) | Choice |
| 03 | [Sort tickets by urgency](03_ticket_urgency.py) | Score |
| 04 | [Check an answer against its source](04_answer_check.py) | Noul |
| 05 | [Rerank search results](05_rerank.py) | Score |
| 06 | [Pick the best reply draft](06_best_reply.py) | Choice |
| 07 | [Many questions on one contract](07_contract_review.py) | Choice + Noul + Score |
| 08 | [Email triage from yes/no signals](08_email_triage.py) | Noul + your rule |
| 09 | [Reviews to a table](09_review_insights.py) | Choice per aspect |

## Using it well

- **Own the options.** Write short, distinct descriptions. Add a "none of
  these" or "ask a human" option instead of forcing a guess (01, 02).
- **Batch.** `model.decide_many([...])` scores many requests in one call (03, 05).
- **Ask many questions at once.** `model.predict({"state": ..., "questions": ...})`
  shares one document across all questions, and can mix types (07, 08).
  `model.predict_many([...])` does that for many documents in one call (09).
- **Probabilities compare options; they are not guarantees.** Pick thresholds
  from your own labelled examples (04, 08).
- **Keep the decision separate from the action.** The model selects; your code
  checks the selection and does the work (02, 08).

## Going further

The [expert examples](expert/README.md) combine these decisions into larger
systems: triage with abstention, agents with guards, calibrated thresholds,
RAG gates, LLM judging and bulk jobs.

The [integration examples](integrations/README.md) plug Wyat into LangChain,
LangGraph, LlamaIndex, MCP, pandas and Pydantic through their usual
extension points.

## Make your own

Copy the closest example, then change three things: the question, the options
and the context. Try it on 20 to 50 real cases where you know the right
answer before trusting it.
