# Your first wyat decision

Give wyat a message and the choices your application supports. It returns the
selected choice's ID. Here we choose between creating and cancelling a booking.

## 1. Install

Start in your application's [uv](https://docs.astral.sh/uv/getting-started/installation/)
project with Python **3.12 or newer**. If you do not have a project yet, create
one first:

```sh
uv init --bare --python 3.12 wyat-demo
cd wyat-demo
```

Then add wyat:

```sh
uv add wyat
```

uv manages the project's environment and lockfile; no activation is needed.
The core needs no dependencies or weights. For runtime choices and the pip
alternative, see [installation](install.md).

## 2. Try it without downloading a model

Copy [the runnable demo](../examples/quickstart/main.py) into your application as
`demo.py`, then run:

```sh
uv run python demo.py --demo
```

```text
Demo mode: fixed scores, no trained model.
Message: Please cancel my booking.
Choices:
  create: Create a booking
  cancel: Cancel the booking
Selected: cancel
```

This mode gives `cancel` a fixed score so you can try the API immediately.
It does not interpret the message. A trained model supplies the scores in the
next step; the request and result use the same API.

## 3. Use a trained local model

Install the PyTorch runtime:

```sh
uv add wyat --extra inference
```

Run the same demo with
[wyat-1-800m](https://huggingface.co/teilomillet/wyat-1-800m). The first run
downloads about 1.82 GB and verifies every file against the package's release
pins. Later runs verify and reuse the cache:

```sh
uv run python demo.py --model teilomillet/wyat-1-800m
```

The model chooses the result; the fixed demonstration output above is not an
expected answer for trained inference. 800M uses PyTorch on CPU by default;
MLX does not support its option-fork format. See
[model acquisition](model-acquisition.md) for the cache and local artifact details.

## 4. Put the API in your application

This is the complete ordinary Python path, after installing the inference
runtime and obtaining weights:

```python
import wyat

model = wyat.load("teilomillet/wyat-1-800m")

request = wyat.Request(
    query="Choose the requested booking action.",
    context="Please cancel my booking.",
    candidates=[
        wyat.Candidate(id="create", text="Create a booking"),
        wyat.Candidate(id="cancel", text="Cancel the booking"),
    ],
)
decision = model.decide(request)
print(decision.selected)
```

- `query` tells the model what to choose.
- `context` is the message or evidence to consider.
- Each candidate has an `id` your application uses and `text` describing the choice.
- `decision.selected` is one of those IDs. Selecting it performs no booking action.

Load the model once and reuse it for later requests. To try your own task, edit
the query, context, and candidates. IDs must be unique and a decision requires at
least two choices. Invalid requests raise an error rather than silently changing the input.

Continue with the [Python API](api.md) for batching, scores, and errors, or
[watch a coding agent use wyat step by step](embedded.md). See
[application workflows](agent-workflows.md) for using a selected ID in an application.
