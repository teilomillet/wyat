# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Shared by the integration examples: a local model, or a transparent stand-in.

The stand-in of ``expert/common.py`` plus a rule for Score, kept here so this
folder can be copied on its own.
"""

import argparse
import re
import shlex
import sys
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path
from typing import NoReturn

import wyat


def words(text: str) -> set[str]:
    """Lowercase stems (first four letters) of words longer than three letters."""
    return {w[:4] for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 3}


class OverlapScorer:
    """Word overlap between each option and the input. NOT a model.

    It only makes the demos produce varied numbers. Its decisions mean nothing.
    """

    def score_many(
        self, requests: Sequence[wyat.Request]
    ) -> tuple[tuple[float, ...], ...]:
        rows = []
        for request in requests:
            if request.task == "noul":
                asked = words(request.query)
                share = len(asked & words(request.context)) / max(len(asked), 1)
                true = 4.0 * share - 1.0
                rows.append(
                    tuple(true if c.id == "true" else 0.0 for c in request.candidates)
                )
            elif request.task == "score" and "\n" in request.context:
                # Overlap of the first context line (e.g. the question) with the
                # rest (e.g. the passage), mapped onto the rubric levels.
                first, _, rest = request.context.partition("\n")
                asked = words(first)
                share = len(asked & words(rest)) / max(len(asked), 1)
                top = share * (len(request.candidates) - 1)
                rows.append(
                    tuple(-((i - top) ** 2) for i in range(len(request.candidates)))
                )
            else:
                seen = words(request.context)
                rows.append(
                    tuple(float(len(words(c.text) & seen)) for c in request.candidates)
                )
        return tuple(rows)


def missing(requirement: str, error: ImportError) -> NoReturn:
    """Stop with install advice: Wyat never installs integration libraries.

    ``requirement`` is a pip requirement such as ``"mcp>=2.3"``. The advice
    differs when the library is installed but too old to have what we import.
    """
    package = re.split(r"[<>=!~ ]", requirement, maxsplit=1)[0]
    command = shlex.join(["python", Path(sys.argv[0]).name, *sys.argv[1:]])
    quoted = shlex.quote(requirement)
    try:
        installed = metadata.version(package)
    except metadata.PackageNotFoundError:
        problem = f"This example needs {requirement}, which Wyat does not install."
        install = f"pip install {quoted}"
    else:
        problem = (
            f"This example needs {requirement}; the installed {package} "
            f"{installed} fails to import: {error}"
        )
        install = f"pip install --upgrade {quoted}"
    raise SystemExit(
        f"{problem}\n"
        f"Run it with:   uv run --with {quoted} {command}\n"
        f"or install it: {install}\n"
        "In a wyat checkout, `uv sync --group examples` installs every example "
        "library; a plain `uv sync` removes them again."
    )


def load_model(description: str | None) -> wyat.Model:
    """Read --model or --demo; other flags are left for the example itself."""
    parser = argparse.ArgumentParser(description=description)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--model",
        type=Path,
        help="Model name (teilomillet/wyat-1-800m) or local model directory",
    )
    source.add_argument(
        "--demo", action="store_true", help="Word-overlap stand-in, no model"
    )
    parser.add_argument(
        "--device",
        help="Torch device, for example mps on Apple Silicon (default: cpu)",
    )
    arguments, _ = parser.parse_known_args()
    if arguments.demo:
        # stderr: stdout may be a protocol stream (MCP stdio), not a terminal.
        print("Demo mode: word-overlap stand-in, not a model.\n", file=sys.stderr)
        return wyat.Model(OverlapScorer(), model_id="demo-overlap")
    return wyat.load(arguments.model, device=arguments.device)
