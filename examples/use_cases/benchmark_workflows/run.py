# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Run a teaching workflow with sample JSON or your own application records."""

import argparse
import importlib
import inspect
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import wyat
from examples.use_cases.common import DemoScorer

WORKFLOWS = {
    "group_statement": ("01_group_statement", "propose_statement"),
    "phishing_triage": ("02_phishing_triage", "triage_email"),
    "review_tags": ("03_review_tags", "tag_review"),
    "product_relevance": ("04_product_relevance", "place_product"),
    "code_output_review": ("05_code_output_review", "suggest_test_output"),
    "chess_move": ("06_chess_move", "choose_move"),
    "application_conditions": ("07_application_conditions", "check_application"),
    "service_handoff": ("08_service_handoff", "prepare_handoff"),
    "updated_request": ("09_updated_request", "rank_proposals"),
    "incident_priority": ("10_incident_priority", "prioritize_incident"),
}


def workflow_function(name: str) -> Callable[..., dict[str, object]]:
    if name not in WORKFLOWS:
        raise ValueError(
            f"unknown workflow {name!r}; choose from {', '.join(WORKFLOWS)}"
        )
    module_name, function_name = WORKFLOWS[name]
    module = importlib.import_module(
        f"examples.use_cases.benchmark_workflows.{module_name}"
    )
    return cast(Callable[..., dict[str, object]], getattr(module, function_name))


def validate_input(name: str, value: object) -> dict[str, Any]:
    """Validate JSON against the selected example's existing function parameters."""
    function = workflow_function(name)
    if not isinstance(value, dict):
        raise ValueError("input must be a JSON object")
    required = set(inspect.signature(function).parameters) - {"model"}
    missing, extra = required - value.keys(), value.keys() - required
    if missing or extra:
        raise ValueError(
            f"input fields for {name}: {', '.join(sorted(required))}; "
            f"missing={sorted(missing)}, unexpected={sorted(extra)}"
        )
    for field, item in value.items():
        if field in {"drafts", "outputs", "proposals"}:
            if not isinstance(item, dict) or len(item) < 2:
                raise ValueError(f"{field} must map at least two IDs to descriptions")
            if not all(
                isinstance(k, str) and k.strip() and isinstance(v, str) and v.strip()
                for k, v in item.items()
            ):
                raise ValueError(f"{field} must contain nonempty string IDs and text")
        elif field == "opinions":
            if (
                not isinstance(item, list)
                or not item
                or not all(isinstance(text, str) and text.strip() for text in item)
            ):
                raise ValueError("opinions must be a nonempty array of text")
        elif not isinstance(item, str) or not item.strip():
            raise ValueError(f"{field} must be nonempty text")
    return value


def load_input(name: str, path: Path) -> dict[str, Any]:
    """Accept the supplied ten-example pack or a single workflow's input object."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict) and name in value:
        value = value[name]
    return validate_input(name, value)


def run_workflow(
    model: wyat.Model, name: str, inputs: dict[str, Any]
) -> dict[str, object]:
    return workflow_function(name)(model, **validate_input(name, inputs))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example", choices=WORKFLOWS)
    parser.add_argument("--input", type=Path, required=True, help="Input JSON file")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--model", default="teilomillet/wyat-1-800m")
    source.add_argument("--demo", action="store_true", help="Equal scores; no weights")
    parser.add_argument("--backend", choices=("torch", "mlx"), default="torch")
    parser.add_argument("--device", help="cpu, mps, cuda; gpu or cpu for MLX")
    args = parser.parse_args(argv)
    try:
        # Reject malformed user data before loading or downloading model weights.
        inputs = load_input(args.example, args.input)
        if args.demo:
            model = wyat.Model(DemoScorer(), model_id="demo")
        else:
            model = wyat.load(args.model, backend=args.backend, device=args.device)
        result = run_workflow(model, args.example, inputs)
    except (OSError, ValueError, wyat.ModelInputError, wyat.ModelOutputError) as error:
        print(f"Cannot run {args.example}: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "example": args.example,
                "mode": "demo_equal_scores" if args.demo else "model",
                "result": result,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
