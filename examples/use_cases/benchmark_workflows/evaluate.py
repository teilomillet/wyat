# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Exercise the ten original workflow fixtures once and retain every outcome.

Run as a module from the checkout. Expected values below belong to this
illustrative diagnostic; they are not JDI or DMBench gold labels.
"""

import hashlib
import importlib
import json
import platform
import sys
from contextlib import redirect_stdout
from pathlib import Path
from time import perf_counter

import wyat
from examples.use_cases.common import load_model

CHECKS = (
    ("01_group_statement", "draft_id", "night_access"),
    ("02_phishing_triage", "queue", "suspected_phishing"),
    (
        "03_review_tags",
        "tags",
        {
            "food_positive": True,
            "service_negative": True,
            "service_positive": False,
            "ambience_negative": False,
        },
    ),
    ("04_product_relevance", "section", "accessories"),
    ("05_code_output_review", "suggested_output", "ba:fed:hg"),
    ("06_chess_move", "move", "h5f7"),
    ("07_application_conditions", "next_step", "request_missing_evidence"),
    ("08_service_handoff", "destination", "network_billing"),
    ("09_updated_request", "proposal_id", "updated_quantity"),
    ("10_incident_priority", "queue", "team_incident"),
)


def evaluate(model: wyat.Model) -> dict[str, object]:
    outcomes = []
    for name, field, expected in CHECKS:
        module = importlib.import_module(
            f"examples.use_cases.benchmark_workflows.{name}"
        )
        start = perf_counter()
        row: dict[str, object] = {
            "example": name,
            "field": field,
            "expected": expected,
            "source_sha256": hashlib.sha256(
                Path(__file__).with_name(f"{name}.py").read_bytes()
            ).hexdigest(),
        }
        try:
            result = module.run(model)
            row.update(result=result, passed=result[field] == expected, error=None)
        except (Exception, SystemExit) as error:
            row.update(
                result=None, passed=False, error=f"{type(error).__name__}: {error}"
            )
        row["seconds"] = perf_counter() - start
        outcomes.append(row)
        print(f"{name}: {'PASS' if row['passed'] else 'FAIL'}", file=sys.stderr)
    return {
        "scope": "original illustrative workflow fixtures; not an official benchmark",
        "model_id": model.model_id,
        "runtime_arguments": sys.argv[1:],
        "python": platform.python_version(),
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "quality_valid": not model.model_id.startswith("demo"),
        "passed": sum(bool(row["passed"]) for row in outcomes),
        "expected_examples": len(CHECKS),
        "outcomes": outcomes,
    }


if __name__ == "__main__":
    # Keep loader progress and the --demo banner out of the JSON stream.
    with redirect_stdout(sys.stderr):
        model = load_model(__doc__)
    report = evaluate(model)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(
        1 if report["quality_valid"] and report["passed"] != len(CHECKS) else 0
    )
