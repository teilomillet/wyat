# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Suggest the expected output of code for a developer's test-review queue."""

import json
from collections.abc import Mapping

import wyat
from examples.use_cases.common import load_model


def suggest_test_output(
    model: wyat.Model, code: str, expression: str, outputs: Mapping[str, str]
) -> dict[str, object]:
    decision = model.decide(
        wyat.Request.choice(
            "What does the expression return under this Python function?",
            outputs,
            context={"code": code, "expression": expression},
        )
    )
    return {
        "expression": expression,
        "suggested_output": outputs[decision.selected],
        "next_step": "verify_in_existing_test_runner",
        "decision": decision.to_dict(),
    }


def run(model: wyat.Model) -> dict[str, object]:
    return suggest_test_output(
        model,
        'def f(s):\n    parts = s.split("-")\n'
        '    return ":".join(p[::-1] for p in parts if len(p) > 1)',
        'f("ab-c-def--gh")',
        {
            "keep_single": "ba:c:fed:hg",
            "filtered": "ba:fed:hg",
            "unreversed": "ab:def:gh",
            "reverse_order": "hg:fed:ba",
        },
    )


if __name__ == "__main__":
    print(json.dumps(run(load_model(__doc__)), ensure_ascii=False, indent=2))
