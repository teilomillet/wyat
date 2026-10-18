# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Every use case runs end to end without weights, and on a real model if given.

Set ``WYAT_TEST_MODEL`` to a model name or directory (and optionally
``WYAT_TEST_DEVICE``, for example ``mps``) to check the clear cases of each
example on that model. Without it those tests are skipped.
"""

import os
import runpy
import sys
from pathlib import Path

import pytest

import wyat

USE_CASES = Path(__file__).resolve().parents[2] / "examples" / "use_cases"
LEVELS = (USE_CASES, USE_CASES / "expert")
SCRIPTS = [path for level in LEVELS for path in sorted(level.glob("[0-9][0-9]_*.py"))]


# What each expert demo must still show with the word-overlap stand-in, so a
# change to the stand-in or an example cannot silently flatten the lesson.
EXPERT_SHOWS = {
    "01_support_triage.py": ["T-1: route", "T-3: human"],
    "02_hierarchical_routing.py": ["-> card_lost", "-> human"],
    "03_rag_gates.py": ["context: ['returns']", "I don't know"],
    "04_agent_loop.py": ["refused (already done)"],
    "05_calibrate_threshold.py": ["policy:"],
    "06_order_robustness.py": ["stable ->"],
    "07_llm_judge.py": ["A/B:"],
    "08_document_check.py": ["id_card: accept", "missing postal address"],
    "09_batch_throughput.py": ["1 refused", "input-length"],
}


@pytest.mark.parametrize("level", LEVELS, ids=["basic", "expert"])
def test_every_use_case_is_listed_in_its_readme(level: Path) -> None:
    readme = (level / "README.md").read_text(encoding="utf-8")
    names = [path.name for path in level.glob("[0-9][0-9]_*.py")]
    assert len(names) == 9
    assert all(f"({name})" in readme for name in names)


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.stem)
def test_use_case_runs_in_demo_mode(
    script: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def unexpected_load(*args: object, **kwargs: object) -> wyat.Model:
        raise AssertionError("demo mode must not load weights")

    monkeypatch.setattr(wyat, "load", unexpected_load)
    # Each level has its own ``common`` module; drop the other level's copy.
    monkeypatch.delitem(sys.modules, "common", raising=False)
    monkeypatch.syspath_prepend(str(script.parent))
    monkeypatch.setattr(sys, "argv", [script.name, "--demo"])
    runpy.run_path(str(script), run_name="__main__")
    output = capsys.readouterr().out
    assert output.startswith("Demo mode")
    assert len(output.splitlines()) > 2
    if script.parent.name == "expert":
        for expected in EXPERT_SHOWS[script.name]:
            assert expected in output


# Clear cases each example must get right on wyat-1 800M. Borderline cases the
# examples discuss (05 rerank, 08 lookalike sender, 09 sentiment) are omitted.
MODEL_SHOWS = {
    "01_support_routing.py": [
        "'I was charged twice for my order.' -> billing",
        "'The app crashes when I open my profile.' -> technical",
        "'Can you recommend a good restaurant nearby?' -> other",
    ],
    "02_agent_next_step.py": ["-> call_tool", "-> ask_user", "-> decline"],
    "04_answer_check.py": ["-> show:", "-> send for review:"],
    "06_best_reply.py": ["Send draft: a"],
    "07_contract_review.py": [
        "share_with_employees: yes",
        "share_with_partners: not_mentioned",
    ],
    "expert/01_support_triage.py": [
        "T-1: route queue=billing",
        "T-2: route queue=account",
        "T-3: human",
    ],
    "expert/02_hierarchical_routing.py": ["-> card_lost", "-> transfer_pending"],
    "expert/03_rag_gates.py": ["context: ['shipping']", "-> show"],
    "expert/05_calibrate_threshold.py": ["policy:"],
    "expert/06_order_robustness.py": ["stable -> approve"],
    "expert/08_document_check.py": ["id_card: accept", "missing postal address"],
    "expert/09_batch_throughput.py": ["4 scored once each, 1 refused", "input-length"],
}


@pytest.fixture(scope="module")
def real_model() -> wyat.Model:
    source = os.environ.get("WYAT_TEST_MODEL")
    if not source:
        pytest.skip("set WYAT_TEST_MODEL to check the examples on a real model")
    return wyat.load(source, device=os.environ.get("WYAT_TEST_DEVICE"))


@pytest.mark.parametrize("name", sorted(MODEL_SHOWS))
def test_use_case_clear_cases_on_a_real_model(
    name: str,
    real_model: wyat.Model,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    script = USE_CASES / name
    monkeypatch.setattr(wyat, "load", lambda *args, **kwargs: real_model)
    monkeypatch.delitem(sys.modules, "common", raising=False)
    monkeypatch.syspath_prepend(str(script.parent))
    monkeypatch.setattr(sys, "argv", [script.name, "--model", "loaded-by-fixture"])
    runpy.run_path(str(script), run_name="__main__")
    output = capsys.readouterr().out
    for expected in MODEL_SHOWS[name]:
        assert expected in output
