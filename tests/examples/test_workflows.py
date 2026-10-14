# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""No-weight workflow contracts: permitted effects and retained failures."""

import json
import runpy
from pathlib import Path
from typing import Any

import pytest

import wyat

WORKFLOWS = Path(__file__).resolve().parents[2] / "examples" / "workflows"


def example(name: str) -> dict[str, Any]:
    return runpy.run_path(str(WORKFLOWS / f"{name}.py"))


@pytest.mark.parametrize("name", ["routing", "human_review"])
def test_workflow_has_guarded_no_weight_main(
    name: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def unexpected_load(*args: object, **kwargs: object) -> wyat.Model:
        raise AssertionError("Workflow examples must not load weights")

    monkeypatch.setattr(wyat, "load", unexpected_load)
    example(name)
    assert capsys.readouterr().out == ""
    runpy.run_path(str(WORKFLOWS / f"{name}.py"), run_name="__main__")
    assert isinstance(json.loads(capsys.readouterr().out), dict)


def test_routing_selection_is_separate_from_effect_and_recovery() -> None:
    report = example("routing")["run_demo"]()
    assert report["result"]["selected"] == "billing"
    assert report["effects_after_selection"] == report["effects_after_failure"] == []
    assert report["effects"] == [{"event_id": "event-1", "queue": "billing"}]
    assert report["reviews"] == [
        {"event_id": "event-2", "reason": "model_selected_review"}
    ]
    assert {row["stage"] for row in report["operational_failures"]} == {
        "action",
        "inference",
        "output",
    }
    assert report["input_rejections"][0]["stage"] == "prepare"
    assert report["recovery"]["status"] == "succeeded"
    assert report["unknown_id_blocked"] is True


def test_routing_blocks_unpermitted_result_before_effect() -> None:
    values = example("routing")
    request = values["prepare_request"]("Duplicate invoice")
    unknown = wyat.Decision("fixture", (1.0,), (1.0,), ("delete-account",))
    with pytest.raises(ValueError, match="permitted"):
        values["permitted_queue"](request, unknown)
    effects: list[dict[str, str]] = []
    values["enqueue"](effects, "event", "billing")
    with pytest.raises(ValueError, match="different queue"):
        values["enqueue"](effects, "event", "technical")
    assert effects == [{"event_id": "event", "queue": "billing"}]


def test_human_review_has_pending_record_and_separate_simulated_resolution() -> None:
    report = example("human_review")["run_demo"]()
    assert report["policy_decision"] == "review"
    assert report["pending_reviews"][0]["status"] == "pending"
    assert report["effects_before_resolution"] == []
    assert report["resolved_reviews"][0]["status"] == "resolved"
    assert (
        report["resolved_reviews"][0]["resolution"]["reviewer"] == "simulated-reviewer"
    )
    assert report["effects"] == [{"event_id": "event-1", "queue": "billing"}]
    assert len(report["operational_failures"]) == 1
    assert len(report["resolved_reviews"]) == 1


def test_review_rejects_unmatched_resolution_without_effect() -> None:
    values = example("human_review")
    effects: list[dict[str, str]] = []
    review: dict[str, object] = {"event_id": "a", "status": "pending"}
    resolution = {"event_id": "b", "decision": "approve_billing_route"}
    with pytest.raises(ValueError, match="match"):
        values["resolve_review"](review, resolution, effects)
    assert review["status"] == "pending" and effects == []


def test_review_policy_requires_explicit_valid_cutoffs() -> None:
    policy = example("human_review")["apply_policy"]
    assert policy(0.9, accept_min=0.9, reject_max=0.1) == "accept"
    assert policy(0.1, accept_min=0.9, reject_max=0.1) == "reject"
    assert policy(0.5, accept_min=0.9, reject_max=0.1) == "review"
    with pytest.raises(ValueError, match="policy"):
        policy(0.5, accept_min=0.1, reject_max=0.9)
