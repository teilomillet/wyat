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


@pytest.mark.parametrize(
    "name", ["routing", "prioritization", "human_review", "policy_evaluation"]
)
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


def test_prioritization_keeps_batch_failure_and_bimodal_distribution() -> None:
    report = example("prioritization")["run_demo"]()
    assert report["queue_after_failure"] == ["split", "urgent", "invalid"]
    assert report["queue_after_recovery"] == ["urgent", "split", "invalid"]
    assert len(report["initial_results"]) == 2
    assert len(report["operational_failures"]) == 1
    assert report["initial_attempts"] == 3
    split = report["initial_results"]["split"]
    assert split["expected_ordinal_value"] == pytest.approx(1.0)
    assert split["probabilities"]["1"] < 0.002
    assert split["probabilities"]["0"] == split["probabilities"]["2"]


def test_priority_policy_preserves_ties_and_rejects_incomplete_batch() -> None:
    values = example("prioritization")
    from wyat.tasks import interpret, public_questions

    _, questions = public_questions(
        {
            "state": "fixture",
            "questions": {"s": {"type": "score", "criteria": ["low", "high"]}},
        }
    )
    answer = interpret(questions[0], (0.0, 0.0))
    assert values["rank_queue"](
        ["second", "first"], {"first": answer, "second": answer}
    ) == ["second", "first"]
    with pytest.raises(ValueError, match="incomplete"):
        values["rank_queue"](["first", "missing"], {"first": answer})


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


def test_policy_evaluation_keeps_every_fixture_in_appropriate_denominators() -> None:
    report = example("policy_evaluation")["run_demo"]()
    assert report["population"] == {"total": 12, "positive": 6, "negative": 6}
    assert report["statuses"] == {
        "ok": 8,
        "missing": 1,
        "failed": 1,
        "invalid": 1,
        "unsupported": 1,
    }
    counts = report["counts"]
    assert (
        counts["accept"],
        counts["reject"],
        counts["review"],
        counts["operator"],
    ) == (3, 2, 3, 4)
    assert counts["false_accepts"] == 2 and counts["false_rejects"] == 1
    assert counts["unresolved_positives"] == 4 and counts["unresolved_negatives"] == 3
    metrics = report["metrics"]
    for name, numerator, denominator in (
        ("valid_prediction_coverage", 8, 12),
        ("autonomous_coverage", 5, 12),
        ("uncertainty_review_load", 3, 12),
        ("operational_failure_load", 4, 12),
        ("false_accept_rate", 2, 6),
        ("false_reject_rate", 1, 6),
        ("autonomous_error_rate", 3, 5),
    ):
        assert metrics[name]["numerator"] == numerator
        assert metrics[name]["denominator"] == denominator
    failures = [row for row in report["rows"] if row["status"] != "ok"]
    assert all(row["outcome"] == "operator" and "answer" not in row for row in failures)


def test_policy_compares_unrounded_canonical_probability() -> None:
    report = example("policy_evaluation")["run_demo"]()
    row = next(row for row in report["rows"] if row["id"] == "boundary-positive")
    probability = row["answer"]["probability_true"]
    assert probability < 0.8 and round(probability, 6) == 0.8
    assert row["outcome"] == "review"


def test_policy_rejects_empty_or_duplicate_inventory() -> None:
    values = example("policy_evaluation")
    arguments = {"policy_id": "test-v1", "accept_min": 0.8, "reject_max": 0.2}
    with pytest.raises(ValueError, match="nonempty"):
        values["evaluate_policy"]([], **arguments)
    case = values["FROZEN_CASES"][0]
    with pytest.raises(ValueError, match="unique"):
        values["evaluate_policy"]([case, case], **arguments)
