"""Check the teaching story's actual actions and visible stopping boundaries."""

from collections.abc import Sequence

import pytest

import wyat
from examples.embedded.main import (
    PracticeLargerScorer,
    PracticeScorer,
    ToyAgent,
    choose_action,
    run_agent,
)


class FixedScores:
    def __init__(self, scores: tuple[float, ...]) -> None:
        self.scores = scores
        self.calls = 0

    def score_many(
        self, requests: Sequence[wyat.Request]
    ) -> tuple[tuple[float, ...], ...]:
        self.calls += 1
        return tuple(self.scores for _ in requests)


def test_choice_alone_does_not_change_the_program() -> None:
    agent = ToyAgent()
    model = wyat.Model(PracticeScorer(), model_id="practice")
    decision = model.decide(agent.request())
    assert decision.selected == "inspect"
    assert agent.inspected is False
    assert agent.code == "def add(a, b):\n    return a - b\n"


def test_real_toy_check_fails_before_patch_and_passes_after() -> None:
    agent = ToyAgent()
    assert "FAIL" in agent.run_test()
    assert agent.status == "running"
    assert {item.id for item in agent.request().candidates} == {"inspect", "ask"}
    agent.inspect_code()
    assert {item.id for item in agent.request().candidates} == {
        "inspect",
        "fix",
        "test",
        "ask",
    }
    agent.apply_known_fix()
    assert "return a + b" in agent.code
    assert "PASS!" in agent.run_test()
    assert agent.status == "passed"


def test_practice_story(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        run_agent(
            wyat.Model(PracticeScorer(), model_id="practice"),
            wyat.Model(PracticeLargerScorer(), model_id="larger"),
        )
        == "passed"
    )
    output = capsys.readouterr().out
    assert output.index("Final choice: inspect") < output.index("Final choice: fix")
    assert output.index("Final choice: fix") < output.index("Final choice: test")
    assert "Agent runs: apply_known_fix()" in output
    assert "Expected: 5. Got: 5. PASS!" in output
    assert "Small Wyat picks: test. Score gap: 0.10." in output
    assert "Larger model picks: fix." in output
    assert output.count("ask the larger model") == 1


def test_repeated_choice_stops_without_claiming_success(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Two choices at first, then four after inspection; offer one score per ID.
    class InspectOnly:
        def score_many(
            self, requests: Sequence[wyat.Request]
        ) -> tuple[tuple[float, ...], ...]:
            return tuple(
                tuple(
                    2.0 if item.id == "inspect" else 0.0 for item in request.candidates
                )
                for request in requests
            )

    assert (
        run_agent(
            wyat.Model(InspectOnly(), model_id="repeat"),
            wyat.Model(PracticeLargerScorer(), model_id="larger"),
        )
        == "turn_limit"
    )
    output = capsys.readouterr().out
    assert output.count("Final choice: inspect") == 6
    assert "No success claimed" in output
    assert "PASS!" not in output


@pytest.mark.parametrize("scores", [(0.1, 0.0), (2.0, 2.0), (-4.0, -4.1)])
def test_close_scores_and_ties_use_larger_model(scores: tuple[float, ...]) -> None:
    small = FixedScores(scores)
    larger = FixedScores((0.0, 2.0))
    result = choose_action(
        ToyAgent().request(),
        wyat.Model(small, "small"),
        wyat.Model(larger, "larger"),
    )
    assert result.model_id == "larger"
    assert result.selected == "ask"
    assert small.calls == larger.calls == 1


def test_clear_score_gap_keeps_small_model() -> None:
    small = FixedScores((0.0, 1.0))
    larger = FixedScores((2.0, 0.0))
    result = choose_action(
        ToyAgent().request(),
        wyat.Model(small, "small"),
        wyat.Model(larger, "larger"),
    )
    assert result.model_id == "small"
    assert result.selected == "ask"
    assert small.calls == 1 and larger.calls == 0


def test_larger_failure_does_not_execute_an_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Unavailable:
        def score_many(
            self, requests: Sequence[wyat.Request]
        ) -> Sequence[Sequence[float]]:
            raise RuntimeError("larger model unavailable")

    def unexpected_action(self: ToyAgent) -> str:
        raise AssertionError("No action may run after failed routing")

    monkeypatch.setattr(ToyAgent, "inspect_code", unexpected_action)
    monkeypatch.setattr(ToyAgent, "ask_for_help", unexpected_action)
    with pytest.raises(RuntimeError, match="larger model unavailable"):
        run_agent(
            wyat.Model(FixedScores((0.1, 0.0)), "small"),
            wyat.Model(Unavailable(), "larger"),
        )


@pytest.mark.parametrize("threshold", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_threshold_does_not_call_either_model(threshold: float) -> None:
    scorer = FixedScores((2.0, 0.0))
    model = wyat.Model(scorer, "unused")
    with pytest.raises(ValueError, match="positive and finite"):
        choose_action(ToyAgent().request(), model, model, min_score_gap=threshold)
    assert scorer.calls == 0
