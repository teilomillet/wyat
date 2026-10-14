# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Watch a tiny coding agent ask Wyat what to do, then do the chosen action."""

import argparse
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, cast

import wyat

MIN_SCORE_GAP = 1.0
DEFAULT_TEMPERATURE = 1.0


class PracticeScorer:
    """Hand-written scores: clear inspection, uncertain fix, clear test."""

    def score_many(
        self, requests: Sequence[wyat.Request]
    ) -> tuple[tuple[float, ...], ...]:
        rows = []
        for request in requests:
            ids = {candidate.id for candidate in request.candidates}
            selected = "test" if "test" in ids else "inspect"
            # Before the fix, this small practice chooser makes a close call.
            score = 0.1 if "fix" in ids else 2.0
            rows.append(
                tuple(
                    score if candidate.id == selected else 0.0
                    for candidate in request.candidates
                )
            )
        return tuple(rows)


class PracticeLargerScorer:
    """Supply the planned next action when the small chooser asks for help."""

    def score_many(
        self, requests: Sequence[wyat.Request]
    ) -> tuple[tuple[float, ...], ...]:
        rows = []
        for request in requests:
            ids = {candidate.id for candidate in request.candidates}
            selected = "fix" if "fix" in ids else "test" if "test" in ids else "inspect"
            rows.append(
                tuple(
                    2.0 if item.id == selected else 0.0 for item in request.candidates
                )
            )
        return tuple(rows)


def choose_action(
    request: wyat.Request,
    small_model: wyat.Model,
    larger_model: wyat.Model,
    *,
    min_score_gap: float = MIN_SCORE_GAP,
    min_probability_margin: float | None = None,
    max_normalized_entropy: float | None = None,
) -> wyat.Decision:
    """Ask the larger model when the small model's top two scores are close."""
    if not math.isfinite(min_score_gap) or min_score_gap <= 0:
        raise ValueError("min_score_gap must be positive and finite")
    if min_probability_margin is not None and (
        not math.isfinite(min_probability_margin)
        or min_probability_margin <= 0
        or min_probability_margin > 1.0
    ):
        raise ValueError("min_probability_margin must be in (0, 1]")
    if max_normalized_entropy is not None and (
        not math.isfinite(max_normalized_entropy)
        or max_normalized_entropy <= 0
        or max_normalized_entropy > 1.0
    ):
        raise ValueError("max_normalized_entropy must be in (0, 1]")
    if len(request.candidates) < 2:
        raise ValueError("Score-gap routing needs at least two choices")
    small = small_model.decide(request)
    gap = small.score_gap
    print(f"Small Wyat picks: {small.selected}. Score gap: {gap:.2f}.")
    if gap < min_score_gap:
        print(f"Router: gap below {min_score_gap:.2f}; ask the larger model.")
        decision = larger_model.decide(request)
        print(f"Larger model picks: {decision.selected}.")
        return decision
    if (
        min_probability_margin is not None
        and small.probability_margin < min_probability_margin
    ):
        print(
            f"Router: margin {small.probability_margin:.2f} below "
            f"{min_probability_margin:.2f}; ask the larger model."
        )
        decision = larger_model.decide(request)
        print(f"Larger model picks: {decision.selected}.")
        return decision
    if (
        max_normalized_entropy is not None
        and small.normalized_entropy > max_normalized_entropy
    ):
        print(
            f"Router: normalized entropy {small.normalized_entropy:.2f} above "
            f"{max_normalized_entropy:.2f}; ask the larger model."
        )
        decision = larger_model.decide(request)
        print(f"Larger model picks: {decision.selected}.")
        return decision
    print(f"Router: gap at least {min_score_gap:.2f}; keep the small model's choice.")
    return small


class ToyAgent:
    """Own the tiny program, its actions, and the result of each action."""

    def __init__(self) -> None:
        self.code = "def add(a, b):\n    return a - b\n"
        self.inspected = False
        self.patched = False
        self.status = "running"
        self.observation = "The user says add(2, 3) returns -1. It should return 5."

    def inspect_code(self) -> str:
        self.inspected = True
        self.observation = (
            f"I looked at the code. It says {self.code.splitlines()[-1].strip()}."
        )
        return self.observation

    def apply_known_fix(self) -> str:
        # The example supplies this patch; Wyat chooses when to use it.
        self.code = self.code.replace("a - b", "a + b")
        self.patched = True
        self.observation = "I changed return a - b to return a + b. No test run yet."
        return self.observation

    def run_test(self) -> str:
        # Only this example's own tiny program is executed, in memory.
        namespace: dict[str, object] = {}
        exec(self.code, namespace)
        add = cast(Callable[[int, int], int], namespace["add"])
        result = add(2, 3)
        self.observation = f"I ran add(2, 3). Expected: 5. Got: {result}."
        if result == 5:
            self.status = "passed"
            return self.observation + " PASS!"
        return self.observation + " FAIL. The bug is still here."

    def ask_for_help(self) -> str:
        self.status = "needs_help"
        self.observation = "I stopped and asked a person for help."
        return self.observation

    def request(self) -> wyat.Request:
        choices = [wyat.Candidate("inspect", "Look at the code to find the bug.")]
        if self.inspected:
            if not self.patched:
                choices.append(
                    wyat.Candidate("fix", "Replace subtraction with addition.")
                )
            choices.append(wyat.Candidate("test", "Run a check to see if 2 + 3 is 5."))
        choices.append(wyat.Candidate("ask", "Stop and ask a person for help."))
        return wyat.Request(
            query="Choose the next action to fix the addition bug and check the fix.",
            context=self.observation,
            candidates=choices,
        )


def run_agent(
    small_model: wyat.Model,
    larger_model: wyat.Model,
    *,
    min_score_gap: float = MIN_SCORE_GAP,
    min_probability_margin: float | None = None,
    max_normalized_entropy: float | None = None,
) -> str:
    agent = ToyAgent()
    actions = {
        "inspect": agent.inspect_code,
        "fix": agent.apply_known_fix,
        "test": agent.run_test,
        "ask": agent.ask_for_help,
    }
    print("Goal: fix a tiny program so 2 + 3 returns 5.")
    print("Wyat chooses an action's name. The agent runs that action.")

    for turn in range(1, 7):
        request = agent.request()
        print(f"\nTurn {turn}")
        print(f"Agent knows: {request.context}")
        print("Agent offers: " + ", ".join(item.id for item in request.candidates))

        decision = choose_action(
            request,
            small_model,
            larger_model,
            min_score_gap=min_score_gap,
            min_probability_margin=min_probability_margin,
            max_normalized_entropy=max_normalized_entropy,
        )
        action = actions[decision.selected]
        print(f"Final choice: {decision.selected}")
        print(f"Agent runs: {action.__name__}()")
        print(f"Result: {action()}")

        if agent.status != "running":
            print(f"\nAgent stopped: {agent.status}.")
            return agent.status

    print("\nAgent stopped: reached the six-turn limit. No success claimed.")
    return "turn_limit"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--demo", action="store_true", help="Use the scripted practice chooser"
    )
    source.add_argument("--model", type=Path, help="Use trained local Wyat weights")
    parser.add_argument("--larger-model", type=Path, help="Local fallback Choice model")
    parser.add_argument(
        "--min-score-gap",
        type=float,
        default=MIN_SCORE_GAP,
        help="Ask the larger model below this score gap (default: %(default)s)",
    )
    parser.add_argument(
        "--min-probability-margin",
        type=float,
        default=None,
        help="Ask the larger model below this top-2 probability margin in (0, 1]",
    )
    parser.add_argument(
        "--max-normalized-entropy",
        type=float,
        default=None,
        help="Ask the larger model above this normalized entropy in (0, 1]",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Softmax temperature for the small model (default: %(default)s)",
    )
    parser.add_argument("--backend", choices=("torch", "mlx"), default="torch")
    arguments = parser.parse_args()
    if arguments.demo:
        if arguments.larger_model is not None:
            parser.error("Use --model with --larger-model, or --demo on its own")
        print("Practice mode: hand-written scores, no trained models.")
        print("Watch a close call go to the larger practice chooser.\n")
        model = wyat.Model(
            PracticeScorer(),
            model_id="embedded-practice-small",
            temperature=arguments.temperature,
        )
        larger = wyat.Model(PracticeLargerScorer(), model_id="embedded-practice-larger")
    else:
        if arguments.larger_model is None:
            parser.error("--model requires --larger-model for uncertain choices")
        model = wyat.load(
            cast(Path, arguments.model),
            backend=cast(Literal["torch", "mlx"], arguments.backend),
            temperature=arguments.temperature,
        )
        larger = wyat.load(
            cast(Path, arguments.larger_model),
            backend=cast(Literal["torch", "mlx"], arguments.backend),
        )
        print("Trained model mode: choices can differ from the practice script.\n")
    run_agent(
        model,
        larger,
        min_score_gap=arguments.min_score_gap,
        min_probability_margin=arguments.min_probability_margin,
        max_normalized_entropy=arguments.max_normalized_entropy,
    )


if __name__ == "__main__":
    main()
