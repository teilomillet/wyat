"""Fictional human-study scorers; no weights, downloads or business actions."""

import time
from collections.abc import Sequence

import wyat


class FixtureScorer:
    def __init__(self, mode: str) -> None:
        if mode not in {
            "rank",
            "last",
            "length",
            "malformed",
            "backend",
            "slow",
            "evaluation",
        }:
            raise ValueError("unknown fixture mode")
        self.mode = mode
        self.calls = 0

    def score_many(self, requests: Sequence[wyat.Request]) -> Sequence[Sequence[float]]:
        self.calls += 1
        if self.mode == "backend":
            raise RuntimeError("fictional backend unavailable")
        if self.mode == "length" and any(
            len(request.context) > 80 for request in requests
        ):
            raise wyat.ModelInputError(
                "fixture context exceeds 80 characters", reason="input-length"
            )
        if self.mode == "slow":
            time.sleep(2)
        if self.mode == "evaluation" and any(
            request.context == "failure" for request in requests
        ):
            raise RuntimeError("fictional prediction failure")
        if self.mode == "malformed":
            return [() for _ in requests]
        if self.mode == "rank":
            if any(len(request.candidates) != 3 for request in requests):
                raise ValueError("rank fixture requires three candidates")
            return [(0.0, 2.0, 2.0) for _ in requests]
        if self.mode == "evaluation":
            return [
                tuple(float(index == 0) for index in range(len(request.candidates)))
                for request in requests
            ]
        return [
            tuple(
                float(index == len(request.candidates) - 1)
                for index in range(len(request.candidates))
            )
            for request in requests
        ]


def make_scorer(mode: str) -> FixtureScorer:
    return FixtureScorer(mode)


def make_model(mode: str) -> wyat.Model:
    return wyat.Model(make_scorer(mode), model_id=f"human-study-fixed-{mode}")
