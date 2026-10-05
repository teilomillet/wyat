"""Show the scorer interface with fixed logits; no trained model or dependencies."""

from collections.abc import Sequence

import wyat


class FixedScorer:
    def score_many(
        self, requests: Sequence[wyat.Request]
    ) -> tuple[tuple[float, ...], ...]:
        return tuple((2.0, 0.0) for _ in requests)


def main() -> None:
    model = wyat.Model(FixedScorer(), model_id="demo-fixed-logits")
    request = wyat.Request(
        query="Choose the requested action",
        context="Cancel my booking",
        candidates=(
            wyat.Candidate("cancel", "Cancel the booking"),
            wyat.Candidate("create", "Create a booking"),
        ),
    )
    print(model.decide(request).to_dict())


if __name__ == "__main__":
    main()
