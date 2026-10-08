"""Read results by exact candidate ID, including Unicode IDs.

The fixed logits demonstrate the API, not trained quality. Softmax probabilities
depend on the supplied candidates; they are not calibrated confidence.
"""

import json
from collections.abc import Sequence

import wyat


class FixedScorer:
    def score_many(
        self, requests: Sequence[wyat.Request]
    ) -> tuple[tuple[float, ...], ...]:
        return tuple((0.0, 2.0, 1.0) for _ in requests)


def main() -> None:
    model = wyat.Model(FixedScorer(), model_id="demo-fixed-logits")
    decision = model.decide(
        wyat.Request(
            query="Choose the requested action",
            context="Create a booking",
            candidates=[
                wyat.Candidate("cancel", "Cancel the booking"),
                wyat.Candidate("予約", "Create a booking"),
                wyat.Candidate("استرداد", "Refund the booking"),
            ],
        )
    )
    print(
        json.dumps(
            {
                "candidate_ids": decision.candidate_ids,
                "selected": decision.selected,
                "ranking": decision.ranking,
                "scores_by_id": decision.scores_by_id,
                "probabilities_by_id": decision.probabilities_by_id,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
