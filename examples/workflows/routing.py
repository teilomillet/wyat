# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Fixed Choice scorer -> permitted queue -> toy effect and explicit recovery.

No weights or external effects. The retry is safe only because this fixture
fails before its effect; real actions need their own recovery contract.
"""

import json
from collections.abc import Sequence

import wyat


class FixtureScorer:
    def score_many(
        self, requests: Sequence[wyat.Request]
    ) -> tuple[tuple[float, ...], ...]:
        rows: list[tuple[float, ...]] = []
        for request in requests:
            if request.context == "backend unavailable":
                raise RuntimeError("fixture backend unavailable")
            if request.context == "invalid output":
                rows.append((0.0,))
            elif request.context == "needs review":
                rows.append((0.0, -1.0, 2.0))
            else:
                rows.append((0.0, 2.0, -1.0))
        return tuple(rows)


def prepare_request(context: str) -> wyat.Request:
    return wyat.Request(
        query="Choose the permitted support queue.",
        context=context,
        candidates=(
            wyat.Candidate("technical", "Technical support"),
            wyat.Candidate("billing", "Billing support"),
            wyat.Candidate("review", "Human review"),
        ),
    )


def permitted_queue(request: wyat.Request, decision: wyat.Decision) -> str:
    # Here the result came directly from Model.decide(request). An imported
    # result would additionally need authenticated producer and input binding.
    allowed = {candidate.id for candidate in request.candidates}
    if decision.selected not in allowed or decision.selected not in {
        "technical",
        "billing",
        "review",
    }:
        raise ValueError("selected ID is not a permitted queue")
    return decision.selected


def enqueue(
    effects: list[dict[str, str]],
    event_id: str,
    queue: str,
    *,
    fail_before_effect: bool = False,
) -> None:
    existing = next((item for item in effects if item["event_id"] == event_id), None)
    if existing is not None:
        if existing["queue"] != queue:
            raise ValueError("event already routed to a different queue")
        return
    if fail_before_effect:
        raise RuntimeError("fixture queue unavailable before effect")
    effects.append({"event_id": event_id, "queue": queue})


def run_demo() -> dict[str, object]:
    model = wyat.Model(FixtureScorer(), "routing-fixed-logits")
    effects: list[dict[str, str]] = []
    reviews: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    input_rejections: list[dict[str, str]] = []

    # Preparation has its own boundary: scorer ValueError is not input rejection.
    try:
        wyat.Request(
            query="Choose a queue",
            candidates=(
                wyat.Candidate("billing", "Billing"),
                wyat.Candidate("billing", "Duplicate ID"),
            ),
        )
    except ValueError as error:
        input_rejections.append({"stage": "prepare", "error": str(error)})

    request = prepare_request("Duplicate invoice.")
    decision = model.decide(request)
    effects_after_selection = list(effects)
    queue = permitted_queue(request, decision)
    try:
        enqueue(effects, "event-1", queue, fail_before_effect=True)
    except RuntimeError as error:
        failures.append(
            {
                "event_id": "event-1",
                "stage": "action",
                "error": str(error),
                "effect_status": "failed_before_effect",
            }
        )
    effects_after_failure = list(effects)
    # Simulated operator explicitly retries the retained selection, then replays it.
    enqueue(effects, "event-1", queue)
    enqueue(effects, "event-1", queue)

    review = model.decide(prepare_request("needs review"))
    if permitted_queue(prepare_request("needs review"), review) == "review":
        reviews.append({"event_id": "event-2", "reason": "model_selected_review"})

    for event_id, context in (
        ("event-3", "backend unavailable"),
        ("event-4", "invalid output"),
    ):
        try:
            model.decide(prepare_request(context))
        except wyat.ModelOutputError as error:
            failures.append(
                {"event_id": event_id, "stage": "output", "error": str(error)}
            )
        except RuntimeError as error:
            failures.append(
                {"event_id": event_id, "stage": "inference", "error": str(error)}
            )

    unknown = wyat.Decision("untrusted-replay", (2.0,), (1.0,), ("delete-account",))
    try:
        permitted_queue(request, unknown)
    except ValueError:
        unknown_id_blocked = True
    else:
        unknown_id_blocked = False
    return {
        "inference": "fixed scorer; no weights",
        "input": {
            "query": request.query,
            "context": request.context,
            "candidate_ids": [candidate.id for candidate in request.candidates],
        },
        "result": decision.to_dict(),
        "effects_after_selection": effects_after_selection,
        "effects_after_failure": effects_after_failure,
        "effects": effects,
        "reviews": reviews,
        "operational_failures": failures,
        "input_rejections": input_rejections,
        "recovery": {"event_id": "event-1", "action": queue, "status": "succeeded"},
        "unknown_id_blocked": unknown_id_blocked,
        "limitation": "in-memory idempotency; no durable or external recovery claim",
    }


def main() -> None:
    print(json.dumps(run_demo(), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
