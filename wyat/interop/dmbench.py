"""Expose a loaded choice model through explicit DMBench projection recipes."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, cast

from wyat.core import Candidate, Decision, Model, Request

if TYPE_CHECKING:
    from dmbench.api import Prediction as DMBenchPrediction
    from dmbench.api import Request as DMBenchRequest

type DMBenchRecipe = Literal["dmbench-verbatim-choice-v1", "dmbench-native-choice-v1"]


def as_dmbench(
    model: Model, *, recipe: DMBenchRecipe = "dmbench-verbatim-choice-v1"
) -> Callable[[DMBenchRequest], DMBenchPrediction]:
    """Preserve choices; use public text verbatim unless native roles are selected.

    DMBench owns loading cases, labels, scoring, grouping and provenance. A
    DMBench task identifies its panel; Wyat's task identifies the choice head.
    Choiceless tasks cannot be represented by this adapter. Inference errors
    propagate to the DMBench evaluator, which retains their denominator.
    """
    batch = as_dmbench_batch(model, recipe=recipe)

    def predict(public: DMBenchRequest) -> DMBenchPrediction:
        return batch((public,))[0]

    return predict


def _request(public: DMBenchRequest, recipe: DMBenchRecipe) -> Request:
    query = public.text
    context = ""
    if recipe == "dmbench-native-choice-v1":
        if public.task != "jevbench/choice":
            raise ValueError("Wyat's native recipe requires jevbench/choice")
        from dmbench.adapters.native import native_request
        from dmbench.api import canonical_json

        state, question, _ = native_request(public)
        # native_request validates the declared schema, including this text type.
        query = cast(str, question["instructions"])
        context = state if isinstance(state, str) else canonical_json(state)
    if not public.choices:
        raise ValueError("Wyat's choice adapter requires supplied DMBench choices")
    return Request(
        query=query,
        context=context,
        candidates=tuple(
            Candidate(choice.id, choice.text) for choice in public.choices
        ),
        task="choice",
    )


def as_dmbench_batch(
    model: Model, *, recipe: DMBenchRecipe = "dmbench-verbatim-choice-v1"
) -> Callable[[tuple[DMBenchRequest, ...]], tuple[DMBenchPrediction, ...]]:
    """Expose native batching; validate the entire public chunk before inference.

    Inputs and predictions retain caller order. No concurrency, retry, loading
    or hidden menu reduction is introduced. Scalar and batch calls share the
    same public request and prediction conversions.
    """
    if recipe not in {"dmbench-verbatim-choice-v1", "dmbench-native-choice-v1"}:
        raise ValueError("Unknown DMBench choice adapter recipe")
    from dmbench.api import Prediction

    def prediction(public: DMBenchRequest, decision: Decision) -> DMBenchPrediction:
        return Prediction(
            answer=decision.selected,
            probabilities=dict(
                zip(
                    (choice.id for choice in public.choices),
                    decision.probabilities,
                    strict=True,
                )
            ),
        )

    def predict_many(
        public: tuple[DMBenchRequest, ...],
    ) -> tuple[DMBenchPrediction, ...]:
        if not public:
            return ()
        requests = tuple(_request(request, recipe) for request in public)
        decisions = model.decide_many(requests)
        return tuple(
            prediction(request, decision)
            for request, decision in zip(public, decisions, strict=True)
        )

    return predict_many
