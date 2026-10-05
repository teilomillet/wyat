from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence

import pytest
from hypothesis import given
from hypothesis import strategies as st

from wyat.core import Model, Request
from wyat.interop.dmbench import as_dmbench, as_dmbench_batch

pytest.importorskip("dmbench", reason="DMBench interoperability is optional")

from dmbench.api import Case, Choice  # noqa: E402
from dmbench.api import Request as DMBenchRequest  # noqa: E402
from dmbench.evaluation import evaluate  # noqa: E402


class RecordingScorer:
    def __init__(self, *, fail: bool = False) -> None:
        self.requests: list[Request] = []
        self.batch_sizes: list[int] = []
        self.fail = fail

    def score_many(self, requests: Sequence[Request]) -> tuple[tuple[float, ...], ...]:
        self.requests.extend(requests)
        self.batch_sizes.append(len(requests))
        if self.fail:
            raise RuntimeError("inference failed")
        return tuple(
            tuple(float(index) for index, _ in enumerate(request.candidates))
            for request in requests
        )


def public_request(text: str = "Choose a destination.") -> DMBenchRequest:
    return DMBenchRequest(
        "source-id-is-not-model-input",
        "logicbench/MCQA",
        text,
        (
            Choice("route-b", "second destination"),
            Choice("route-a", "first destination"),
        ),
    )


def test_choice_mapping_preserves_text_and_order_without_extra_prompt() -> None:
    scorer = RecordingScorer()
    public = public_request('{"context":"a fact", "question":"which route?"}')
    prediction = as_dmbench(Model(scorer, "fixture"))(public)
    assert prediction.answer == "route-a"
    assert prediction.probabilities is not None
    assert tuple(prediction.probabilities) == ("route-b", "route-a")
    assert prediction.probabilities["route-a"] > prediction.probabilities["route-b"]
    assert prediction.usage is None
    observed = scorer.requests[0]
    assert observed.query == public.text
    assert observed.context == ""
    assert observed.task == "choice"
    assert [(item.id, item.text) for item in observed.candidates] == [
        (choice.id, choice.text) for choice in public.choices
    ]
    assert public.id not in repr(observed)


@given(st.text(min_size=1).filter(lambda text: bool(text.strip())))
def test_public_text_is_never_parsed_or_reformatted(text: str) -> None:
    scorer = RecordingScorer()
    as_dmbench(Model(scorer, "fixture"))(public_request(text))
    assert scorer.requests[0].query == text


def test_actual_dmbench_evaluator_scores_choices_and_isolates_gold() -> None:
    scorer = RecordingScorer()
    case = Case(public_request(), ("route-a",), "group", "source", "development")
    report = evaluate((case,), as_dmbench(Model(scorer, "fixture")))
    assert report["count"] == report["correct"] == 1
    assert report["accuracy"] == 1
    assert len(scorer.requests) == 1
    assert not hasattr(scorer.requests[0], "expected")
    assert not hasattr(scorer.requests[0], "id")


def test_unsupported_and_failed_requests_retain_dmbench_denominator() -> None:
    model = as_dmbench(Model(RecordingScorer(fail=True), "fixture"))
    cases = (
        Case(public_request(), ("route-a",), "one", "source", "development"),
        Case(
            DMBenchRequest("ordinal", "jevbench/score", "Rate the state."),
            (2,),
            "two",
            "source",
            "development",
        ),
    )
    report = evaluate(cases, model)
    assert report["count"] == 2
    assert report["correct"] == 0
    records = report["predictions"]
    assert isinstance(records, list)
    assert all(record["error"] for record in records)
    assert all(record["answer"] is None for record in records)
    assert all(record["probabilities"] is None for record in records)


def test_choiceless_request_is_rejected_before_inference() -> None:
    scorer = RecordingScorer()
    model = as_dmbench(Model(scorer, "fixture"))
    with pytest.raises(ValueError, match="supplied DMBench choices"):
        model(DMBenchRequest("judgment", "judgment", "Assess every action."))
    assert not scorer.requests


def test_optional_adapter_import_does_not_import_dmbench() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import wyat.interop.dmbench; "
            "assert 'dmbench' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_native_batch_preserves_public_order_and_probabilities() -> None:
    scorer = RecordingScorer()
    public = (public_request("first query"), public_request("second query"))
    predictions = as_dmbench_batch(Model(scorer, "fixture"))(public)
    assert scorer.batch_sizes == [2]
    assert [request.query for request in scorer.requests] == [
        item.text for item in public
    ]
    assert len(predictions) == len(public)
    assert all(prediction.answer == "route-a" for prediction in predictions)
    assert all(prediction.probabilities is not None for prediction in predictions)
    assert all(prediction.usage is None for prediction in predictions)


def test_empty_batch_does_not_call_loaded_model() -> None:
    scorer = RecordingScorer()
    assert as_dmbench_batch(Model(scorer, "fixture"))(()) == ()
    assert not scorer.requests and not scorer.batch_sizes


def test_invalid_member_rejects_whole_batch_before_model_call() -> None:
    scorer = RecordingScorer()
    batch = as_dmbench_batch(Model(scorer, "fixture"))
    with pytest.raises(ValueError, match="supplied DMBench choices"):
        batch(
            (public_request(), DMBenchRequest("invalid", "judgment", "Assess state."))
        )
    assert not scorer.requests and not scorer.batch_sizes


def test_actual_dmbench_evaluator_uses_explicit_batches_and_isolates_gold() -> None:
    scorer = RecordingScorer()
    model = Model(scorer, "fixture")
    cases = tuple(
        Case(
            DMBenchRequest(
                f"source-id-{index}",
                "logicbench/MCQA",
                f"public query {index}",
                (Choice("first", "first option"), Choice("second", "second option")),
            ),
            ("second",),
            "group",
            "source",
            "development",
        )
        for index in range(3)
    )
    report = evaluate(
        cases, as_dmbench(model), batch=as_dmbench_batch(model), batch_size=2
    )
    assert report["count"] == report["correct"] == 3
    assert scorer.batch_sizes == [2, 1]
    assert [request.query for request in scorer.requests] == [
        case.request.text for case in cases
    ]
    assert all(not hasattr(request, "expected") for request in scorer.requests)
    assert all(not hasattr(request, "id") for request in scorer.requests)
    records = report["predictions"]
    assert isinstance(records, list)
    assert [record["id"] for record in records] == [case.request.id for case in cases]
    assert all(record["elapsed_seconds"] is None for record in records)
    assert all(record["batch_elapsed_seconds"] >= 0 for record in records)
