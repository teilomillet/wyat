"""Native projection uses the peer's validator; no model weights or devices."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from hypothesis import given
from hypothesis import strategies as st

from wyat.backends.kai.encoding import KaiEncoder
from wyat.core import Model, Request
from wyat.interop.dmbench import DMBenchRecipe, as_dmbench, as_dmbench_batch

pytest.importorskip("dmbench", reason="DMBench interoperability is optional")

from dmbench.adapters import native as peer
from dmbench.api import (
    Case,
    Choice,
    JSONValue,
    canonical_json,
    request_from_dict,
    strict_json_loads,
)
from dmbench.api import Request as DMBenchRequest
from dmbench.evaluation import evaluate

NATIVE: DMBenchRecipe = "dmbench-native-choice-v1"


class RecordingScorer:
    def __init__(self) -> None:
        self.requests: list[Request] = []
        self.batch_sizes: list[int] = []

    def score_many(self, requests: Sequence[Request]) -> tuple[tuple[float, ...], ...]:
        self.requests.extend(requests)
        self.batch_sizes.append(len(requests))
        return tuple(
            tuple(float(index) for index, _ in enumerate(request.candidates))
            for request in requests
        )


def public_request(
    state: JSONValue = "public state",
    *,
    instructions: str = "Choose an action.",
    choices: tuple[Choice, ...] = (Choice("z", "Zulu"), Choice("a", "Alpha")),
) -> DMBenchRequest:
    criteria: dict[str, JSONValue] = {choice.id: choice.text for choice in choices}
    question: dict[str, JSONValue] = {
        "type": "choice",
        "instructions": instructions,
        "criteria": criteria,
    }
    body: dict[str, JSONValue] = {
        "state": state,
        "question": question,
        "labels": [choice.id for choice in choices],
    }
    return DMBenchRequest(
        "source-id-not-model-input",
        "jevbench/choice",
        canonical_json(body),
        choices,
    )


@pytest.fixture
def peer_requests() -> tuple[DMBenchRequest, ...]:
    fixture = (
        Path(peer.__file__).resolve().parents[3]
        / "tests/fixtures/native-role-requests.jsonl"
    )
    if not fixture.is_file():
        pytest.skip("Installed DMBench has no source-checkout role fixtures")
    return tuple(
        request_from_dict(strict_json_loads(line))
        for line in fixture.read_text().splitlines()
    )


def test_actual_peer_fixtures_project_roles_without_double_prompt(
    peer_requests: tuple[DMBenchRequest, ...],
) -> None:
    scorer = RecordingScorer()
    predictions = as_dmbench_batch(Model(scorer, "fixture"), recipe=NATIVE)(
        peer_requests
    )
    assert scorer.batch_sizes == [2]
    assert [request.context for request in scorer.requests] == [
        "Please check a parcel status.",
        '{"facts":{"receipt":false},"required":["receipt"]}',
    ]
    assert all(
        request.query == "Choose the action that fits the supplied state."
        and request.task == "choice"
        for request in scorer.requests
    )
    for public, request, prediction in zip(
        peer_requests, scorer.requests, predictions, strict=True
    ):
        assert [(candidate.id, candidate.text) for candidate in request.candidates] == [
            (choice.id, choice.text) for choice in public.choices
        ]
        assert prediction.answer == "decline"
        assert prediction.probabilities is not None
        assert tuple(prediction.probabilities) == ("check", "decline")
        assert prediction.usage is None
        assert public.id not in repr(request)


@given(
    state=st.text(max_size=100),
    instructions=st.text(min_size=1, max_size=100).filter(
        lambda text: bool(text.strip())
    ),
)
def test_string_state_and_instructions_preserve_whitespace_and_unicode(
    state: str, instructions: str
) -> None:
    scorer = RecordingScorer()
    as_dmbench(Model(scorer, "fixture"), recipe=NATIVE)(
        public_request(state, instructions=instructions)
    )
    assert scorer.requests[0].context == state
    assert scorer.requests[0].query == instructions


@pytest.mark.parametrize(
    "state",
    [
        {"z": "é\n", "a": [True, 2, None]},
        [
            {"role": "user", "content": "  日本語\n"},
            {"role": "assistant", "content": "ok"},
        ],
    ],
)
def test_nontext_state_uses_actual_peer_canonical_json(state: JSONValue) -> None:
    scorer = RecordingScorer()
    as_dmbench(Model(scorer, "fixture"), recipe=NATIVE)(public_request(state))
    assert scorer.requests[0].context == canonical_json(state)


def test_default_recipe_keeps_native_envelope_verbatim() -> None:
    scorer = RecordingScorer()
    public = public_request()
    as_dmbench(Model(scorer, "fixture"))(public)
    assert scorer.requests[0].query == public.text
    assert scorer.requests[0].context == ""


def test_actual_evaluator_batches_native_requests_and_keeps_gold_isolated() -> None:
    scorer = RecordingScorer()
    model = Model(scorer, "fixture")
    cases = tuple(
        Case(
            replace(public_request(), id=f"private-id-{index}"),
            ("a",),
            "g",
            "source",
            "dev",
        )
        for index in range(3)
    )
    report = evaluate(
        cases,
        as_dmbench(model, recipe=NATIVE),
        batch=as_dmbench_batch(model, recipe=NATIVE),
        batch_size=2,
    )
    assert report["count"] == report["correct"] == 3
    assert scorer.batch_sizes == [2, 1]
    assert all(not hasattr(request, "expected") for request in scorer.requests)
    assert all(not hasattr(request, "id") for request in scorer.requests)
    assert [request.context for request in scorer.requests] == ["public state"] * 3


@pytest.mark.parametrize("task", ["jevbench/noul", "jevbench/score", "unknown"])
def test_native_choice_recipe_rejects_other_tasks_before_inference(task: str) -> None:
    scorer = RecordingScorer()
    adapter = as_dmbench(Model(scorer, "fixture"), recipe=NATIVE)
    with pytest.raises(ValueError, match="jevbench/choice"):
        adapter(replace(public_request(), task=task))
    assert not scorer.requests


@pytest.mark.parametrize(
    "body",
    [
        {"question": {}, "labels": ["z", "a"]},
        {"state": "state", "question": {}, "labels": ["z", "a"], "gold": "a"},
        {"state": "state", "question": {"type": "choice"}, "labels": ["z", "a"]},
        {
            "state": "state",
            "question": {
                "type": "score",
                "instructions": "Pick.",
                "criteria": {"z": "Zulu", "a": "Alpha"},
            },
            "labels": ["z", "a"],
        },
        {
            "state": "state",
            "question": {
                "type": "choice",
                "instructions": "Pick.",
                "criteria": {"z": "changed", "a": "Alpha"},
            },
            "labels": ["z", "a"],
        },
        {
            "state": None,
            "question": {
                "type": "choice",
                "instructions": "Pick.",
                "criteria": {"z": "Zulu", "a": "Alpha"},
            },
            "labels": ["z", "a"],
        },
    ],
)
def test_peer_validation_rejects_missing_unknown_or_disagreeing_roles(
    body: JSONValue,
) -> None:
    scorer = RecordingScorer()
    public = replace(public_request(), text=canonical_json(body))
    adapter = as_dmbench_batch(Model(scorer, "fixture"), recipe=NATIVE)
    with pytest.raises(ValueError):
        adapter((public_request(), public))
    assert not scorer.requests and not scorer.batch_sizes


def test_actual_peer_menu_mismatch_rejects_whole_batch(
    peer_requests: tuple[DMBenchRequest, ...],
) -> None:
    scorer = RecordingScorer()
    altered = replace(
        peer_requests[1],
        choices=(Choice("check", "Altered criterion"), peer_requests[1].choices[1]),
    )
    with pytest.raises(ValueError, match="text disagree"):
        as_dmbench_batch(Model(scorer, "fixture"), recipe=NATIVE)(
            (peer_requests[0], altered)
        )
    assert not scorer.requests


def test_empty_native_batch_never_calls_model() -> None:
    scorer = RecordingScorer()
    assert as_dmbench_batch(Model(scorer, "fixture"), recipe=NATIVE)(()) == ()
    assert not scorer.requests and not scorer.batch_sizes


def test_unknown_recipe_rejected_at_construction() -> None:
    scorer = RecordingScorer()
    for factory in (as_dmbench, as_dmbench_batch):
        with pytest.raises(ValueError, match="Unknown"):
            factory(Model(scorer, "fixture"), recipe=cast(DMBenchRecipe, "automatic"))
    assert not scorer.requests


class CharacterTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return [ord(character) for character in text]


class EncoderScorer(RecordingScorer):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = KaiEncoder(CharacterTokenizer(), max_length=512, max_options=64)

    def score_many(self, requests: Sequence[Request]) -> tuple[tuple[float, ...], ...]:
        for request in requests:
            self.encoder.encode_request(request)
        return super().score_many(requests)


@pytest.mark.parametrize("boundary", ["choices", "context"])
def test_native_projection_preserves_actual_encoder_limits(boundary: str) -> None:
    scorer = EncoderScorer()
    public = (
        public_request(
            choices=tuple(Choice(str(index), "candidate") for index in range(65))
        )
        if boundary == "choices"
        else public_request("long state " * 100)
    )
    with pytest.raises(ValueError, match="2..64|max_length"):
        as_dmbench_batch(Model(scorer, "fixture"), recipe=NATIVE)(
            (public_request(), public)
        )
    assert not scorer.requests
