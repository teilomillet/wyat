"""Repeatable agent interpretation fixtures; no learned language-quality claim."""

import json
from collections.abc import Sequence

import pytest

from wyat import Candidate, Model, Request
from wyat.core import request_from_dict
from wyat.tasks import NoulAnswer, ScoreAnswer, interpret, public_questions
from wyat.wire import decode_json


@pytest.mark.parametrize(
    "identifiers,instruction",
    [
        (("attendre", "réserver", "annuler"), "Choisir l’action autorisée."),
        (("é", "e\u0301", "review"), "Preserve each exact application ID."),
        (("待機", "予約_查询_예약", "確認"), "許可された操作を選択してください。"),
        (("انتظار", "حجز_عاجل", "مراجعة"), "اختر الإجراء المسموح."),
        (("👩🏽\u200d💻", "予約-🧪", "⏸️"), "Choose the permitted action."),
        (("hold", "facturé_日本_عربية", "review"), "Choisir / 選択 / اختر"),
    ],
)
def test_exact_ids_and_interpretation_across_script_families(
    identifiers: tuple[str, str, str], instruction: str
) -> None:
    class FixedScorer:
        def score_many(self, requests: Sequence[Request]) -> Sequence[Sequence[float]]:
            return [(0.0, 2.0, 2.0) for _ in requests]

    body = {
        "query": instruction,
        "context": "é e\u0301 日本語 العربية 🧪",
        "candidates": [{"id": key, "text": key} for key in identifiers],
    }
    request = request_from_dict(decode_json(json.dumps(body, ensure_ascii=False)))
    decision = Model(FixedScorer(), "fixed-agent-contract").decide(request)
    assert decision.selected == identifiers[1]
    assert decision.ranking == (identifiers[1], identifiers[2], identifiers[0])
    assert decision.candidate_ids == identifiers
    assert decision.scores_by_id == dict(zip(identifiers, (0.0, 2.0, 2.0), strict=True))
    assert request.query == instruction
    assert request.context == body["context"]
    assert (
        json.loads(json.dumps(decision.to_dict(), ensure_ascii=False))["selected"]
        == identifiers[1]
    )


def test_ordinal_mean_does_not_establish_the_likely_middle_level() -> None:
    _, questions = public_questions(
        {
            "state": "Fixture only",
            "questions": {
                "urgence": {"type": "score", "criteria": ["Low", "Medium", "High"]}
            },
        }
    )
    answer = interpret(questions[0], (0.0, -1000.0, 0.0))
    assert isinstance(answer, ScoreAnswer)
    assert answer.expected_ordinal_value == 1.0
    assert answer.probabilities_by_id == {"0": 0.5, "1": 0.0, "2": 0.5}


def test_noul_uses_true_id_and_full_precision_for_application_policy() -> None:
    _, questions = public_questions(
        {
            "state": "Fixture only",
            "questions": {
                "確認": {
                    "type": "noul",
                    "criteria": {
                        "true": "Oui / はい / نعم",
                        "false": "Non / いいえ / لا",
                    },
                }
            },
        }
    )
    answer = interpret(questions[0], (0.0, 1.3862943611198906))
    assert isinstance(answer, NoulAnswer)
    assert answer.option_ids == ("true", "false")
    assert answer.probability_true == answer.probabilities_by_id["true"]
    assert answer.probability_true <= 0.2
    assert answer.probability_true != answer.probabilities[-1]


def test_protocol_keys_remain_fixed_when_descriptions_are_translated() -> None:
    with pytest.raises(ValueError, match="true and false"):
        public_questions(
            {
                "state": "Fixture only",
                "questions": {
                    "confirmation": {
                        "type": "noul",
                        "criteria": {"vrai": "Oui", "faux": "Non"},
                    }
                },
            }
        )
    with pytest.raises(ValueError, match="unique"):
        Request("Choose", [Candidate("é", "First"), Candidate("é", "Duplicate")])
