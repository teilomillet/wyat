"""Tiny production prefix smoke in an environment with no Torch installation."""

from __future__ import annotations

import importlib.util
import json
import sys

import mlx.core as mx
from mlx_lm.models.qwen3 import ModelArgs, Qwen3Model

from wyat.backends.kai.encoding import KaiEncoder
from wyat.backends.kai.mlx import MLXCandidateHead, MLXKaiScorer
from wyat.backends.kai.prefix import PrefixScorer
from wyat.core import Candidate, Model, Request


class CharacterTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return [ord(character) % 128 for character in text]


with mx.stream(mx.cpu):
    mx.random.seed(53)
    backbone = Qwen3Model(
        ModelArgs("qwen3", 16, 2, 32, 2, 1e-6, 128, 1, 512, 1e6, 16, True)
    )
    head = MLXCandidateHead(16, 8)
    backbone.eval()
    head.eval()
base = MLXKaiScorer(
    backbone,
    head,
    KaiEncoder(CharacterTokenizer(), max_length=512, max_options=8),
    model_id="torch-free-prefix-mechanics",
    device=mx.Device(mx.cpu),
)
request = Request(
    "Pick", (Candidate("a", "First"), Candidate("b", "Second")), "Shared context"
)
scorer = PrefixScorer(base, base.encode_request(request).ids[:25])
decision = Model(scorer, scorer.model_id).decide(request)
reference = Model(base, base.model_id).decide(request)
if (
    importlib.util.find_spec("torch") is not None
    or "torch" in sys.modules
    or decision.selected != reference.selected
    or decision.ranking != reference.ranking
    or any(
        abs(left - right) > 3e-6
        for left, right in zip(decision.scores, reference.scores, strict=True)
    )
):
    raise SystemExit(1)
print(
    json.dumps(
        {
            "passed": True,
            "torch_available": False,
            "torch_imported": False,
            "device": "cpu",
            "model_id": decision.model_id,
            "scores": decision.scores,
            "reference_scores": reference.scores,
            "scope": "Tiny mechanics only; no real checkpoint or performance inference",
        }
    )
)
