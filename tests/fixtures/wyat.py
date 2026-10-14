# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Small, seeded offline bundles for mechanics tests; these have no learned quality."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3Model

from wyat.backends.wyat.torch import ARCHITECTURE, PROMPT_VERSION, CandidateHead


def create_wyat_bundle(path: Path, seed: int = 7) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    tokens = [
        "[UNK]",
        "[PAD]",
        "Context",
        "Task",
        "type",
        "choice",
        "Question",
        "Options",
        "Select",
        "the",
        "single",
        "option",
        "best",
        "supported",
        "by",
        "context",
        "and",
        "instructions",
        "Decision",
        "key",
        "description",
        "true",
        "false",
        "a",
        "b",
        "c",
        "yes",
        "no",
        "read",
        "write",
        "Bonjour",
        "世界",
        "é",
        "?",
        ":",
        "<",
        ">",
        "/",
        '"',
        "{",
        "}",
        ",",
        "</",
        ">\n",
    ]
    vocabulary = {token: index for index, token in enumerate(tokens)}
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    wrapped = PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]  # HF boundary.
        tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]"
    )
    wrapped.save_pretrained(path)
    config = Qwen3Config(
        vocab_size=len(tokens),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=512,
        rope_parameters={"rope_type": "default", "rope_theta": 1000000.0},
        use_cache=False,
        attention_dropout=0.0,
    )
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        backbone = Qwen3Model(config)
        head = CandidateHead(16, 8)
    backbone.save_pretrained(path / "backbone", safe_serialization=True)
    save_file(head.state_dict(), path / "decision_head.safetensors")
    metadata = {
        "architecture": ARCHITECTURE,
        "prompt_version": PROMPT_VERSION,
        "head_variant": "shared",
        "checkpoint_format": "full",
        "head_dim": 8,
        "max_options": 255,
        "head_compute_dtype": "float32",
    }
    (path / "decision_config.json").write_text(json.dumps(metadata), encoding="utf-8")
    return path
