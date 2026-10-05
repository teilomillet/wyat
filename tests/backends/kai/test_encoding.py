"""Tokenizer metadata regressions, independent of model weights and devices."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("tokenizers")
pytest.importorskip("transformers")

from tokenizers import AddedToken, Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import ByteLevel, Whitespace
from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer

from wyat.backends.kai.artifact import normalize_config
from wyat.backends.kai.encoding import load_tokenizer
from wyat.wire import JsonObject, JsonValue, read_object


def write_json(path: Path, value: JsonValue) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def tokenizer_root(tmp_path: Path) -> Path:
    tokenizer = Tokenizer(
        WordLevel(
            {"[UNK]": 0, "hello": 1, "<": 2, "special": 3, ">": 4},
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.add_special_tokens([AddedToken("<special>", special=True)])
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    write_json(
        tmp_path / "tokenizer_config.json",
        {
            "tokenizer_class": "TokenizersBackend",
            "unk_token": "[UNK]",
            "pad_token": "[UNK]",
        },
    )
    return tmp_path


@pytest.mark.parametrize(
    ("fields", "text", "expected"),
    [
        ({"split_special_tokens": True}, "<special>", [2, 3, 4]),
        ({"eos_token": "<new>"}, "<new>", [6]),
        ({"extra_special_tokens": ["hello"]}, "helloX", [1, 0]),
        (
            {"added_tokens_decoder": {"6": {"content": "<new>"}}},
            "<new>",
            [6],
        ),
    ],
)
def test_configured_tokens_keep_hugging_face_semantics(
    tokenizer_root: Path, fields: JsonObject, text: str, expected: list[int]
) -> None:
    config_path = tokenizer_root / "tokenizer_config.json"
    config = read_object(config_path)
    config.update(fields)
    write_json(config_path, config)

    tokenizer = load_tokenizer(tokenizer_root)

    assert tokenizer.encode(text, add_special_tokens=False) == expected


@pytest.mark.parametrize(
    ("filename", "value"),
    [
        ("special_tokens_map.json", {"eos_token": "<new>"}),
        ("added_tokens.json", {"<new>": 6}),
    ],
)
def test_legacy_token_files_are_consumed(
    tokenizer_root: Path, filename: str, value: JsonObject
) -> None:
    write_json(tokenizer_root / filename, value)

    tokenizer = load_tokenizer(tokenizer_root)

    assert tokenizer.encode("<new>", add_special_tokens=False) == [6]


def test_empty_decoder_takes_precedence_over_legacy_files(
    tokenizer_root: Path,
) -> None:
    config_path = tokenizer_root / "tokenizer_config.json"
    config = read_object(config_path)
    config["added_tokens_decoder"] = {}
    write_json(config_path, config)
    write_json(tokenizer_root / "special_tokens_map.json", {"eos_token": "<new>"})
    write_json(tokenizer_root / "added_tokens.json", {"<new>": 6})

    tokenizer = load_tokenizer(tokenizer_root)

    assert tokenizer.encode("<new>", add_special_tokens=False) == [2, 0, 4]


def test_alternate_tokenizer_files_are_explicitly_unsupported(
    tokenizer_root: Path,
) -> None:
    config_path = tokenizer_root / "tokenizer_config.json"
    config = read_object(config_path)
    config["fast_tokenizer_files"] = ["tokenizer.5.0.0.json"]
    write_json(config_path, config)

    with pytest.raises(ValueError):
        load_tokenizer(tokenizer_root)


def test_custom_tokenizer_code_is_not_executed(tokenizer_root: Path) -> None:
    config_path = tokenizer_root / "tokenizer_config.json"
    config = read_object(config_path)
    config["auto_map"] = {"AutoTokenizer": ["evil.EvilTokenizer", "evil.EvilTokenizer"]}
    write_json(config_path, config)
    marker = tokenizer_root / "executed"
    (tokenizer_root / "evil.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).touch()\n"
        "raise RuntimeError('custom code executed')\n",
        encoding="utf-8",
    )

    tokenizer = load_tokenizer(tokenizer_root)

    assert tokenizer.encode("hello", add_special_tokens=False) == [1]
    assert not marker.exists()


@pytest.mark.parametrize("filename", ["tokenizer_config.json", "tokenizer.json"])
def test_malformed_tokenizer_json_has_value_error(
    tokenizer_root: Path, filename: str
) -> None:
    (tokenizer_root / filename).write_text("{malformed", encoding="utf-8")

    with pytest.raises(ValueError):
        load_tokenizer(tokenizer_root)


@pytest.fixture
def qwen_tokenizer_root(tmp_path: Path) -> Path:
    vocab = {symbol: index for index, symbol in enumerate(sorted(ByteLevel.alphabet()))}
    vocab["<|endoftext|>"] = 256
    Qwen2Tokenizer(vocab=vocab, merges=[]).save_pretrained(tmp_path)
    return tmp_path


def test_qwen_prefix_space_comes_from_config(qwen_tokenizer_root: Path) -> None:
    config_path = qwen_tokenizer_root / "tokenizer_config.json"
    config = read_object(config_path)
    config["add_prefix_space"] = True
    write_json(config_path, config)

    tokenizer = load_tokenizer(qwen_tokenizer_root)

    assert tokenizer.encode("hello", add_special_tokens=False) == [
        220,
        71,
        68,
        75,
        75,
        78,
    ]


def test_qwen_reconstructs_nfc_normalization(qwen_tokenizer_root: Path) -> None:
    tokenizer_path = qwen_tokenizer_root / "tokenizer.json"
    raw = read_object(tokenizer_path)
    raw["normalizer"] = None
    write_json(tokenizer_path, raw)

    tokenizer = load_tokenizer(qwen_tokenizer_root)

    assert tokenizer.encode("e\u0301", add_special_tokens=False) == [127, 102]


def test_qwen_resets_named_special_token_properties(
    qwen_tokenizer_root: Path,
) -> None:
    tokenizer_path = qwen_tokenizer_root / "tokenizer.json"
    raw = read_object(tokenizer_path)
    tokens = raw["added_tokens"]
    assert isinstance(tokens, list)
    token = tokens[0]
    assert isinstance(token, dict)
    token["lstrip"] = True
    write_json(tokenizer_path, raw)

    tokenizer = load_tokenizer(qwen_tokenizer_root)

    assert tokenizer.encode(" <|endoftext|>", add_special_tokens=False) == [
        220,
        256,
    ]


@pytest.mark.parametrize("activation", ["relu", "gelu", "gelu_new"])
def test_backbone_activation_changes_are_rejected(activation: str) -> None:
    with pytest.raises(ValueError, match="SiLU"):
        normalize_config({"model_type": "qwen3", "hidden_act": activation})


def test_unscaled_rotary_parameters_preserve_theta() -> None:
    config = normalize_config(
        {
            "model_type": "qwen3",
            "rope_parameters": {"rope_type": "default", "rope_theta": 1000000.0},
        }
    )

    assert config["rope_theta"] == 1000000.0
    assert "rope_parameters" not in config
