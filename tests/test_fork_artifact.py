# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Option-fork artifacts: exact cached inference, refused tampering, public API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
from tokenizers import Tokenizer  # noqa: E402
from tokenizers.decoders import ByteLevel as ByteLevelDecoder  # noqa: E402
from tokenizers.models import BPE  # noqa: E402
from tokenizers.pre_tokenizers import ByteLevel  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402
from transformers.models.qwen3_5.configuration_qwen3_5 import (  # noqa: E402
    Qwen3_5TextConfig,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel  # noqa: E402

import wyat  # noqa: E402
from wyat.loading import fork_artifact  # noqa: E402
from wyat.loading.fork import (  # noqa: E402
    ForkScorer,
    _ScalarHead,
    write_artifact,
)
from wyat.loading.fork_artifact import adapter_name  # noqa: E402

RECORD = {
    "state": {"text": "Bonjour 世界 é", "count": 7},
    "questions": {
        "choice": {
            "type": "choice",
            "instructions": "Which applies?",
            "criteria": {"a": "short", "b": "a much longer option " * 3, "c": "x"},
        },
        "score": {
            "type": "score",
            "instructions": "Rate.",
            "criteria": ["no", "maybe", "yes"],
        },
        "noul": {
            "type": "noul",
            "instructions": "True?",
            "criteria": {"false": "no", "true": "yes"},
        },
    },
}


def _tokenizer(path: Path) -> list[Path]:
    alphabet = ByteLevel.alphabet()
    vocabulary = {symbol: index for index, symbol in enumerate(sorted(alphabet))}
    vocabulary["[PAD]"] = len(vocabulary)
    tokenizer = Tokenizer(BPE(vocabulary, [], unk_token=None))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    wrapped = PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]  # HF boundary.
        tokenizer_object=tokenizer, pad_token="[PAD]"
    )
    wrapped.save_pretrained(path)
    return [path / "tokenizer.json", path / "tokenizer_config.json"]


def _layer_types(config: Qwen3_5TextConfig) -> list[str]:
    assert config.layer_types is not None
    return list(config.layer_types)


def _config(vocabulary: int) -> Qwen3_5TextConfig:
    config = Qwen3_5TextConfig(
        vocab_size=vocabulary,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        layer_types=["linear_attention", "full_attention"] * 2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        max_position_embeddings=4096,
        attention_dropout=0.0,
    )
    return config


def _artifact(
    tmp_path: Path, *, seed: int = 3, config: Qwen3_5TextConfig | None = None
) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    tokenizer_files = _tokenizer(source / "tokenizer")
    config = _config(257) if config is None else config
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        backbone = Qwen3_5TextModel(config)
        head = _ScalarHead(config.hidden_size, 16)
        adapters = {}
        for index, kind in enumerate(_layer_types(config)):
            block = backbone.layers[index]
            for owner, name in fork_artifact.QWEN3_5_ADAPTER_TARGETS[kind]:
                linear = getattr(getattr(block, owner), name)
                adapters[adapter_name(index, owner, name, "down")] = 0.05 * torch.randn(
                    16, linear.in_features
                )
                adapters[adapter_name(index, owner, name, "up")] = 0.05 * torch.randn(
                    linear.out_features, 16
                )
    backbone.save_pretrained(source / "base", safe_serialization=True)
    output = tmp_path / "artifact"
    write_artifact(
        output,
        base=source / "base",
        tokenizer_files=tokenizer_files,
        adapters=adapters,
        head=head.state_dict(),
        max_length=2048,
        max_batch_tokens=4096,
        head_width=16,
        input_rendering=None,
        provenance={"test": True},
    )
    return output


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _artifact(tmp_path_factory.mktemp("fork"))


def _reference_logits(scorer: ForkScorer, row: Any) -> Any:
    """Independent check: recompute prefix + branch in full for every option."""
    values = []
    with torch.no_grad():
        for suffix in row.option_ids:
            ids = torch.tensor([row.prefix_ids + suffix])
            hidden = scorer.backbone(input_ids=ids, use_cache=False).last_hidden_state
            values.append(scorer.head(hidden[:, -1]))
    return torch.cat(values)


def test_cached_branches_equal_full_recompute(artifact: Path) -> None:
    scorer = ForkScorer.from_directory(artifact)
    for row in scorer.encode_record(RECORD):
        cached = torch.tensor(scorer.question_logits(row))
        torch.testing.assert_close(
            cached, _reference_logits(scorer, row), atol=2e-5, rtol=1e-4
        )


def test_many_options_split_into_groups_without_changing_logits(artifact: Path) -> None:
    scorer = ForkScorer.from_directory(artifact)
    scorer.max_batch_tokens = 600
    record = {
        "state": "x" * 40,
        "questions": {
            "q": {
                "type": "choice",
                "instructions": "Pick.",
                "criteria": {f"o{i}": "w" * (1 + i % 9) for i in range(45)},
            }
        },
    }
    (row,) = scorer.encode_record(record)
    groups = fork_artifact.branch_groups(
        row.option_ids, len(row.prefix_ids), scorer.max_batch_tokens
    )
    assert len(list(groups)) > 2
    cached = torch.tensor(scorer.question_logits(row))
    torch.testing.assert_close(
        cached, _reference_logits(scorer, row), atol=2e-5, rtol=1e-4
    )


def test_option_permutation_only_remaps_logits(artifact: Path) -> None:
    scorer = ForkScorer.from_directory(artifact)
    swapped = json.loads(json.dumps(RECORD))
    swapped["questions"]["choice"]["criteria"] = {
        "c": "x",
        "a": "short",
        "b": "a much longer option " * 3,
    }
    left = scorer.logits(RECORD)[0]
    right = scorer.logits(swapped)[0]
    torch.testing.assert_close(
        torch.tensor(right),
        torch.tensor([left[2], left[0], left[1]]),
        atol=2e-5,
        rtol=1e-4,
    )


def test_public_load_predicts_through_the_model_interface(artifact: Path) -> None:
    model = wyat.load(artifact)
    assert model.model_id.startswith("fork-")
    answers = ForkScorer.from_directory(artifact).predict(RECORD)
    assert set(answers) == {"choice", "score", "noul"}


@pytest.mark.parametrize(
    "tamper",
    ["weights", "extra-file", "contract", "adapter"],
)
def test_changed_artifacts_are_refused(tmp_path: Path, tamper: str) -> None:
    root = _artifact(tmp_path)
    manifest_path = root / fork_artifact.MANIFEST
    manifest = json.loads(manifest_path.read_text())
    if tamper == "weights":
        (root / "head.safetensors").write_bytes(
            (root / "head.safetensors").read_bytes() + b"\0"
        )
    elif tamper == "extra-file":
        (root / "notes.txt").write_text("unexpected")
    elif tamper == "contract":
        manifest["contract"]["prompt"] = "other"
        manifest_path.write_text(json.dumps(manifest))
    else:
        manifest["adapter"]["rank"] = 8
        manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        fork_artifact.validate_artifact(root)


def test_adapter_targets_name_every_block_linear() -> None:
    config = _config(64)
    backbone = Qwen3_5TextModel(config)
    for kind, block in zip(_layer_types(config), backbone.layers, strict=True):
        linear = {
            tuple(name.split("."))
            for name, module in block.named_modules()
            if isinstance(module, torch.nn.Linear)
        }
        assert linear == set(fork_artifact.QWEN3_5_ADAPTER_TARGETS[kind])


def _rebuild_manifest(root: Path) -> None:
    previous = json.loads((root / fork_artifact.MANIFEST).read_text())
    (root / fork_artifact.MANIFEST).unlink()
    manifest = fork_artifact.manifest_for(
        root,
        max_length=previous["max_length"],
        max_batch_tokens=previous["max_batch_tokens"],
        head_width=previous["head_width"],
        input_rendering=previous["input_rendering"],
        provenance=previous["provenance"],
    )
    (root / fork_artifact.MANIFEST).write_text(json.dumps(manifest))


def test_a_declared_shard_index_is_not_admitted(tmp_path: Path) -> None:
    # A weight map could name files outside the artifact; even declared and
    # hashed, an index is refused.
    root = _artifact(tmp_path)
    (root / "base" / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": "/elsewhere/model.safetensors"}})
    )
    _rebuild_manifest(root)
    with pytest.raises(ValueError, match="unsupported files"):
        fork_artifact.validate_artifact(root)


def test_symlinked_directories_are_refused(tmp_path: Path) -> None:
    root = _artifact(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "evil.jinja").write_text("OUTSIDE")
    (root / "additional_chat_templates").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        fork_artifact.validate_artifact(root)


def test_undeclared_directories_are_refused(tmp_path: Path) -> None:
    root = _artifact(tmp_path)
    (root / "extra").mkdir()
    (root / "extra" / "notes.txt").write_text("x")
    with pytest.raises(ValueError, match="undeclared directory"):
        fork_artifact.validate_artifact(root)


def test_overlong_records_raise_model_input_error(artifact: Path) -> None:
    from wyat.core import ModelInputError

    scorer = ForkScorer.from_directory(artifact, max_length=64)
    record = {"state": "y" * 400, "questions": RECORD["questions"]}
    for call in (scorer.logits, scorer.predict):
        with pytest.raises(ModelInputError) as raised:
            call(record)
        assert raised.value.reason == "input-length"
    with pytest.raises(ValueError):
        scorer.logits({"state": "x", "questions": {}})


def test_a_refused_request_costs_no_forward_pass(
    artifact: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scorer = ForkScorer.from_directory(artifact)
    passes: list[object] = []
    score = scorer.question_logits

    def counted(row: Any) -> list[float]:
        passes.append(row)
        return score(row)

    monkeypatch.setattr(scorer, "question_logits", counted)
    options = {"a": "short", "b": "x"}
    fits = [wyat.Request.choice("Which?", options, context=t) for t in ("u", "v")]
    long = "y" * (scorer.max_length * 40)
    refused = wyat.Request.choice("Which?", options, context=long)
    with pytest.raises(wyat.ModelInputError) as raised:
        scorer.score_many([*fits, refused])
    assert raised.value.reason == "input-length"
    assert passes == []

    model = wyat.Model(scorer, model_id="fork-test")
    results = model.decide_many([*fits, refused], errors="return")
    assert [type(result) for result in results] == [
        wyat.Decision,
        wyat.Decision,
        wyat.ModelInputError,
    ]
    assert len(passes) == len(fits)


def test_missing_adapter_tensor_is_a_value_error(tmp_path: Path) -> None:
    from safetensors.torch import load_file, save_file

    root = _artifact(tmp_path)
    tensors = load_file(root / "adapters.safetensors")
    tensors.pop(next(iter(tensors)))
    save_file(tensors, root / "adapters.safetensors")
    _rebuild_manifest(root)
    with pytest.raises(ValueError, match="adapter inventory"):
        ForkScorer.from_directory(root)


def test_tokenizer_keeps_combining_marks_with_their_letters(artifact: Path) -> None:
    # Qwen3.5 pre-tokenizes letters with their combining marks; the Qwen2 class
    # named in tokenizer_config.json splits them and would diverge from training.
    scorer = ForkScorer.from_directory(artifact)
    pieces = scorer.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("नमस्ते")
    assert type(scorer.tokenizer).__name__ == "Qwen3_5Tokenizer"
    assert len(pieces) == 1


def test_base_loading_drops_only_the_transformers_load_report() -> None:
    import logging

    from transformers.utils import logging as transformers_logging

    from wyat.loading.fork import _quiet_base_loading

    messages: list[str] = []

    class Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            messages.append(record.getMessage())

    logger = logging.getLogger("transformers.modeling_utils")
    handler = Collect()
    logger.addHandler(handler)
    enabled = transformers_logging.is_progress_bar_enabled()
    try:
        with _quiet_base_loading():
            assert not transformers_logging.is_progress_bar_enabled()
            logger.warning("Qwen3_5TextModel LOAD REPORT from: base")
            logger.warning("another warning")
        logger.warning("later LOAD REPORT")
    finally:
        logger.removeHandler(handler)
    assert messages == ["another warning", "later LOAD REPORT"]
    assert transformers_logging.is_progress_bar_enabled() == enabled


def test_cpu_scoring_drops_only_the_cuda_kernel_advice() -> None:
    import logging

    from wyat.loading.fork import _without_cuda_kernel_advice

    messages: list[str] = []

    class Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            messages.append(record.getMessage())

    logger = logging.getLogger("transformers.integrations.hub_kernels")
    handler = Collect()
    logger.addHandler(handler)
    advice = (
        "`chunk_gated_delta_rule` is falling back to its reference PyTorch "
        "implementation because `flash-linear-attention` is not installed."
    )
    try:
        with _without_cuda_kernel_advice():
            logger.warning(advice)
            logger.warning("another warning")
        logger.warning(advice)
    finally:
        logger.removeHandler(handler)
    assert messages == ["another warning", advice]
