# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Ouro inference-speed paths: exact K/V workspace, batched candidates, streamed load.

The default candidate path must stay bitwise identical to the former
per-candidate ``torch.cat`` cache, so these tests compare against an explicit
cat-cache reference on a tiny in-memory backbone. The opt-in candidate-batching
path is tolerance-gated (max ``|Δlogit|`` <= 2e-5, identical ranking), and the
streaming loader must reproduce the checkpoint values exactly.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    import torch
else:
    torch = pytest.importorskip("torch")
from safetensors.torch import load_file, save_file  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from tokenizers.decoders import ByteLevel as ByteLevelDecoder  # noqa: E402
from tokenizers.models import BPE  # noqa: E402
from tokenizers.pre_tokenizers import ByteLevel  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

from wyat.loading import ouro, ouro_artifact  # noqa: E402
from wyat.loading.ouro import (  # noqa: E402
    OuroScorer,
    ScalarScoreHead,
    attach_step_lora,
    candidate_kv_workspace_bytes,
    load_ouro,
    load_ouro_classes,
    prefill_context,
    run_candidate_steps,
    score_candidates,
    score_candidates_from_cache,
)

SOURCE_FIXTURE = Path(__file__).parent / "fixtures" / "ouro_source"
REPO_ROOT = Path(__file__).resolve().parents[1]
BATCHING_ATOL = 2e-5

# Real-artifact stored-golden checks are heavy (1.4B FP32 model, ~6 GiB) and opt
# in through the environment so the regular test command stays cheap. Run with:
#   WYAT_OURO_SPEED_ARTIFACT=/tmp/ouro-speed/artifact-last2-cpu-fp32 \
#       .venv/bin/python -m pytest tests/test_ouro_inference_speed.py -q
REAL_ARTIFACT_ENV = "WYAT_OURO_SPEED_ARTIFACT"
REAL_ARTIFACT_VOCAB = 49152
LEAD_BASELINE_RECEIPT = Path("/tmp/ouro-speed/baseline-cpu-p400-k6-s16.json")
LEAD_BASELINE_INPUT_SHA256 = (
    "61aa0f6dea97598fbfaf355a12876100aba4ec237712f6f78f5ee90a1a20a31b"
)
P832_GOLDEN = (
    REPO_ROOT / "artifacts" / "ouro-speed-20261021" / "logits-cpu-p832-k10-s64.pt"
)


class _CatContextKV:
    """Reference cache: concatenate context K/V with the candidate suffix."""

    def __init__(
        self,
        bank: tuple[Any, ...],
        step: int,
    ) -> None:
        self.bank = bank
        self.step = step
        self.count = 0

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
        cache_kwargs: object = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.bank[self.count]
        self.count += 1
        return (
            torch.cat((context.key, key), dim=2),
            torch.cat((context.value, value), dim=2),
        )


def _tiny_config_dict(vocabulary: int = 320) -> dict[str, Any]:
    return {
        "architectures": ["OuroForCausalLM"],
        "attention_dropout": 0.0,
        "bos_token_id": 0,
        "eos_token_id": 0,
        "head_dim": 4,
        "hidden_act": "silu",
        "hidden_size": 16,
        "initializer_range": 0.02,
        "intermediate_size": 32,
        "layer_types": ["full_attention"] * ouro_artifact.NATIVE_LAYERS,
        "max_position_embeddings": 65536,
        "max_window_layers": ouro_artifact.NATIVE_LAYERS,
        "model_type": "ouro",
        "num_attention_heads": 4,
        "num_hidden_layers": ouro_artifact.NATIVE_LAYERS,
        "num_key_value_heads": 4,
        "rms_norm_eps": 1e-6,
        "rope_scaling": None,
        "rope_theta": 1000000.0,
        "sliding_window": None,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "total_ut_steps": 4,
        "early_exit_threshold": 1.0,
        "use_cache": True,
        "use_sliding_window": False,
        "vocab_size": vocabulary,
    }


def _tiny_backbone_and_head() -> tuple[Any, ScalarScoreHead]:
    config_cls, model_cls = load_ouro_classes(SOURCE_FIXTURE)
    config = config_cls(**_tiny_config_dict())
    config._attn_implementation = "sdpa"
    with torch.random.fork_rng():
        torch.manual_seed(11)
        backbone = model_cls(config).eval()
        attach_step_lora(
            backbone,
            rank=ouro_artifact.ADAPTER_RANK,
            alpha=ouro_artifact.ADAPTER_ALPHA,
            steps=ouro_artifact.CANDIDATE_STEPS,
            targets=ouro_artifact.ADAPTER_TARGETS,
        )
        # Non-zero lora_B so both adapted candidate steps have an effect.
        for name, param in backbone.named_parameters():
            if ".lora_B." in name:
                param.data.normal_(0.0, 0.05)
        head = ScalarScoreHead(config.hidden_size, ouro_artifact.HEAD_WIDTH).eval()
        head.norm.weight.data.normal_(1.0, 0.1)
        head.projection.weight.data.normal_(0.0, 0.1)
        head.output.weight.data.normal_(0.0, 0.1)
    backbone.requires_grad_(False)
    head.requires_grad_(False)
    return backbone, head


def _bare_backbone() -> Any:
    config_cls, model_cls = load_ouro_classes(SOURCE_FIXTURE)
    config = config_cls(**_tiny_config_dict())
    config._attn_implementation = "sdpa"
    with torch.random.fork_rng():
        torch.manual_seed(11)
        backbone = model_cls(config).eval()
    backbone.requires_grad_(False)
    return backbone


def _reference_last_hidden(
    backbone: Any,
    context_cache: Any,
    candidates: list[torch.Tensor],
    candidate_steps: int,
) -> torch.Tensor:
    """Former per-candidate path: fresh mask/rotary per step and a cat cache."""
    native = ouro._native_backbone(backbone)
    router = backbone._step_lora_router
    rows = []
    for candidate_ids in candidates:
        hidden = native.embed_tokens(candidate_ids)
        for step in range(candidate_steps):
            bank = context_cache.banks[step if context_cache.policy == "matched" else 0]
            positions = torch.arange(
                context_cache.length,
                context_cache.length + candidate_ids.shape[1],
                device=candidate_ids.device,
            )
            rotary = native.rotary_emb(hidden, positions[None])
            mask = ouro._causal_mask(hidden, context_cache.length)
            reference_cache = _CatContextKV(bank, step)
            with router.activate(step):
                for layer in native.layers:
                    hidden = layer(
                        hidden,
                        attention_mask=mask,
                        position_ids=positions[None],
                        past_key_value=reference_cache,
                        use_cache=True,
                        cache_position=positions,
                        position_embeddings=rotary,
                        current_ut=step,
                    )
            hidden = native.norm(hidden)
        rows.append(hidden[:, -1])
    return torch.cat(rows, dim=0)


def _tokenizer(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    alphabet = ByteLevel.alphabet()
    vocabulary = {symbol: index for index, symbol in enumerate(sorted(alphabet))}
    vocabulary["[PAD]"] = len(vocabulary)
    tokenizer = Tokenizer(BPE(vocabulary, [], unk_token=None))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    wrapped = PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]
        tokenizer_object=tokenizer, pad_token="[PAD]"
    )
    wrapped.save_pretrained(path)
    special_map = path / "special_tokens_map.json"
    if not special_map.is_file():
        special_map.write_text(json.dumps({"pad_token": "[PAD]"}) + "\n")
    return path


def _build_tiny_artifact(tmp_path: Path) -> Path:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    for name in ("configuration_ouro.py", "modeling_ouro.py"):
        shutil.copy2(SOURCE_FIXTURE / name, source_dir / name)
    config_dict = _tiny_config_dict()
    (source_dir / "config.json").write_text(json.dumps(config_dict, indent=2) + "\n")
    tokenizer_dir = _tokenizer(tmp_path / "tokenizer")

    config_cls, model_cls = load_ouro_classes(source_dir)
    config = config_cls(**config_dict)
    config._attn_implementation = "sdpa"
    with torch.random.fork_rng():
        torch.manual_seed(11)
        backbone = model_cls(config).to(dtype=torch.bfloat16).eval()
        base_weights = {
            key: value.detach().clone() for key, value in backbone.state_dict().items()
        }
        attach_step_lora(
            backbone,
            rank=ouro_artifact.ADAPTER_RANK,
            alpha=ouro_artifact.ADAPTER_ALPHA,
            steps=ouro_artifact.CANDIDATE_STEPS,
            targets=ouro_artifact.ADAPTER_TARGETS,
        )
        for name, param in backbone.named_parameters():
            if ".lora_B." in name:
                param.data.normal_(0.0, 0.05)
        adapters = {
            f"backbone.{name}": param.detach().float().clone()
            for name, param in backbone.named_parameters()
            if ".lora_" in name
        }
        head = ScalarScoreHead(config.hidden_size, ouro_artifact.HEAD_WIDTH)
        head_state = {
            f"head.{key}": value.detach().float().clone()
            for key, value in head.state_dict().items()
        }

    artifact = tmp_path / "artifact"
    ouro.write_ouro_artifact(
        artifact,
        source_dir=source_dir,
        tokenizer_dir=tokenizer_dir,
        base_weights=base_weights,
        adapters=adapters,
        head=head_state,
        max_length=2048,
        attention="sdpa-math",
        device_policy="cpu-fp32",
        contract_prompt=ouro_artifact.PROMPT,
    )
    return artifact


@pytest.fixture(scope="module")
def tiny_model() -> tuple[Any, ScalarScoreHead]:
    return _tiny_backbone_and_head()


@pytest.fixture(scope="module")
def tiny_artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _build_tiny_artifact(tmp_path_factory.mktemp("ouro-speed"))


def _candidates(lengths: list[int], seed: int = 7) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [
        torch.randint(0, 320, (1, length), generator=generator) for length in lengths
    ]


def test_exact_workspace_is_bitwise_identical_to_cat_reference(
    tiny_model: tuple[Any, ScalarScoreHead],
) -> None:
    backbone, head = tiny_model
    context = torch.randint(0, 320, (1, 23))
    candidates = _candidates([5, 7, 5, 3, 9, 5])
    cache = prefill_context(backbone, context, policy="last")
    reference = _reference_last_hidden(backbone, cache, candidates, 2)
    tested = torch.cat(
        [
            run_candidate_steps(backbone, cache, candidate, candidate_steps=2)[:, -1]
            for candidate in candidates
        ],
        dim=0,
    )
    assert torch.equal(reference, tested)
    expected_scores = head(reference)[:, 0]
    scores = score_candidates_from_cache(backbone, head, cache, candidates)
    assert torch.equal(expected_scores, scores)


def test_exact_workspace_matches_cat_reference_with_matched_policy(
    tiny_model: tuple[Any, ScalarScoreHead],
) -> None:
    backbone, head = tiny_model
    context = torch.randint(0, 320, (1, 19))
    candidates = _candidates([4, 6, 4], seed=13)
    cache = prefill_context(
        backbone, context, policy="matched", context_steps=ouro_artifact.CONTEXT_STEPS
    )
    assert len(cache.banks) == ouro_artifact.CONTEXT_STEPS
    reference = _reference_last_hidden(backbone, cache, candidates, 2)
    tested = score_candidates_from_cache(backbone, head, cache, candidates)
    assert torch.equal(head(reference)[:, 0], tested)


def test_scoring_leaves_context_banks_unchanged_and_repeats(
    tiny_model: tuple[Any, ScalarScoreHead],
) -> None:
    backbone, head = tiny_model
    cache = prefill_context(backbone, torch.randint(0, 320, (1, 17)), policy="last")
    bank = cache.banks[0]
    before = [(entry.key.clone(), entry.value.clone()) for entry in bank]
    candidates = _candidates([6, 4, 6])
    first = score_candidates_from_cache(backbone, head, cache, candidates)
    second = score_candidates_from_cache(backbone, head, cache, candidates)
    assert torch.equal(first, second)
    for (key, value), entry in zip(before, bank, strict=True):
        assert torch.equal(key, entry.key)
        assert torch.equal(value, entry.value)


def test_candidate_batching_matches_default_within_tolerance(
    tiny_model: tuple[Any, ScalarScoreHead],
) -> None:
    backbone, head = tiny_model
    cache = prefill_context(backbone, torch.randint(0, 320, (1, 21)), policy="last")
    candidates = _candidates([5, 3, 5, 9, 3, 5, 7])
    expected = score_candidates_from_cache(backbone, head, cache, candidates)
    batched = score_candidates_from_cache(
        backbone, head, cache, candidates, candidate_batching=True
    )
    delta = (expected - batched).abs().max().item()
    assert delta <= BATCHING_ATOL
    assert torch.equal(expected.argsort(), batched.argsort())


def test_candidate_batching_single_and_equal_length_groups(
    tiny_model: tuple[Any, ScalarScoreHead],
) -> None:
    backbone, head = tiny_model
    cache = prefill_context(backbone, torch.randint(0, 320, (1, 12)), policy="last")
    for candidates in (
        _candidates([4], seed=3),
        _candidates([6, 6, 6, 6], seed=5),
    ):
        expected = score_candidates_from_cache(backbone, head, cache, candidates)
        batched = score_candidates_from_cache(
            backbone, head, cache, candidates, candidate_batching=True
        )
        assert (expected - batched).abs().max().item() <= BATCHING_ATOL


def test_score_candidates_forwards_candidate_batching(
    tiny_model: tuple[Any, ScalarScoreHead],
) -> None:
    backbone, head = tiny_model
    context = torch.randint(0, 320, (1, 15))
    candidates = _candidates([5, 8, 5])
    expected = score_candidates(backbone, head, context, candidates)
    batched = score_candidates(
        backbone, head, context, candidates, candidate_batching=True
    )
    assert (expected - batched).abs().max().item() <= BATCHING_ATOL


def test_candidate_batching_rejects_non_bool(
    tiny_model: tuple[Any, ScalarScoreHead],
) -> None:
    backbone, head = tiny_model
    cache = prefill_context(backbone, torch.randint(0, 320, (1, 8)), policy="last")
    with pytest.raises(ValueError, match="candidate_batching"):
        score_candidates_from_cache(
            backbone,
            head,
            cache,
            _candidates([4]),
            candidate_batching=1,  # type: ignore[arg-type]
        )


def test_candidate_kv_workspace_bytes_matches_buffer_shape(
    tiny_model: tuple[Any, ScalarScoreHead],
) -> None:
    backbone, _head = tiny_model
    native = ouro._native_backbone(backbone)
    expected = (
        2
        * len(native.layers)
        * native.config.num_key_value_heads
        * (20 + 6)
        * int(native.config.head_dim)
        * 4
    )
    assert candidate_kv_workspace_bytes(backbone, 20, 6) == expected
    assert candidate_kv_workspace_bytes(backbone, 20, 6, batch=3) == 3 * expected
    with pytest.raises(ValueError, match="candidate_length"):
        candidate_kv_workspace_bytes(backbone, 20, 0)
    with pytest.raises(ValueError, match="context_length"):
        candidate_kv_workspace_bytes(backbone, -1, 6)


def test_streaming_loader_matches_checkpoint_values_exactly(
    tiny_artifact: Path,
) -> None:
    scorer = load_ouro(tiny_artifact, device="cpu")
    checkpoint = load_file(str(tiny_artifact / ouro_artifact.BASE_WEIGHT_FILE))
    parameters = {
        key.replace(".base.weight", ".weight"): value
        for key, value in scorer.backbone.named_parameters()
        if ".lora_" not in key
    }
    assert set(checkpoint) == set(parameters)
    for key, tensor in checkpoint.items():
        assert torch.equal(parameters[key].detach(), tensor.to(torch.float32)), key
    expected_inv_freq, _ = ouro._default_rope_init(scorer.backbone.config)
    assert torch.equal(
        scorer.backbone.model.rotary_emb.inv_freq.detach(), expected_inv_freq
    )


def test_streaming_loader_refuses_nonfinite_base_tensor(
    tiny_artifact: Path, tmp_path: Path
) -> None:
    backbone = _bare_backbone()
    base = load_file(str(tiny_artifact / ouro_artifact.BASE_WEIGHT_FILE))
    assert set(base) == set(backbone.state_dict())
    corrupt = dict(base)
    first_key = next(iter(corrupt))
    corrupt[first_key] = corrupt[first_key].clone()
    corrupt[first_key].view(-1)[0] = float("nan")
    corrupt_path = tmp_path / "corrupt.safetensors"
    save_file(corrupt, str(corrupt_path))
    with pytest.raises(ValueError, match="nonfinite Ouro base tensor"):
        ouro._stream_base_weights(backbone, corrupt_path)


def test_streaming_loader_refuses_unknown_base_tensor(
    tiny_artifact: Path, tmp_path: Path
) -> None:
    backbone = _bare_backbone()
    base = load_file(str(tiny_artifact / ouro_artifact.BASE_WEIGHT_FILE))
    unknown = dict(base)
    unknown["model.unexpected.weight"] = base["model.norm.weight"].clone()
    unknown_path = tmp_path / "unknown.safetensors"
    save_file(unknown, str(unknown_path))
    with pytest.raises(ValueError, match="do not match native architecture"):
        ouro._stream_base_weights(backbone, unknown_path)


def test_scorer_candidate_batching_flag_matches_default(
    tiny_artifact: Path,
) -> None:
    record = {
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
        },
    }
    # Shared-context batching is the default; "per_candidate" is the exact path.
    exact = OuroScorer.from_directory(
        tiny_artifact, device="cpu", candidate_batching="per_candidate"
    )
    batched = OuroScorer.from_directory(tiny_artifact, device="cpu")
    assert batched.candidate_batching == "shared_context"
    assert exact.candidate_batching == "per_candidate"
    for exact_logits, batched_logits in zip(
        exact.logits(record), batched.logits(record), strict=True
    ):
        assert (
            max(
                abs(left - right)
                for left, right in zip(exact_logits, batched_logits, strict=True)
            )
            <= BATCHING_ATOL
        )


def test_context_cache_is_owned_by_the_loaded_scorer(tiny_artifact: Path) -> None:
    scorer = load_ouro(tiny_artifact, device="cpu")
    context = torch.randint(0, 320, (1, 10))
    cache = prefill_context(scorer.backbone, context, policy="last")
    assert cache.owner is scorer.backbone
    rows = _candidates([4, 6])
    scores = score_candidates_from_cache(scorer.backbone, scorer.head, cache, rows)
    assert scores.shape == (2,)
    assert torch.isfinite(scores).all()


def _real_synthetic_inputs(
    seed: int, context_tokens: int, candidates: int, candidate_length: int
) -> tuple[torch.Tensor, list[torch.Tensor], str]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    context = torch.randint(
        0,
        REAL_ARTIFACT_VOCAB,
        (1, context_tokens),
        generator=generator,
        dtype=torch.long,
    )
    candidate_tensors = [
        torch.randint(
            0,
            REAL_ARTIFACT_VOCAB,
            (1, candidate_length),
            generator=generator,
            dtype=torch.long,
        )
        for _ in range(candidates)
    ]
    legacy_sha = hashlib.sha256(
        torch.cat([context, *candidate_tensors], dim=1).numpy().tobytes()
    ).hexdigest()
    return context, candidate_tensors, legacy_sha


@pytest.fixture(scope="module")
def real_scorer() -> Any:
    artifact = os.environ.get(REAL_ARTIFACT_ENV)
    if not artifact:
        pytest.skip(f"set {REAL_ARTIFACT_ENV} to run the real 1.4B artifact checks")
    root = Path(artifact)
    if not (root.is_dir() and (root / ouro_artifact.MANIFEST).is_file()):
        pytest.skip(f"missing Ouro artifact at {root}")
    return load_ouro(root, device="cpu")


def _real_logits(
    scorer: Any,
    context: torch.Tensor,
    candidates: list[torch.Tensor],
    *,
    candidate_batching: bool = False,
) -> torch.Tensor:
    context = context.to(scorer.device)
    tensors = [candidate.to(scorer.device) for candidate in candidates]
    with torch.inference_mode():
        cache = prefill_context(scorer.backbone, context, policy="last")
        return (
            score_candidates_from_cache(
                scorer.backbone,
                scorer.head,
                cache,
                tensors,
                candidate_steps=ouro_artifact.CANDIDATE_STEPS,
                candidate_batching=candidate_batching,
            )
            .detach()
            .cpu()
        )


def test_real_artifact_default_path_matches_lead_stored_baseline(
    real_scorer: Any,
) -> None:
    if not LEAD_BASELINE_RECEIPT.is_file():
        pytest.skip(f"missing stored baseline receipt {LEAD_BASELINE_RECEIPT}")
    baseline = json.loads(LEAD_BASELINE_RECEIPT.read_text())
    context, candidates, legacy_sha = _real_synthetic_inputs(20261021, 400, 6, 16)
    assert legacy_sha == LEAD_BASELINE_INPUT_SHA256
    golden = torch.tensor(baseline["logits"], dtype=torch.float32)
    logits = _real_logits(real_scorer, context, candidates)
    assert (logits - golden).abs().max().item() <= BATCHING_ATOL
    assert torch.equal(logits.argsort(), golden.argsort())


def test_real_artifact_candidate_batching_matches_lead_stored_baseline(
    real_scorer: Any,
) -> None:
    if not LEAD_BASELINE_RECEIPT.is_file():
        pytest.skip(f"missing stored baseline receipt {LEAD_BASELINE_RECEIPT}")
    baseline = json.loads(LEAD_BASELINE_RECEIPT.read_text())
    context, candidates, legacy_sha = _real_synthetic_inputs(20261021, 400, 6, 16)
    assert legacy_sha == LEAD_BASELINE_INPUT_SHA256
    golden = torch.tensor(baseline["logits"], dtype=torch.float32)
    logits = _real_logits(real_scorer, context, candidates, candidate_batching=True)
    assert (logits - golden).abs().max().item() <= BATCHING_ATOL
    assert torch.equal(logits.argsort(), golden.argsort())


def test_real_artifact_k10_p832_matches_stored_golden(real_scorer: Any) -> None:
    if not P832_GOLDEN.is_file():
        pytest.skip(f"missing stored P832/K10 golden {P832_GOLDEN}")
    payload = torch.load(P832_GOLDEN, weights_only=False)
    golden = torch.tensor(payload["logits"], dtype=torch.float32)
    context, candidates, _legacy_sha = _real_synthetic_inputs(20261021, 832, 10, 64)
    exact = _real_logits(real_scorer, context, candidates)
    batched = _real_logits(real_scorer, context, candidates, candidate_batching=True)
    assert (exact - golden).abs().max().item() <= BATCHING_ATOL
    assert torch.equal(exact.argsort(), golden.argsort())
    assert (batched - golden).abs().max().item() <= BATCHING_ATOL
    assert torch.equal(batched.argsort(), golden.argsort())
