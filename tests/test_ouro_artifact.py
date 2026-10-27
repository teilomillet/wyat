# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Ouro decision artifacts: clean nomenclature, inference, and tampering refusal."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
from tokenizers import Tokenizer  # noqa: E402
from tokenizers.decoders import ByteLevel as ByteLevelDecoder  # noqa: E402
from tokenizers.models import BPE  # noqa: E402
from tokenizers.pre_tokenizers import ByteLevel  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

import wyat  # noqa: E402
from wyat.loading import ouro, ouro_artifact  # noqa: E402
from wyat.loading.ouro import (  # noqa: E402
    CandidateBatch,
    ContextKVCache,
    LayerKVCache,
    OuroScorer,
    ScalarScoreHead,
    StepLoRALinear,
    StepLoRARouter,
    attach_step_lora,
    load_ouro_classes,
    prefill_context,
    run_candidate_steps,
    score_candidates,
    write_ouro_artifact,
)

SOURCE_FIXTURE = Path(__file__).parent / "fixtures" / "ouro_source"

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


def _build_ouro_artifact(
    tmp_path: Path,
    *,
    seed: int = 11,
    device_policy: str = "cuda-bf16-autocast",
    contract_prompt: str = ouro_artifact.PROMPT,
    cache_entries: int = 4,
    context_prefill: dict[str, Any] | None = None,
    numerics: str | None = None,
) -> Path:
    src_dir = tmp_path / "source"
    src_dir.mkdir()
    shutil.copy2(
        SOURCE_FIXTURE / "configuration_ouro.py", src_dir / "configuration_ouro.py"
    )
    shutil.copy2(SOURCE_FIXTURE / "modeling_ouro.py", src_dir / "modeling_ouro.py")
    cfg_dict = _tiny_config_dict(320)
    (src_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2) + "\n")
    tok_dir = _tokenizer(tmp_path / "tokenizer")

    config_cls, model_cls = load_ouro_classes(src_dir)
    config = config_cls(**cfg_dict)
    config._attn_implementation = "sdpa"
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        backbone = model_cls(config).to(dtype=torch.bfloat16).eval()
        base_weights = {k: v.detach().clone() for k, v in backbone.state_dict().items()}
        attach_step_lora(
            backbone,
            rank=ouro_artifact.ADAPTER_RANK,
            alpha=ouro_artifact.ADAPTER_ALPHA,
            steps=ouro_artifact.CANDIDATE_STEPS,
            targets=ouro_artifact.ADAPTER_TARGETS,
        )
        # Initialize non-zero lora_B weights so both step adapters have an effect.
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
            f"head.{k}": v.detach().float().clone()
            for k, v in head.state_dict().items()
        }

    artifact_dir = tmp_path / "artifact"
    write_ouro_artifact(
        artifact_dir,
        contract_prompt=contract_prompt,
        source_dir=src_dir,
        tokenizer_dir=tok_dir,
        base_weights=base_weights,
        adapters=adapters,
        head=head_state,
        max_length=2048,
        attention="sdpa-math",
        device_policy=device_policy,
        cache_entries=cache_entries,
        cache_state_block_tokens=128 if cache_entries else 0,
        context_prefill=context_prefill,
        numerics=numerics,
    )
    return artifact_dir


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _build_ouro_artifact(tmp_path_factory.mktemp("ouro"))


def test_write_ouro_artifact_refuses_quoted_contract_before_mkdir(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError) as raised:
        _build_ouro_artifact(tmp_path, contract_prompt="quoted-menu-isolated-option-v1")
    assert str(raised.value) == (
        "write_ouro_artifact only writes the frozen current-prompt layout; "
        "quoted/preview contracts must use a v2 artifact writer"
    )
    assert not (tmp_path / "artifact").exists()


def test_mps_fp32_device_policy_artifact_is_accepted(tmp_path: Path) -> None:
    artifact_dir = _build_ouro_artifact(tmp_path, device_policy="mps-fp32")
    manifest = ouro_artifact.validate_artifact(artifact_dir)
    assert manifest["execution"]["device_policy"] == "mps-fp32"
    assert "mps-fp32" in ouro_artifact.DEVICE_POLICIES


def test_unknown_device_policy_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported device policy"):
        _build_ouro_artifact(tmp_path, device_policy="tpu-fp32")


def test_device_policy_mismatch_requires_explicit_override(tmp_path: Path) -> None:
    artifact_dir = _build_ouro_artifact(tmp_path, device_policy="cpu-fp32")
    with pytest.raises(ValueError, match="does not permit 'mps'"):
        OuroScorer.from_directory(artifact_dir, device="mps")


def test_cuda_artifact_keeps_its_cpu_fp32_fallback(tmp_path: Path) -> None:
    artifact_dir = _build_ouro_artifact(tmp_path)
    scorer = OuroScorer.from_directory(artifact_dir, device="cpu")
    assert scorer.device.type == "cpu"
    assert scorer.device_policy == "cuda-bf16-autocast"
    assert next(scorer.backbone.parameters()).dtype == torch.float32


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS is unavailable"
)
def test_public_loader_runs_a_cuda_artifact_on_apple_mps(tmp_path: Path) -> None:
    artifact_dir = _build_ouro_artifact(tmp_path)
    model = wyat.load(artifact_dir, backend="torch", device="mps")
    scorer = model._scorer
    assert isinstance(scorer, OuroScorer)
    assert scorer.device.type == "mps"
    assert scorer.device_policy == "cuda-bf16-autocast"
    assert next(scorer.backbone.parameters()).dtype == torch.float32
    assert set(model.predict(RECORD)) == set(RECORD["questions"])


def test_mps_artifact_selects_mps_by_default_and_needs_override_for_cpu(
    tmp_path: Path,
) -> None:
    artifact_dir = _build_ouro_artifact(tmp_path, device_policy="mps-fp32")
    if torch.backends.mps.is_available():
        scorer = OuroScorer.from_directory(artifact_dir)
        assert scorer.device.type == "mps"
        assert scorer.device_policy == "mps-fp32"
        assert len(scorer.logits(RECORD)) == 3
    else:
        with pytest.raises(ValueError, match="MPS is unavailable"):
            OuroScorer.from_directory(artifact_dir)
    with pytest.raises(ValueError, match="does not permit 'cpu'"):
        OuroScorer.from_directory(artifact_dir, device="cpu")
    overridden = OuroScorer.from_directory(
        artifact_dir, device="cpu", allow_device_policy_override=True
    )
    assert overridden.device.type == "cpu"


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS is unavailable"
)
def test_mps_override_matches_cpu_within_declared_tolerance(tmp_path: Path) -> None:
    artifact_dir = _build_ouro_artifact(tmp_path)
    cpu_scorer = OuroScorer.from_directory(artifact_dir, device="cpu")
    mps_scorer = OuroScorer.from_directory(
        artifact_dir, device="mps", allow_device_policy_override=True
    )
    row = cpu_scorer.encode_record(RECORD)[0]
    reference = torch.tensor(cpu_scorer.question_logits(row))
    observed = torch.tensor(mps_scorer.question_logits(row))
    torch.testing.assert_close(
        observed,
        reference,
        atol=ouro_artifact.MPS_EXECUTION_TOLERANCE_ATOL,
        rtol=ouro_artifact.MPS_EXECUTION_TOLERANCE_RTOL,
    )


def test_clean_nomenclature_symbols_and_forbidden_legacy_names() -> None:
    assert ouro_artifact.TOTAL_STEPS == 4
    assert ouro_artifact.CONTEXT_STEPS == 4
    assert ouro_artifact.CONTEXT_KV_STEP == 4
    assert ouro_artifact.CANDIDATE_STEPS == 2
    assert len(ouro_artifact.adapter_tensor_names()) == 384
    for legacy in (
        "OuroForkModel",
        "OptionForkModel",
        "SegmentedQuestion",
        "PrefixState",
        "LayerKV",
        "ScalarHead",
        "BranchAdapterController",
        "BranchAdapterLinear",
        "attach_branch_adapters",
        "branch_hidden",
        "branch_logits",
        "prefill",
    ):
        assert not hasattr(ouro, legacy), legacy
        assert not hasattr(ouro_artifact, legacy), legacy


def test_candidate_batch_encoding(
    artifact: Path,
) -> None:
    scorer = OuroScorer.from_directory(artifact, device="cpu")
    rows = scorer.encode_record(RECORD)
    assert len(rows) == 3
    for row in rows:
        assert isinstance(row, CandidateBatch)
        assert not hasattr(row, "prefix_ids")
        assert not hasattr(row, "option_ids")
        assert len(row.candidate_ids) == len(row.question.options)


def test_scorer_matches_direct_score_candidates(artifact: Path) -> None:
    scorer = OuroScorer.from_directory(artifact, device="cpu")
    for row in scorer.encode_record(RECORD):
        scored = torch.tensor(scorer.question_logits(row), dtype=torch.float32)
        context_tensor = torch.tensor([row.context_ids], dtype=torch.long)
        ctx_cache = prefill_context(
            scorer.backbone,
            context_tensor,
            policy="last",
            context_steps=ouro_artifact.CONTEXT_STEPS,
        )
        assert isinstance(ctx_cache, ContextKVCache)
        assert len(ctx_cache.banks) == 1
        assert isinstance(ctx_cache.banks[0][0], LayerKVCache)
        first_cand = run_candidate_steps(
            scorer.backbone,
            ctx_cache,
            torch.tensor([row.candidate_ids[0]], dtype=torch.long),
            candidate_steps=ouro_artifact.CANDIDATE_STEPS,
        )
        assert first_cand.shape == (1, len(row.candidate_ids[0]), 16)
        direct = score_candidates(
            scorer.backbone,
            scorer.head,
            context_tensor,
            [torch.tensor([ids], dtype=torch.long) for ids in row.candidate_ids],
            policy="last",
            context_steps=ouro_artifact.CONTEXT_STEPS,
            candidate_steps=ouro_artifact.CANDIDATE_STEPS,
        )
        torch.testing.assert_close(scored, direct, atol=1e-7, rtol=1e-5)


def test_step_lora_router_activates_distinct_steps(artifact: Path) -> None:
    scorer = OuroScorer.from_directory(artifact, device="cpu")
    router = scorer.backbone._step_lora_router
    assert isinstance(router, StepLoRARouter)
    assert router.active_step is None
    layer0_q = scorer.backbone.model.layers[0].self_attn.q_proj
    assert isinstance(layer0_q, StepLoRALinear)

    x = torch.randn(1, 3, 16)
    out_disabled = layer0_q(x)
    with router.activate(0):
        assert router.active_step == 0
        out_step0 = layer0_q(x)
        with router.disabled():
            assert router.active_step is None
            out_nested_disabled = layer0_q(x)
    with router.activate(1):
        assert router.active_step == 1
        out_step1 = layer0_q(x)

    torch.testing.assert_close(out_disabled, out_nested_disabled, atol=0.0, rtol=0.0)
    assert not torch.allclose(out_disabled, out_step0)
    assert not torch.allclose(out_step0, out_step1)


def _step_lora_reference(layer: StepLoRALinear, x: Any, step: int) -> Any:
    base = layer.base(x)
    with torch.autocast(device_type=x.device.type, enabled=False):
        update = torch.nn.functional.linear(
            torch.nn.functional.linear(x.float(), layer.lora_A[step]),
            layer.lora_B[step],
        )
    return base + (update * layer.scaling).to(base.dtype)


def test_step_lora_without_autocast_enters_no_guard(
    artifact: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scorer = OuroScorer.from_directory(artifact, device="cpu")
    layer = scorer.backbone.model.layers[0].self_attn.q_proj
    x = torch.randn(1, 5, 16)
    with scorer.backbone._step_lora_router.activate(1):
        expected = _step_lora_reference(layer, x, 1)
        guards: list[bool] = []
        autocast = torch.autocast

        def counted(*args: Any, **kwargs: Any) -> Any:
            guards.append(True)
            return autocast(*args, **kwargs)

        monkeypatch.setattr(torch, "autocast", counted)
        actual = layer(x)
    assert guards == []
    assert torch.equal(actual, expected)


def test_step_lora_keeps_fp32_factors_under_autocast(artifact: Path) -> None:
    scorer = OuroScorer.from_directory(artifact, device="cpu")
    layer = scorer.backbone.model.layers[0].self_attn.q_proj
    x = torch.randn(1, 5, 16)
    with (
        scorer.backbone._step_lora_router.activate(0),
        torch.autocast("cpu", dtype=torch.bfloat16),
    ):
        actual = layer(x)
        expected = _step_lora_reference(layer, x, 0)
    assert actual.dtype == torch.bfloat16
    assert torch.equal(actual, expected)


def test_scoring_validates_every_candidate_before_any_pass(
    artifact: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Checking between passes would make an async device drain each pass first.
    scorer = OuroScorer.from_directory(artifact, device="cpu")
    vocabulary = scorer.backbone.config.vocab_size
    cache = prefill_context(scorer.backbone, torch.randint(0, vocabulary, (1, 11)))
    passes: list[int] = []
    run_pass = ouro._candidate_hidden_steps

    def counted(*args: Any, **kwargs: Any) -> Any:
        passes.append(1)
        return run_pass(*args, **kwargs)

    monkeypatch.setattr(ouro, "_candidate_hidden_steps", counted)
    candidates = [
        torch.randint(0, vocabulary, (1, 4)),
        torch.randint(0, vocabulary, (1, 6)),
        torch.full((1, 3), vocabulary, dtype=torch.long),
    ]
    with pytest.raises(ValueError, match="outside the vocabulary"):
        ouro.score_candidates_from_cache(
            scorer.backbone, scorer.head, cache, candidates
        )
    assert passes == []


def test_option_permutation_only_remaps_logits(artifact: Path) -> None:
    scorer = OuroScorer.from_directory(artifact, device="cpu")
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
        atol=1e-6,
        rtol=1e-5,
    )


def test_context_prefix_cache_reuses_state_across_questions(artifact: Path) -> None:
    uncached = OuroScorer.from_directory(artifact, device="cpu", prefix_cache_size=0)
    cached = OuroScorer.from_directory(artifact, device="cpu", prefix_cache_size=4)

    ref_logits = uncached.logits(RECORD)
    hit_logits = cached.logits(RECORD)
    assert cached.prefix_cache_currsize == 1
    assert cached.prefix_cache_misses == 1
    assert cached.prefix_cache_hits == 2

    for ref, got in zip(ref_logits, hit_logits, strict=True):
        torch.testing.assert_close(
            torch.tensor(got),
            torch.tensor(ref),
            atol=ouro_artifact.CACHE_TOLERANCE_ATOL,
            rtol=ouro_artifact.CACHE_TOLERANCE_RTOL,
        )

    cached.clear_prefix_cache()
    assert cached.prefix_cache_currsize == 0
    assert cached.prefix_cache_hits == 0
    assert cached.prefix_cache_misses == 0


def test_public_load_predicts_through_the_model_interface(artifact: Path) -> None:
    model = wyat.load(artifact, device="cpu")
    assert model.model_id.startswith("ouro-")
    answers = OuroScorer.from_directory(artifact, device="cpu").predict(RECORD)
    assert set(answers) == {"choice", "score", "noul"}


def test_overlong_records_raise_model_input_error_without_forward_pass(
    artifact: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scorer = OuroScorer.from_directory(artifact, device="cpu", max_length=64)
    passes: list[object] = []
    orig = scorer.question_logits

    def counted(row: CandidateBatch) -> list[float]:
        passes.append(row)
        return orig(row)

    monkeypatch.setattr(scorer, "question_logits", counted)
    options = {"a": "short", "b": "x"}
    fits = [wyat.Request.choice("Which?", options, context=t) for t in ("u", "v")]
    refused = wyat.Request.choice("Which?", options, context="y" * 200)
    with pytest.raises(wyat.ModelInputError) as raised:
        scorer.score_many([*fits, refused])
    assert raised.value.reason == "input-length"
    assert passes == []


@pytest.mark.parametrize(
    "tamper",
    ["weights", "extra-file", "contract", "duplicate-key", "subdir"],
)
def test_tampered_artifacts_are_refused(tmp_path: Path, tamper: str) -> None:
    root = _build_ouro_artifact(tmp_path)
    manifest_path = root / ouro_artifact.MANIFEST
    manifest = json.loads(manifest_path.read_text())
    if tamper == "weights":
        (root / "head.safetensors").write_bytes(
            (root / "head.safetensors").read_bytes() + b"\0"
        )
    elif tamper == "extra-file":
        (root / "notes.txt").write_text("unexpected")
    elif tamper == "contract":
        manifest["contract"]["branch_loops"] = 4
        manifest_path.write_text(json.dumps(manifest))
    elif tamper == "duplicate-key":
        text = manifest_path.read_text()
        manifest_path.write_text(
            text.replace(
                '"format": "wyat-ouro-decision-v1",',
                '"format": "wyat-ouro-decision-v1", "format": "wyat-ouro-decision-v1",',
                1,
            )
        )
    elif tamper == "subdir":
        (root / "extra_dir").mkdir()
    with pytest.raises(ValueError):
        ouro_artifact.validate_artifact(root)


# Shared-context candidate batching: the scorer's default for inference.


def test_shared_context_is_the_default_and_keeps_every_ranking(artifact: Path) -> None:
    shared = OuroScorer.from_directory(artifact, device="cpu")
    exact = OuroScorer.from_directory(
        artifact, device="cpu", candidate_batching="per_candidate"
    )
    assert ouro.CANDIDATE_BATCHING == "shared_context"
    assert shared.candidate_batching == "shared_context"
    assert exact.candidate_batching == "per_candidate"
    for row in shared.encode_record(RECORD):
        batched = torch.tensor(shared.question_logits(row))
        reference = torch.tensor(exact.question_logits(row))
        torch.testing.assert_close(batched, reference, atol=1e-6, rtol=0)
        assert torch.equal(batched.argsort(), reference.argsort())


@pytest.mark.parametrize("policy", ["last", "matched"])
@pytest.mark.parametrize("budget", [1, 4096, ouro.SHARED_SCORE_BUDGET_BYTES])
def test_shared_context_scores_each_candidate_in_its_place(
    artifact: Path, policy: ouro.CachePolicy, budget: int
) -> None:
    scorer = OuroScorer.from_directory(artifact, device="cpu")
    vocabulary = scorer.backbone.config.vocab_size
    generator = torch.Generator().manual_seed(9)
    context = torch.randint(0, vocabulary, (1, 13), generator=generator)
    candidates = [
        torch.randint(0, vocabulary, (1, n), generator=generator)
        for n in (5, 2, 7, 2, 4)
    ]
    with torch.inference_mode():
        cache = prefill_context(scorer.backbone, context, policy=policy)
        reference = ouro.score_candidates_from_cache(
            scorer.backbone, scorer.head, cache, candidates
        )
        batched = ouro.score_candidates_shared_context(
            scorer.backbone, scorer.head, cache, candidates, score_budget_bytes=budget
        )
    torch.testing.assert_close(batched, reference, atol=1e-6, rtol=0)
    assert scorer.backbone.model.config._attn_implementation in {"eager", "sdpa"}


def test_shared_context_batches_follow_the_budget_shortest_first() -> None:
    lengths = [5, 2, 7, 2, 4]
    assert ouro.shared_context_batches(lengths, 30, 4, budget_bytes=10**9) == [
        [1, 3, 4, 0, 2]
    ]
    # 8 · 4 heads · packed · (30 + packed) bytes: [1, 3] fits, a third row does not.
    assert ouro.shared_context_batches(lengths, 30, 4, budget_bytes=12_800) == [
        [1, 3],
        [4, 0],
        [2],
    ]
    assert ouro.shared_context_batches(lengths, 30, 4, budget_bytes=1) == [
        [1],
        [3],
        [4],
        [0],
        [2],
    ]


def test_shared_context_validates_every_candidate_before_any_pass(
    artifact: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scorer = OuroScorer.from_directory(artifact, device="cpu")
    vocabulary = scorer.backbone.config.vocab_size
    cache = prefill_context(scorer.backbone, torch.randint(0, vocabulary, (1, 11)))
    passes: list[int] = []
    run_step = ouro._run_recurrent_step

    def counted(*args: Any, **kwargs: Any) -> Any:
        passes.append(1)
        return run_step(*args, **kwargs)

    monkeypatch.setattr(ouro, "_run_recurrent_step", counted)
    candidates = [
        torch.randint(0, vocabulary, (1, 4)),
        torch.full((1, 3), vocabulary, dtype=torch.long),
    ]
    with pytest.raises(ValueError, match="outside the vocabulary"):
        ouro.score_candidates_shared_context(
            scorer.backbone, scorer.head, cache, candidates
        )
    assert passes == []


def test_candidate_batching_takes_two_modes_and_the_earlier_flag(
    artifact: Path,
) -> None:
    on = OuroScorer.from_directory(artifact, device="cpu", candidate_batching=True)
    off = OuroScorer.from_directory(artifact, device="cpu", candidate_batching=False)
    assert on.candidate_batching == "shared_context"
    assert off.candidate_batching == "per_candidate"
    for invalid in ("all", 1, None):
        with pytest.raises(ValueError, match="candidate_batching"):
            OuroScorer.from_directory(
                artifact,
                device="cpu",
                candidate_batching=invalid,  # type: ignore[arg-type]
            )


# v2: the context execution a model was trained and judged with.
CONTEXT_PREFILL = {
    "chunk": 64,
    "single_chunk_max_tokens": 128,
    "chunked_attention": "lean",
    "host_cache_above_tokens": 1024,
}
NUMERICS = "deterministic-no-tf32-v1"


@pytest.fixture
def restore_numerics() -> Any:
    """Undo the process-wide numerics a v2 artifact applies when it loads."""
    saved = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.get_float32_matmul_precision(),
        torch.are_deterministic_algorithms_enabled(),
        torch.utils.deterministic.fill_uninitialized_memory,
    )
    yield
    torch.backends.cuda.matmul.allow_tf32 = saved[0]
    torch.backends.cudnn.allow_tf32 = saved[1]
    torch.set_float32_matmul_precision(saved[2])
    torch.use_deterministic_algorithms(saved[3])
    torch.utils.deterministic.fill_uninitialized_memory = saved[4]


def _v2_artifact(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    return _build_ouro_artifact(
        tmp_path,
        cache_entries=0,
        context_prefill=CONTEXT_PREFILL,
        numerics=NUMERICS,
    )


def test_v2_manifest_records_the_context_execution(tmp_path: Path) -> None:
    v2 = json.loads((_v2_artifact(tmp_path / "v2") / "ouro_manifest.json").read_text())
    (tmp_path / "v1").mkdir()
    v1 = json.loads(
        (
            _build_ouro_artifact(tmp_path / "v1", cache_entries=0)
            / "ouro_manifest.json"
        ).read_text()
    )
    assert (v1["format"], v2["format"]) == (
        ouro_artifact.FORMAT,
        ouro_artifact.FORMAT_V2,
    )
    assert v2["execution"]["context_prefill"] == CONTEXT_PREFILL
    assert v2["execution"]["numerics"] == NUMERICS
    assert "context_prefill" not in v1["execution"]
    # Same files, different execution: the execution is part of the identity.
    assert v1["files_sha256"] == v2["files_sha256"]
    assert v1["model_id"] != v2["model_id"]
    assert v1["model_id"] == ouro_artifact.model_identity(
        v1["files_sha256"], v1["prompt"]
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("chunk", 0, "must be positive"),
        ("single_chunk_max_tokens", 32, "single exact pass"),
        ("chunked_attention", "native", "chunked context attention"),
        ("host_cache_above_tokens", 0, "positive or null"),
        ("numerics", "fast", "execution numerics"),
        ("attention", "eager", "replaces SDPA attention"),
        ("cache", 4, "prefix reuse"),
    ],
)
def test_v2_manifest_refuses_another_context_execution(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    manifest = json.loads((_v2_artifact(tmp_path) / "ouro_manifest.json").read_text())
    if field in CONTEXT_PREFILL:
        manifest["execution"]["context_prefill"][field] = value
    elif field == "cache":
        manifest["cache"]["entries"] = value
        manifest["cache"]["state_block_tokens"] = 128
    else:
        manifest["execution"][field] = value
    with pytest.raises(ValueError, match=message):
        ouro_artifact.validate_manifest(manifest)
    with pytest.raises(ValueError, match="both context_prefill and numerics"):
        ouro_artifact.build_manifest(
            files_sha256=manifest["files_sha256"],
            encoder_sha256=manifest["prompt"]["encoder_sha256"],
            adapter_dtypes=manifest["tensors"]["adapters"],
            head_dtypes=manifest["tensors"]["head"],
            max_length=2048,
            attention="sdpa-math",
            device_policy="cuda-bf16-autocast",
            context_prefill=CONTEXT_PREFILL,
        )


def test_a_context_of_one_chunk_takes_the_exact_single_pass(artifact: Path) -> None:
    scorer = OuroScorer.from_directory(artifact, device="cpu")
    context = torch.randint(0, scorer.backbone.config.vocab_size, (1, 40))
    with torch.inference_mode():
        single = prefill_context(scorer.backbone, context)
        for chunk in (40, 64):
            same = prefill_context(
                scorer.backbone, context, chunk=chunk, attention="lean"
            )
            for a, b in zip(single.banks[0], same.banks[0], strict=True):
                assert torch.equal(a.key, b.key) and torch.equal(a.value, b.value)


def test_lean_chunked_prefill_keeps_the_single_pass_values(artifact: Path) -> None:
    scorer = OuroScorer.from_directory(artifact, device="cpu")
    context = torch.randint(0, scorer.backbone.config.vocab_size, (1, 90))
    with torch.inference_mode():
        single = prefill_context(scorer.backbone, context)
        chunked = prefill_context(
            scorer.backbone, context, chunk=16, attention="lean", bank_device="cpu"
        )
    assert chunked.length == single.length == 90
    for a, b in zip(single.banks[0], chunked.banks[0], strict=True):
        torch.testing.assert_close(b.key, a.key, atol=2e-5, rtol=1e-4)
        torch.testing.assert_close(b.value, a.value, atol=2e-5, rtol=1e-4)
    with pytest.raises(ValueError, match="last policy"):
        prefill_context(scorer.backbone, context, policy="matched", chunk=16)


def test_v2_scorer_chunks_only_contexts_longer_than_one_exact_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restore_numerics: Any
) -> None:
    v2 = OuroScorer.from_directory(_v2_artifact(tmp_path / "v2"), device="cpu")
    (tmp_path / "v1").mkdir()
    v1 = OuroScorer.from_directory(
        _build_ouro_artifact(tmp_path / "v1", cache_entries=0), device="cpu"
    )
    assert v2.context_prefill == CONTEXT_PREFILL and v2.numerics == NUMERICS
    assert (v1.context_prefill, v1.numerics) == (None, None)
    assert v2._context_execution(128) == (None, "native", None)
    # On CPU the context cache stays where it is computed.
    assert v2._context_execution(2000) == (64, "lean", None)
    chunked: list[int] = []
    original = ouro._chunked_prefill_context

    def spy(*args: Any, **kwargs: Any) -> Any:
        chunked.append(kwargs["chunk"])
        return original(*args, **kwargs)

    monkeypatch.setattr(ouro, "_chunked_prefill_context", spy)
    for row in v2.encode_record(RECORD):
        expected = v1.question_logits(row)
        torch.testing.assert_close(
            torch.tensor(v2.question_logits(row)),
            torch.tensor(expected),
            atol=2e-5,
            rtol=1e-4,
        )
    assert chunked and set(chunked) == {64}
    assert (
        v1.encode_record(RECORD)[0].context_ids
        == v2.encode_record(RECORD)[0].context_ids
    )
    assert len(v2.encode_record(RECORD)[0].context_ids) > 128


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_v2_flash_context_on_mac_matches_cpu_single_pass(
    tmp_path: Path, device: str, restore_numerics: Any
) -> None:
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Apple MPS is unavailable")
    source = tmp_path / "v2"
    source.mkdir()
    artifact_dir = _build_ouro_artifact(
        source,
        cache_entries=0,
        context_prefill={**CONTEXT_PREFILL, "chunked_attention": "flash"},
        numerics=NUMERICS,
    )
    reference_source = tmp_path / "reference"
    reference_source.mkdir()
    reference_dir = _build_ouro_artifact(reference_source, cache_entries=0)
    assert (
        ouro_artifact.validate_artifact(artifact_dir)["files_sha256"]
        == ouro_artifact.validate_artifact(reference_dir)["files_sha256"]
    )
    scorer = OuroScorer.from_directory(artifact_dir, device=device)
    reference = OuroScorer.from_directory(reference_dir, device="cpu")
    assert scorer._context_execution(2000) == (64, "flash", None)
    assert scorer.context_prefill is not None
    for row in scorer.encode_record(RECORD):
        assert len(row.context_ids) > scorer.context_prefill["single_chunk_max_tokens"]
        expected = torch.tensor(reference.question_logits(row))
        observed = torch.tensor(scorer.question_logits(row))
        assert torch.isfinite(observed).all()
        torch.testing.assert_close(
            observed,
            expected,
            atol=ouro_artifact.MPS_EXECUTION_TOLERANCE_ATOL,
            rtol=ouro_artifact.MPS_EXECUTION_TOLERANCE_RTOL,
        )
        assert observed.argmax() == expected.argmax()


def test_v2_artifact_applies_its_numerics_and_refuses_prefix_reuse(
    tmp_path: Path, restore_numerics: Any
) -> None:
    artifact_dir = _v2_artifact(tmp_path)
    torch.use_deterministic_algorithms(False)
    OuroScorer.from_directory(artifact_dir, device="cpu")
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.get_float32_matmul_precision() == "highest"
    assert not torch.backends.cuda.matmul.allow_tf32
    with pytest.raises(ValueError, match="prefix reuse"):
        OuroScorer.from_directory(artifact_dir, device="cpu", prefix_cache_size=2)
    with pytest.raises(ValueError, match="unknown execution numerics"):
        ouro.apply_numerics("fast")
