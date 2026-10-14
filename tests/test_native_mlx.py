# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Seeded native runtime oracles establish mechanics, not learned quality."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
torch = pytest.importorskip("torch")
from fixtures.wyat import create_wyat_bundle  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from wyat import tasks  # noqa: E402
from wyat.loading.mlx import NativeMLXScorer, load_native_mlx  # noqa: E402
from wyat.loading.native import (  # noqa: E402
    FORMAT,
    MANIFEST,
    NativeModel,
    artifact_contract,
    file_sha256,
    load_native,
)
from wyat.loading.native_artifact import budget_groups  # noqa: E402


def _refresh_manifest(root: Path, **changes: object) -> None:
    path = root / MANIFEST
    manifest: dict[str, Any] = json.loads(path.read_text())
    files = {name: file_sha256(root / name) for name in manifest["files_sha256"]}
    manifest["files_sha256"] = files
    manifest["model_id"] = (
        "native-"
        + hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    )
    manifest.update(changes)
    path.write_text(json.dumps(manifest))


def _create_native_bundle(root: Path, *, bf16: bool = False) -> Path:
    create_wyat_bundle(root)
    (root / "decision_config.json").unlink()
    if bf16:
        path = root / "backbone/model.safetensors"
        state = load_file(str(path))
        state = {
            name: value.bfloat16() if name.endswith("_proj.weight") else value
            for name, value in state.items()
        }
        save_file(state, path)
    with torch.random.fork_rng():
        torch.manual_seed(91)
        adapters = {
            f"layers.0.self_attn.{projection}.{factor}": torch.randn(shape) * 0.06
            for projection, output in (("q_proj", 16), ("v_proj", 8))
            for factor, shape in (("down", (16, 16)), ("up", (output, 16)))
        }
    save_file(adapters, root / "native-adapters.safetensors")
    files = {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in root.rglob("*")
        if path.is_file()
    }
    manifest = {
        "format": FORMAT,
        "model_id": "native-"
        + hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        "contract": artifact_contract(),
        "adapter": {
            "rank": 16,
            "alpha": 32,
            "targets": ["q_proj", "v_proj"],
            "merged": False,
        },
        "max_length": 512,
        "max_batch_tokens": 2048,
        "head_dim": 8,
        "files_sha256": files,
        "provenance": {"fixture": "seeded-mechanics-only"},
        "production_wyat_compatible": False,
        "resume_supported": False,
    }
    (root / MANIFEST).write_text(json.dumps(manifest))
    return root


@pytest.fixture(params=[False, True], ids=["fp32", "mixed-bf16"])
def bundle(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    return _create_native_bundle(tmp_path / "native", bf16=bool(request.param))


@pytest.fixture(params=["cpu", "gpu"])
def scorer(bundle: Path, request: pytest.FixtureRequest) -> NativeMLXScorer:
    device = str(request.param)
    if device == "gpu" and not mx.metal.is_available():
        pytest.skip("Metal GPU is unavailable")
    return load_native_mlx(bundle, device=device, metal=False)


def _record(*, reverse_noul: bool = False) -> dict[str, Any]:
    binary = (
        {"true": "yes", "false": "no"}
        if reverse_noul
        else {"false": "no", "true": "yes"}
    )
    return {
        "state": {"text": "Bonjour 世界 é", "count": 7},
        "questions": {
            "choice": {
                "type": "choice",
                "instructions": "read?",
                "criteria": {"é": "yes", "世界": "no", "a": "write " * 9},
            },
            "score": {"type": "score", "criteria": ["no", "read " * 4, "yes"]},
            "binary": {"type": "noul", "criteria": binary},
        },
    }


@pytest.mark.parametrize("reverse_noul", [False, True])
def test_native_encoding_logits_and_all_typed_readouts_match_torch(
    bundle: Path, scorer: NativeMLXScorer, reverse_noul: bool
) -> None:
    reference = load_native(bundle)
    record = _record(reverse_noul=reverse_noul)
    assert scorer.model_id == reference.model_id
    assert scorer.encode_record(record) == reference.encode_record(record)
    actual_logits = scorer.logits(record)
    expected_logits = reference.logits(record)
    for actual_row, expected_row in zip(actual_logits, expected_logits, strict=True):
        assert actual_row == pytest.approx(expected_row, abs=2e-5, rel=2e-5)
    actual_answers, expected_answers = scorer.predict(record), reference.predict(record)
    assert tuple(actual_answers) == ("choice", "score", "binary")
    for key, answer in actual_answers.items():
        expected = expected_answers[key]
        assert type(answer) is type(expected)
        assert answer.option_ids == expected.option_ids
        assert answer.probabilities == pytest.approx(
            expected.probabilities, abs=2e-5, rel=2e-5
        )
        if isinstance(answer, tasks.ChoiceAnswer):
            assert isinstance(expected, tasks.ChoiceAnswer)
            assert answer.choice == expected.choice
            assert answer.ranking == expected.ranking
        elif isinstance(answer, tasks.ScoreAnswer):
            assert isinstance(expected, tasks.ScoreAnswer)
            assert answer.expected_ordinal_value == pytest.approx(
                expected.expected_ordinal_value, abs=2e-5, rel=2e-5
            )
        else:
            assert isinstance(expected, tasks.NoulAnswer)
            assert answer.probability_true == pytest.approx(
                expected.probability_true, abs=2e-5, rel=2e-5
            )


def test_native_independent_rows_and_right_padding(scorer: NativeMLXScorer) -> None:
    record = _record()
    encoded = scorer.encode_record(record)
    batched = scorer._logits(encoded)
    assert len({len(row.ids) for row in encoded}) > 1
    for row, actual in zip(encoded, batched, strict=True):
        assert scorer._logits([row])[0] == pytest.approx(actual, abs=2e-5, rel=2e-5)
    modified = copy.deepcopy(record)
    modified["questions"]["binary"]["instructions"] = "write " * 75
    assert scorer.encode_record(modified)[0] == encoded[0]
    assert scorer.logits(modified)[0] == pytest.approx(batched[0], abs=2e-5, rel=2e-5)
    assert scorer._logits([]) == []


def test_native_padding_budget_rejected_before_backbone(
    scorer: NativeMLXScorer, monkeypatch: pytest.MonkeyPatch
) -> None:
    encoded = scorer.encode_record(_record())
    scorer.max_batch_tokens = max(len(row.ids) for row in encoded) * len(encoded) - 1

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("backbone executed before budget validation")

    monkeypatch.setattr(type(scorer.backbone), "__call__", forbidden)
    with pytest.raises(ValueError, match="budget"):
        scorer._logits(encoded)


def test_predict_splits_record_over_budget_with_one_group_parity(
    bundle: Path, scorer: NativeMLXScorer
) -> None:
    import wyat

    record = _record()
    reference = load_native(bundle)
    whole_mlx, whole_torch = scorer.logits(record), reference.logits(record)
    expected = scorer.predict(record)
    encoded = scorer.encode_record(record)
    # Below the padded size of the whole record, above the longest row.
    budget = max(len(row.ids) for row in encoded) * (len(encoded) - 1)
    assert len(budget_groups(encoded, budget)) > 1
    for model, whole in ((scorer, whole_mlx), (reference, whole_torch)):
        model.max_batch_tokens = budget
        for actual, row in zip(model.logits(record), whole, strict=True):
            assert actual == pytest.approx(row, abs=2e-5, rel=2e-5)
    answers = scorer.predict(record)
    assert tuple(answers) == tuple(expected)
    for key, answer in answers.items():
        assert answer.probabilities == pytest.approx(
            expected[key].probabilities, abs=2e-5, rel=2e-5
        )
    cooled = wyat.Model(reference, model_id=reference.model_id).predict(
        record, temperature=0.5
    )
    assert tuple(cooled) == tuple(expected)


def test_record_refusals_are_model_input_errors(bundle: Path) -> None:
    from wyat.core import ModelInputError

    for scorer in (
        load_native(bundle),
        load_native_mlx(bundle, device="cpu", metal=False),
    ):
        encoded = scorer.encode_record(_record())
        scorer.max_batch_tokens = min(len(row.ids) for row in encoded) - 1
        with pytest.raises(ModelInputError, match="budget") as budget:
            scorer.predict(_record())
        assert budget.value.reason == "input-length"
        short = (
            load_native(bundle, max_length=16)
            if isinstance(scorer, NativeModel)
            else load_native_mlx(bundle, device="cpu", max_length=16, metal=False)
        )
        with pytest.raises(ModelInputError) as length:
            short.logits(_record())
        assert length.value.reason == "input-length"
        malformed = _record()
        malformed["questions"]["binary"]["type"] = "ranking"
        with pytest.raises(ValueError, match="unknown question type") as schema:
            scorer.predict(malformed)
        assert not isinstance(schema.value, ModelInputError)


def test_json_request_context_encodes_like_record_state(bundle: Path) -> None:
    import wyat

    state = _record()["state"]
    criteria = {"a": "x", "b": "y"}
    question = {"type": "choice", "instructions": "read?", "criteria": criteria}
    request = wyat.Request.choice("read?", criteria, context=state)
    record = {"state": state, "questions": {"decision": question}}
    for scorer in (
        load_native(bundle),
        load_native_mlx(bundle, device="cpu", metal=False),
    ):
        assert scorer.encode_request(request) == scorer.encode_record(record)[0]


def test_predict_many_scores_each_record_in_its_own_groups(
    bundle: Path, scorer: NativeMLXScorer, monkeypatch: pytest.MonkeyPatch
) -> None:
    import wyat

    first, second = _record(), _record(reverse_noul=True)
    second["state"] = {"text": "autre", "count": 2}
    records = [first, second, first]
    rows = [row for item in records for row in scorer.encode_record(item)]
    budget = max(len(row.ids) for row in rows) * 2
    reference = load_native(bundle)
    groups: list[tuple[tuple[int, ...], ...]] = []
    for native in (scorer, reference):
        model = wyat.Model(native, model_id=native.model_id)
        expected = [model.predict(item) for item in records]
        native.max_batch_tokens = budget
        original = native._logits

        def recorded(
            encoded: Any,
            original: Callable[[Any], list[list[float]]] = original,
        ) -> list[list[float]]:
            groups.append(tuple(row.ids[:12] for row in encoded))
            return original(encoded)

        monkeypatch.setattr(native, "_logits", recorded)
        for temperature in (None, 0.5):
            groups.clear()
            batched = model.predict_many(records, temperature=temperature)
            # Three records of three rows, budget of two rows: two groups each.
            assert len(groups) == 6
            assert all(len(set(group)) == 1 for group in groups)
            singles = [model.predict(item, temperature=temperature) for item in records]
            assert [tuple(answers) for answers in batched] == [
                tuple(answers) for answers in expected
            ]
            for got, want in zip(batched, singles, strict=True):
                for key, answer in got.items():
                    assert answer.probabilities == pytest.approx(
                        want[key].probabilities, abs=2e-5, rel=2e-5
                    )


def test_predict_many_refuses_before_scoring_any_record(
    bundle: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import wyat
    from wyat.core import ModelInputError

    native = load_native(bundle)
    calls: list[int] = []
    original = native._logits

    def counted(encoded: Any) -> list[list[float]]:
        calls.append(len(encoded))
        return original(encoded)

    monkeypatch.setattr(native, "_logits", counted)
    model = wyat.Model(native, model_id=native.model_id)
    malformed = _record()
    malformed["questions"]["binary"]["type"] = "ranking"
    with pytest.raises(ValueError, match="unknown question type"):
        model.predict_many([_record(), malformed])
    too_long = _record()
    too_long["questions"]["choice"]["instructions"] = "long " * 600
    with pytest.raises(ModelInputError) as refusal:
        model.predict_many([_record(), too_long])
    assert refusal.value.reason == "input-length"
    assert calls == []


def test_decide_many_return_errors_resolves_budget_and_length_refusals(
    bundle: Path,
) -> None:
    import wyat
    from wyat.core import ModelInputError

    native = load_native(bundle)
    model = wyat.Model(native, model_id=native.model_id)
    requests = [
        wyat.Request.choice(f"read {index}?", {"a": "yes", "b": "no"})
        for index in range(4)
    ]
    expected = model.decide_many(requests)
    longest = max(len(native.encode_request(request).ids) for request in requests)
    native.max_batch_tokens = longest * 2
    with pytest.raises(ModelInputError, match="budget"):
        model.decide_many(requests)
    too_long = wyat.Request.noul("long " * 600)
    results = model.decide_many(
        [*requests[:2], too_long, *requests[2:]], errors="return"
    )
    refused = results[2]
    assert isinstance(refused, ModelInputError) and refused.reason == "input-length"
    decided = [result for result in results if not isinstance(result, ModelInputError)]
    for got, want in zip(decided, expected, strict=True):
        assert isinstance(got, wyat.Decision)
        assert got.ranking == want.ranking
        assert got.scores == pytest.approx(want.scores, abs=2e-5, rel=2e-5)


def test_native_context_limit_never_truncates(bundle: Path) -> None:
    scorer = load_native_mlx(bundle, device="cpu", max_length=16, metal=False)
    with pytest.raises(ValueError):
        scorer.encode_record(_record())
    for maximum in (0, -1, 513, True):
        with pytest.raises(ValueError):
            load_native_mlx(bundle, device="cpu", max_length=maximum, metal=False)


@pytest.mark.parametrize(
    "change",
    [
        {"attention_bias": True},
        {"attention_dropout": 0.1},
        {"hidden_act": "gelu"},
        {"use_sliding_window": True, "sliding_window": 32},
        {
            "rope_parameters": {
                "rope_type": "linear",
                "rope_theta": 1000000.0,
                "factor": 2,
            }
        },
        {"rope_theta": 10000.0},
    ],
)
def test_unsupported_native_configs_fail_closed(
    bundle: Path, change: dict[str, object]
) -> None:
    path = bundle / "backbone/config.json"
    config = json.loads(path.read_text())
    config.update(change)
    path.write_text(json.dumps(config))
    _refresh_manifest(bundle)
    with pytest.raises(ValueError):
        load_native_mlx(bundle, device="cpu", metal=False)


@pytest.mark.parametrize(
    ("filename", "change"),
    [
        ("native-adapters.safetensors", "missing"),
        ("native-adapters.safetensors", "shape"),
        ("native-adapters.safetensors", "dtype"),
        ("native-adapters.safetensors", "nonfinite"),
        ("backbone/model.safetensors", "missing"),
        ("backbone/model.safetensors", "dtype"),
        ("backbone/model.safetensors", "nonfinite"),
        ("decision_head.safetensors", "missing"),
        ("decision_head.safetensors", "dtype"),
        ("decision_head.safetensors", "nonfinite"),
    ],
)
def test_malformed_native_tensors_fail_closed(
    bundle: Path, filename: str, change: str
) -> None:
    path = bundle / filename
    state = load_file(str(path))
    name = next(iter(state))
    transforms: dict[str, Callable[[Any], Any]] = {
        "shape": lambda value: value.reshape(-1),
        "dtype": lambda value: value.half(),
        "nonfinite": lambda value: torch.full_like(value, float("nan")),
    }
    if change == "missing":
        del state[name]
    else:
        state[name] = transforms[change](state[name])
    save_file(state, path)
    _refresh_manifest(bundle)
    with pytest.raises(ValueError):
        load_native_mlx(bundle, device="cpu", metal=False)


def test_native_contract_mismatch_is_not_bypassed(bundle: Path) -> None:
    _refresh_manifest(bundle, contract={"recipe": "incompatible"})
    with pytest.raises(ValueError, match="contract differs"):
        load_native_mlx(bundle, device="cpu", metal=False)


def test_native_changed_weight_identity_is_not_bypassed(bundle: Path) -> None:
    path = bundle / "native-adapters.safetensors"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="identity differs"):
        load_native_mlx(bundle, device="cpu", metal=False)


def test_native_prefix_kv_cache_parity_hits_and_lru_eviction(bundle: Path) -> None:
    torch_uncached = load_native(bundle, prefix_cache_size=0)
    torch_cached = load_native(bundle, prefix_cache_size=2)
    mlx_uncached = load_native_mlx(
        bundle, device="cpu", metal=False, prefix_cache_size=0
    )
    mlx_cached = load_native_mlx(bundle, device="cpu", metal=False, prefix_cache_size=2)

    record_a = _record()
    record_b = copy.deepcopy(record_a)
    record_b["questions"] = {
        "followup": {
            "type": "noul",
            "instructions": "read?",
            "criteria": {"false": "no", "true": "yes"},
        }
    }

    # Uncached vs cached multi-question record parity
    for uncached, cached in (
        (torch_uncached, torch_cached),
        (mlx_uncached, mlx_cached),
    ):
        logits_u = uncached.logits(record_a)
        logits_c = cached.logits(record_a)
        for row_u, row_c in zip(logits_u, logits_c, strict=True):
            assert row_c == pytest.approx(row_u, abs=2e-5, rel=2e-5)
        assert uncached.prefix_cache_currsize == 0
        assert uncached.prefix_cache_hits == 0
        assert uncached.prefix_cache_misses == 0
        assert cached.prefix_cache_currsize == 1
        assert cached.prefix_cache_misses == 1
        assert cached.prefix_cache_hits == 0

        # Follow-up question sharing the same state reuses cached state prefix
        follow_u = uncached.logits(record_b)
        follow_c = cached.logits(record_b)
        assert follow_c[0] == pytest.approx(follow_u[0], abs=2e-5, rel=2e-5)
        assert cached.prefix_cache_hits == 1

        # Repeating the exact same single-question record hits the longer prefix
        repeat_c = cached.logits(record_b)
        assert repeat_c[0] == pytest.approx(follow_u[0], abs=2e-5, rel=2e-5)
        assert cached.prefix_cache_hits == 2

        # LRU eviction with prefix_cache_size=2
        for idx in range(3):
            distinct = {
                "state": f"distinct state {idx}",
                "questions": {
                    "q": {
                        "type": "choice",
                        "instructions": f"pick {idx}",
                        "criteria": {"a": "first", "b": "second"},
                    }
                },
            }
            assert cached.logits(distinct)[0] == pytest.approx(
                uncached.logits(distinct)[0], abs=2e-5, rel=2e-5
            )
            assert cached.prefix_cache_currsize <= 2

        cached.clear_prefix_cache()
        assert cached.prefix_cache_currsize == 0
        assert cached.prefix_cache_hits == 0
        assert cached.prefix_cache_misses == 0


def test_wyat_load_native_decide_and_predict_across_choice_noul_and_score(
    bundle: Path,
) -> None:
    import wyat

    model_torch = wyat.load(bundle, backend="torch", prefix_cache_size=8)
    model_mlx = wyat.load(bundle, backend="mlx", device="cpu", prefix_cache_size=8)

    requests = (
        wyat.Request.choice(
            "read?",
            {"a": "yes", "b": "no"},
            context="Bonjour 世界 é",
        ),
        wyat.Request.noul(
            "missing parameter?",
            context="Bonjour 世界 é",
        ),
        wyat.Request.score(
            "rate completeness",
            ("no", "read", "yes"),
            context="Bonjour 世界 é",
        ),
    )

    decisions_torch = model_torch.decide_many(requests)
    decisions_mlx = model_mlx.decide_many(requests)
    assert len(decisions_torch) == 3
    for dec_t, dec_m in zip(decisions_torch, decisions_mlx, strict=True):
        assert dec_t.selected == dec_m.selected
        assert dec_t.scores == pytest.approx(dec_m.scores, abs=2e-5, rel=2e-5)
        assert dec_t.probabilities == pytest.approx(
            dec_m.probabilities, abs=2e-5, rel=2e-5
        )
    assert 0.0 <= decisions_torch[1].probability_true <= 1.0
    assert 0.0 <= decisions_torch[2].expected_ordinal_value <= 2.0

    # Multi-question PublicRecord via Model.predict and Model.logits
    record = _record()
    answers_t = model_torch.predict(record)
    answers_m = model_mlx.predict(record)
    assert set(answers_t) == {"choice", "score", "binary"}
    for key in answers_t:
        assert answers_t[key].probabilities == pytest.approx(
            answers_m[key].probabilities, abs=2e-5, rel=2e-5
        )
    cooled_t = model_torch.predict(record, temperature=0.5)
    assert set(cooled_t) == {"choice", "score", "binary"}
