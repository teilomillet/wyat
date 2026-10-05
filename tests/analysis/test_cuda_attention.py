"""Offline causal/ownership and diagnostic lifecycle challenges, no CUDA."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.integrations import sdpa_attention

from benchmarks.analysis.cuda.attention import scorer, study, trace
from benchmarks.analysis.cuda.manifest import DATA, MODEL_ID, ROOT, digest
from benchmarks.analysis.cuda.scorer import FP32Scorer
from benchmarks.analysis.evidence import FrozenSources
from benchmarks.fixtures.kai import create_kai_bundle
from wyat import Candidate, Request
from wyat.backends.kai.encoding import EncodedRequest
from wyat.backends.kai.torch import KaiScorer
from wyat.benchmark import BenchmarkCycle, BenchmarkEngine
from wyat.core import decision_from_logits
from wyat.evaluation import EvaluationCase, EvaluationRow, case_from_dict
from wyat.wire import decode_json


@pytest.fixture
def base(tmp_path: Path) -> KaiScorer:
    return KaiScorer.from_directory(create_kai_bundle(tmp_path / "kai"), max_length=512)


@contextmanager
def cpu_math(backend: SDPBackend) -> Iterator[None]:
    assert backend is SDPBackend.EFFICIENT_ATTENTION
    with sdpa_kernel(SDPBackend.MATH):
        yield


@settings(
    max_examples=6,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(repeats=st.integers(1, 6), count=st.integers(2, 5))
def test_mask_adapter_preserves_full_model_head_and_owner(
    base: KaiScorer, monkeypatch: pytest.MonkeyPatch, repeats: int, count: int
) -> None:
    # Source-behavior oracle: Mac availability predicates otherwise select GQA
    # unlike the frozen Linux CUDA host. This does not establish CUDA eligibility.
    monkeypatch.setattr(sdpa_attention, "_is_torch_mps_available", False)
    monkeypatch.setattr(sdpa_attention, "_is_torch_xpu_available", False)
    monkeypatch.setattr(scorer, "sdpa_kernel", cpu_math)
    original = base.backbone
    config = original.config.to_dict()
    requests = (
        Request(
            "read " * repeats,
            tuple(Candidate(str(i), "write " * (i + 1)) for i in range(count)),
        ),
        Request("other?", (Candidate("a", "read"), Candidate("b", "write"))),
    )
    expected = FP32Scorer(base).score_many(requests)
    candidate = scorer.efficient_scorer(base)
    actual = candidate.score_many(requests)
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            torch.tensor(left), torch.tensor(right), atol=1e-6, rtol=1e-5
        )
    assert base.backbone is original
    assert original.config.to_dict() == config
    view = cast(scorer.ScorerView, candidate.base)
    assert view.backbone.model is original and view.head is base.head
    assert candidate.score_many(()) == []
    # Interleaved calls cannot retain a different request's mask or suffix.
    assert candidate.score_many(requests[:1]) == candidate.score_many(requests[:1])
    candidate.score_many(requests[1:])
    for left, right in zip(
        candidate.score_many(requests[:1]), expected[:1], strict=True
    ):
        torch.testing.assert_close(
            torch.tensor(left), torch.tensor(right), atol=1e-6, rtol=1e-5
        )


def test_future_token_cannot_change_earlier_readout(
    base: KaiScorer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sdpa_attention, "_is_torch_mps_available", False)
    monkeypatch.setattr(sdpa_attention, "_is_torch_xpu_available", False)
    monkeypatch.setattr(scorer, "sdpa_kernel", cpu_math)
    adapter = scorer.MaskedBackbone(base.backbone)
    ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    with torch.inference_mode():
        initial = adapter(input_ids=ids, use_cache=False).last_hidden_state
        ids[0, 4] = 9
        perturbed = adapter(input_ids=ids, use_cache=False).last_hidden_state
    assert initial is not None and perturbed is not None
    torch.testing.assert_close(initial[:, :4], perturbed[:, :4], atol=0, rtol=0)
    assert not torch.equal(initial[:, 4], perturbed[:, 4])


def test_fresh_exact_mask_and_nested_backend_restoration(
    base: KaiScorer, monkeypatch: pytest.MonkeyPatch
) -> None:
    masks: list[torch.Tensor] = []
    contexts: list[SDPBackend] = []

    @contextmanager
    def selected(backend: SDPBackend) -> Iterator[None]:
        contexts.append(backend)
        with sdpa_kernel(SDPBackend.MATH):
            yield

    def inspect(
        module: nn.Module, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> None:
        masks.append(cast(torch.Tensor, kwargs["attention_mask"]))

    monkeypatch.setattr(scorer, "sdpa_kernel", selected)
    adapter = scorer.MaskedBackbone(base.backbone)
    handle = base.backbone.register_forward_pre_hook(inspect, with_kwargs=True)
    try:
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            for _ in range(2):
                adapter(input_ids=torch.tensor([[1, 2, 3], [2, 3, 4]]), use_cache=False)
            assert cast(Callable[[], bool], torch.backends.cuda.math_sdp_enabled)()
            assert not cast(
                Callable[[], bool], torch.backends.cuda.mem_efficient_sdp_enabled
            )()
    finally:
        handle.remove()
    assert contexts == [SDPBackend.EFFICIENT_ATTENTION] * 2
    assert masks[0].data_ptr() != masks[1].data_ptr()
    for mask in masks:
        assert mask.dtype == torch.bool and tuple(mask.shape) == (1, 1, 3, 3)
        assert torch.equal(mask[0, 0], torch.ones((3, 3), dtype=torch.bool).tril())


def test_backend_restores_on_backbone_error(
    base: KaiScorer, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = scorer.MaskedBackbone(base.backbone)
    before = tuple(id(parameter) for parameter in base.backbone.parameters())
    config = base.backbone.config.to_dict()

    def failed(*args: object, **kwargs: object) -> object:
        raise RuntimeError("attention failed")

    monkeypatch.setattr(base.backbone, "forward", failed)
    with (
        sdpa_kernel(SDPBackend.MATH),
        pytest.raises(RuntimeError, match="attention failed"),
    ):
        adapter(input_ids=torch.tensor([[1, 2]]), use_cache=False)
    with sdpa_kernel(SDPBackend.MATH):
        assert cast(Callable[[], bool], torch.backends.cuda.math_sdp_enabled)()
        assert not cast(
            Callable[[], bool], torch.backends.cuda.mem_efficient_sdp_enabled
        )()
    assert tuple(id(parameter) for parameter in base.backbone.parameters()) == before
    assert base.backbone.config.to_dict() == config
    assert not any(module.training for module in base.backbone.modules())


@pytest.mark.parametrize(
    "ids,cache",
    [
        (torch.ones(1, 2), False),
        (torch.tensor([1, 2]), False),
        (torch.empty((1, 0), dtype=torch.long), False),
        (torch.tensor([[1, 2]]), True),
    ],
)
def test_adapter_rejects_out_of_contract_inputs(
    base: KaiScorer, ids: torch.Tensor, cache: bool
) -> None:
    with pytest.raises(ValueError):
        scorer.MaskedBackbone(base.backbone)(input_ids=ids, use_cache=cache)


def dispatch_events(batch: int, length: int) -> list[dict[str, object]]:
    qkv: dict[str, object] = {
        "shape": [batch, 16, length, 128],
        "stride": [length * 2048, length * 128, 128, 1],
        "dtype": "torch.float32",
        "device": "cuda:0",
    }
    mask: dict[str, object] = {
        "shape": [1, 1, length, length],
        "dtype": "torch.bool",
        "stride": [length * length, length * length, length, 1],
        "device": "cuda:0",
    }
    return [
        {
            "operator": "aten.scaled_dot_product_attention.default",
            "selector_boundary": "public_before_bias_conversion",
            "converted_bias_strides": "unknown",
            "efficient_eligible": False,
            "call_flags": {
                "dropout_p": 0.0,
                "is_causal": False,
                "enable_gqa": False,
                "scale": None,
            },
            "tensors": [dict(qkv), dict(qkv), dict(qkv), dict(mask)],
        }
        for _ in range(28)
    ]


def profile_evidence(batch: int, length: int) -> dict[str, object]:
    return {
        "dispatch": dispatch_events(batch, length),
        "operators": [
            {"name": "aten::_scaled_dot_product_efficient_attention", "count": 28},
            {"name": "aten::_efficient_attention_forward", "count": 28},
        ],
        "cuda_kernel_names_available": True,
        "cuda_kernel_events": [
            {
                "name": "fmha_cutlassF_f32_aligned_64x128_rf_sm80(Params)",
                "device_index": 0,
            }
            for _ in range(28)
        ],
    }


@given(
    batch=st.sampled_from((1, 8)),
    length=st.integers(158, 214),
    scale=st.sampled_from((None, 128**-0.5)),
)
def test_dispatch_gate_accepts_exact_unrounded_geometry(
    batch: int, length: int, scale: float | None
) -> None:
    evidence = profile_evidence(batch, length)
    for event in cast(list[dict[str, object]], evidence["dispatch"]):
        cast(dict[str, object], event["call_flags"])["scale"] = scale
    assert trace.efficient_dispatch_gate(evidence, batch_size=batch, length=length)
    events = cast(list[dict[str, object]], evidence["dispatch"])
    # The public Boolean-mask selector can decline before C++ bias conversion;
    # private operator and actual kernel observations establish dispatch instead.
    events[0].pop("efficient_eligible")
    assert trace.efficient_dispatch_gate(evidence, batch_size=batch, length=length)
    evidence["dispatch"] = events[:-1]
    assert not trace.efficient_dispatch_gate(evidence, batch_size=batch, length=length)


@pytest.mark.parametrize(
    "field,value",
    [
        ("dtype", "torch.bfloat16"),
        ("shape", [8, 8, 192, 128]),
        ("stride", [1, 1, 1, 0]),
        ("device", "cpu"),
    ],
)
def test_dispatch_gate_rejects_dtype_head_or_stride_drift(
    field: str, value: object
) -> None:
    evidence = profile_evidence(8, 192)
    events = cast(list[dict[str, object]], evidence["dispatch"])
    cast(list[dict[str, object]], events[0]["tensors"])[0][field] = value
    assert not trace.efficient_dispatch_gate(evidence, batch_size=8, length=192)


@pytest.mark.parametrize(
    "field,value",
    [
        ("shape", [8, 16, 192, 192]),
        ("dtype", "torch.float32"),
        ("stride", [0, 0, 192, 1]),
        ("device", "cpu"),
    ],
)
def test_public_mask_metadata_cannot_be_replaced_by_converted_bias(
    field: str, value: object
) -> None:
    evidence = profile_evidence(8, 192)
    events = cast(list[dict[str, object]], evidence["dispatch"])
    cast(list[dict[str, object]], events[0]["tensors"])[3][field] = value
    assert not trace.efficient_dispatch_gate(evidence, batch_size=8, length=192)


@pytest.mark.parametrize(
    "field,value",
    [
        ("dropout_p", 0.1),
        ("dropout_p", False),
        ("is_causal", True),
        ("is_causal", 0),
        ("enable_gqa", True),
        ("enable_gqa", 0),
        ("scale", 0.125),
        ("scale", True),
        ("scale", 0),
        ("scale", float("nan")),
        ("scale", float("inf")),
    ],
)
def test_dispatch_gate_rejects_changed_or_mistyped_public_flags(
    field: str, value: object
) -> None:
    evidence = profile_evidence(1, 172)
    event = cast(list[dict[str, object]], evidence["dispatch"])[0]
    cast(dict[str, object], event["call_flags"])[field] = value
    assert not trace.efficient_dispatch_gate(evidence, batch_size=1, length=172)


@pytest.mark.parametrize(
    "args,kwargs,expected",
    [
        (
            (),
            {},
            {"dropout_p": 0.0, "is_causal": False, "scale": None, "enable_gqa": False},
        ),
        (
            (None, None, None, None, 0.25, True),
            {},
            {"dropout_p": 0.25, "is_causal": True, "scale": None, "enable_gqa": False},
        ),
        (
            (),
            {"dropout_p": 0.5, "is_causal": True, "scale": 0.125, "enable_gqa": True},
            {"dropout_p": 0.5, "is_causal": True, "scale": 0.125, "enable_gqa": True},
        ),
    ],
)
def test_public_call_flags_resolve_pinned_schema(
    args: tuple[object, ...], kwargs: dict[str, object], expected: dict[str, object]
) -> None:
    assert trace.call_flags(args, kwargs) == expected


@pytest.mark.parametrize("keyword_tensors", [False, True])
def test_observer_selector_uses_effective_public_flags(
    monkeypatch: pytest.MonkeyPatch, keyword_tensors: bool
) -> None:
    captured: list[tuple[object, ...]] = []
    query = torch.zeros((1, 16, 2, 128))
    mask = torch.ones((1, 1, 2, 2), dtype=torch.bool).tril()
    inputs = (query, query, query, mask)

    def params(*args: object) -> tuple[object, ...]:
        captured.append(args)
        return args

    def eligible(params: object, *, debug: bool) -> bool:
        assert debug is True
        return False

    class PublicOperator:
        def __str__(self) -> str:
            return "aten.scaled_dot_product_attention.default"

        def __call__(self, *args: object, **kwargs: object) -> str:
            return "forwarded"

    monkeypatch.setattr(torch.backends.cuda, "SDPAParams", params)
    monkeypatch.setattr(torch.backends.cuda, "can_use_efficient_attention", eligible)
    options: dict[str, object] = {"enable_gqa": True}
    if keyword_tensors:
        options.update(
            query=query,
            key=query,
            value=query,
            attn_mask=mask,
            dropout_p=0.25,
            is_causal=True,
        )
        args: tuple[object, ...] = ()
    else:
        args = (*inputs, 0.25, True)
    observer = trace.Dispatch()
    assert (
        observer.__torch_dispatch__(PublicOperator(), (), args, options) == "forwarded"
    )
    assert len(captured) == 1 and captured[0][:4] == inputs
    assert captured[0][4:] == (0.25, True, True)
    event = observer.events[0]
    assert event["efficient_eligible"] is False
    assert event["call_flags"] == {
        "dropout_p": 0.25,
        "is_causal": True,
        "enable_gqa": True,
        "scale": None,
    }
    assert event["selector_boundary"] == "public_before_bias_conversion"
    assert event["converted_bias_strides"] == "unknown"
    assert len(cast(list[dict[str, object]], event["tensors"])) == 4


@pytest.mark.parametrize("scale", [None, 128**-0.5, 0.125])
def test_default_flags_and_unsupported_scale_are_recorded_without_invention(
    monkeypatch: pytest.MonkeyPatch, scale: float | None
) -> None:
    captured: list[tuple[object, ...]] = []
    query = torch.zeros((1, 16, 2, 128))
    mask = torch.ones((1, 1, 2, 2), dtype=torch.bool).tril()

    def params(*args: object) -> tuple[object, ...]:
        captured.append(args)
        return args

    def eligible(params: object, *, debug: bool) -> bool:
        return False

    class PublicOperator:
        def __str__(self) -> str:
            return "aten.scaled_dot_product_attention.default"

        def __call__(self, *args: object, **kwargs: object) -> str:
            return "forwarded"

    monkeypatch.setattr(torch.backends.cuda, "SDPAParams", params)
    monkeypatch.setattr(torch.backends.cuda, "can_use_efficient_attention", eligible)
    observer = trace.Dispatch()
    assert (
        observer.__torch_dispatch__(
            PublicOperator(), (), (query, query, query, mask), {"scale": scale}
        )
        == "forwarded"
    )
    event = observer.events[0]
    assert event["call_flags"] == {
        "dropout_p": 0.0,
        "is_causal": False,
        "enable_gqa": False,
        "scale": scale,
    }
    if scale is None:
        assert len(captured) == 1 and captured[0][4:] == (0.0, False, False)
        assert event["efficient_eligible"] is False
    else:
        assert captured == [] and "efficient_eligible" not in event
        assert "scale" in cast(str, event["selector_error"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("operator", "aten._scaled_dot_product_efficient_attention.default"),
        ("selector_boundary", "private_after_bias_conversion"),
    ],
)
def test_public_observation_cannot_be_relabelled_as_private(
    field: str, value: str
) -> None:
    evidence = profile_evidence(1, 172)
    cast(list[dict[str, object]], evidence["dispatch"])[0][field] = value
    assert not trace.efficient_dispatch_gate(evidence, batch_size=1, length=172)


@pytest.mark.parametrize(
    "missing",
    ["dispatch", "operators", "cuda_kernel_events", "cuda_kernel_names_available"],
)
def test_dispatch_requires_independent_observation_boundaries(missing: str) -> None:
    evidence = profile_evidence(1, 172)
    evidence.pop(missing)
    assert not trace.efficient_dispatch_gate(evidence, batch_size=1, length=172)


@pytest.mark.parametrize(
    "name",
    [
        "aten::_scaled_dot_product_efficient_attention",
        "aten::_efficient_attention_forward",
    ],
)
@pytest.mark.parametrize("count", [0, 27, 29])
def test_private_operator_count_requires_every_layer(name: str, count: int) -> None:
    evidence = profile_evidence(1, 172)
    operators = cast(list[dict[str, object]], evidence["operators"])
    next(operator for operator in operators if operator["name"] == name)["count"] = (
        count
    )
    assert not trace.efficient_dispatch_gate(evidence, batch_size=1, length=172)


@pytest.mark.parametrize(
    "fallback",
    [
        "aten::_scaled_dot_product_attention_math",
        "aten::_scaled_dot_product_flash_attention",
        "aten::_scaled_dot_product_cudnn_attention",
    ],
)
def test_private_fallback_rejects_otherwise_complete_evidence(fallback: str) -> None:
    evidence = profile_evidence(1, 172)
    cast(list[dict[str, object]], evidence["operators"]).append(
        {"name": fallback, "count": 1}
    )
    assert not trace.efficient_dispatch_gate(evidence, batch_size=1, length=172)


@pytest.mark.parametrize("changed", ["missing", "half", "architecture", "extra_half"])
def test_actual_cuda_kernels_require_exact_float_dispatch(changed: str) -> None:
    evidence = profile_evidence(1, 172)
    kernels = cast(list[dict[str, object]], evidence["cuda_kernel_events"])
    if changed == "missing":
        kernels.pop()
    elif changed == "half":
        kernels[0]["name"] = "fmha_cutlassF_f16_aligned_64x128_rf_sm80(Params)"
    elif changed == "architecture":
        kernels[0]["name"] = "fmha_cutlassF_f32_aligned_64x128_rf_sm90(Params)"
    else:
        kernels.append({"name": "fmha_cutlassF_f16_aligned_64x128_rf_sm80(Params)"})
    assert not trace.efficient_dispatch_gate(evidence, batch_size=1, length=172)


def cycle(cases: tuple[EvaluationCase, ...]) -> BenchmarkCycle:
    rows: list[EvaluationRow] = []
    for case in cases:
        scores = [
            2.0 if candidate.id in case.expected else 0.0
            for candidate in case.request.candidates
        ]
        decision = decision_from_logits(case.request, scores, MODEL_ID)
        rows.append(
            EvaluationRow(case.id, case.expected, "ok", decision, True, 0.0, None)
        )
    return BenchmarkCycle(1.0, tuple(rows), 0, 0)


@dataclass
class Setup:
    output: Path
    path: Path
    calls: list[tuple[str, int]]

    def run(self, batch: int = 8) -> bool:
        return study.study(
            self.path,
            self.output,
            manifest=self.path,
            manifest_sha256="fixed",
            sources=self.path,
            sources_sha256="fixed",
            control=self.path,
            control_sha256="fixed",
            batch_size=batch,
        )


@dataclass
class FakeBase:
    backbone: nn.Module
    head: nn.Module
    device: torch.device

    def encode_request(self, request: Request) -> EncodedRequest:
        return EncodedRequest((1, 2), (0, 1), 0)


class FakeScorer:
    def score_many(self, requests: Sequence[Request]) -> list[list[float]]:
        raise AssertionError("Fake runner returns cycles without executing scorer")


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Setup:
    data = tmp_path / DATA
    data.parent.mkdir(parents=True)
    data.write_bytes((ROOT / DATA).read_bytes())
    full = tuple(
        case_from_dict(decode_json(line)) for line in data.read_text().splitlines()
    )
    calls: list[tuple[str, int]] = []
    payload = Setup(tmp_path / "output.json", tmp_path / "identity", calls)
    for flag in (
        "_is_torch_mps_available",
        "_is_torch_xpu_available",
        "_is_torch_npu_available",
    ):
        monkeypatch.setattr(sdpa_attention, flag, False)
    monkeypatch.setattr(study, "ROOT", tmp_path)

    def fixed(*args: object, **kwargs: object) -> str:
        return "fixed"

    def admission(*args: object, **kwargs: object) -> dict[str, object]:
        return {"fixed": True}

    def hashes(*args: object, **kwargs: object) -> dict[str, str]:
        return {}

    def control(*args: object, **kwargs: object) -> dict[str, str]:
        return {
            "sha256": "fixed",
            "wall_path": str(payload.path),
            "wall_sha256": "fixed",
        }

    def paths() -> tuple[Path, ...]:
        return ()

    def frozen(*args: object, **kwargs: object) -> FrozenSources:
        return FrozenSources((), "fake", "fixed")

    def artifact(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(model_id=MODEL_ID)

    def reference(*args: object, **kwargs: object) -> BenchmarkCycle:
        return cycle(full)

    def cuda_cycle(path: Path, cases: tuple[EvaluationCase, ...]) -> BenchmarkCycle:
        return cycle(cases)

    for name, function in (
        ("digest", fixed),
        ("admit", admission),
        ("admit_sources", hashes),
        ("admit_control", control),
        ("asset_paths", paths),
        ("freeze_sources", frozen),
        ("load_artifact", artifact),
        ("occupancy", admission),
        ("read_reference", reference),
        ("control_cycle", cuda_cycle),
    ):
        monkeypatch.setattr(study, name, function)
    backbone, head = nn.Module(), nn.Module()

    def no_device(module: nn.Module, *args: object, **kwargs: object) -> nn.Module:
        return module

    monkeypatch.setattr(nn.Module, "to", no_device)
    base = FakeBase(backbone, head, torch.device("cpu"))

    def load(*args: object, **kwargs: object) -> FakeBase:
        return base

    def noop(*args: object, **kwargs: object) -> None:
        return None

    def factory(base: object) -> FakeScorer:
        return FakeScorer()

    monkeypatch.setattr(KaiScorer, "from_directory", load)
    monkeypatch.setattr(study, "require_fp32", noop)
    monkeypatch.setattr(study, "FP32Scorer", factory)
    monkeypatch.setattr(study, "efficient_scorer", factory)

    @contextmanager
    def no_precision() -> Iterator[None]:
        yield

    monkeypatch.setattr(study, "ieee_precision", no_precision)

    def forbidden() -> None:
        raise AssertionError("No GPU allocation in fake study")

    def zero() -> int:
        return 0

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", zero)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", zero)

    def ordinary(
        engine: BenchmarkEngine, cases: tuple[EvaluationCase, ...]
    ) -> BenchmarkCycle:
        calls.append((engine.name, len(cases)))
        return cycle(cases)

    def profiled(
        scorer: object, cases: tuple[EvaluationCase, ...], *, name: str
    ) -> tuple[BenchmarkCycle, dict[str, object]]:
        calls.append((f"profile_{name}", len(cases)))
        return cycle(cases), profile_evidence(len(cases), 2)

    monkeypatch.setattr(study, "_run", ordinary)
    monkeypatch.setattr(study, "profile_batch", profiled)
    return payload


@pytest.mark.parametrize("batch", [1, 8])
def test_successful_ordinary_to_profile_transition(setup: Setup, batch: int) -> None:
    assert setup.run(batch)
    expected = [
        (name, batch) for name in ("math", "efficient") for _ in range(8 // batch)
    ]
    assert setup.calls == expected + [
        ("profile_math", batch),
        ("profile_efficient", batch),
    ]
    saved = json.loads(setup.output.read_text())
    assert saved["completed"] and saved["quality_passed"] and saved["dispatch_passed"]
    assert saved["accepted"] is False
    assert all(not ids for ids in saved["unknown_case_ids"].values())
    assert saved["math_drift"]["cpu_reference"]["total"] == 8
    assert saved["efficient_drift"]["cuda_control"]["total"] == 8


def test_missing_private_profile_evidence_keeps_quality_without_dispatch_success(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    def public_only(
        scorer: object, cases: tuple[EvaluationCase, ...], *, name: str
    ) -> tuple[BenchmarkCycle, dict[str, object]]:
        return cycle(cases), {"dispatch": dispatch_events(len(cases), 2)}

    monkeypatch.setattr(study, "profile_batch", public_only)
    assert not setup.run()
    saved = json.loads(setup.output.read_text())
    assert saved["completed"] is True and saved["quality_passed"] is True
    assert saved["dispatch_passed"] is False and saved["accepted"] is False
    assert len(saved["efficient_cycle"]["rows"]) == 8
    assert len(saved["efficient_profile_cycle"]["rows"]) == 8
    assert all(not ids for ids in saved["unknown_case_ids"].values())


@pytest.mark.parametrize("fail_at", [1, 3])
def test_ordinary_backend_failure_retains_rows_and_unstarted_profiles(
    setup: Setup, monkeypatch: pytest.MonkeyPatch, fail_at: int
) -> None:
    completed = 0

    def failed(
        engine: BenchmarkEngine, cases: tuple[EvaluationCase, ...]
    ) -> BenchmarkCycle:
        nonlocal completed
        if engine.name == "math":
            return cycle(cases)
        completed += 1
        if completed != fail_at:
            return cycle(cases)
        row = cases[0]
        return BenchmarkCycle(
            1,
            (
                EvaluationRow(
                    row.id, row.expected, "error", None, False, 0, "unsupported backend"
                ),
            ),
            0,
            0,
        )

    monkeypatch.setattr(study, "_run", failed)
    assert not setup.run(1)
    saved = json.loads(setup.output.read_text())
    assert len(saved["efficient_completed_batches"]) == fail_at
    assert len(saved["unknown_case_ids"]["efficient_ordinary"]) == 8 - fail_at
    assert len(saved["unknown_case_ids"]["math_profile"]) == 1
    assert not any(name.startswith("profile") for name, _ in setup.calls)


def test_final_write_failure_resets_flags_and_keeps_raw_cycles(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = Path.write_text
    seen = False

    def write(path: Path, data: str, *args: object, **kwargs: object) -> int:
        nonlocal seen
        if json.loads(data).get("completed") is True and not seen:
            seen = True
            raise OSError("final write failed")
        return original(path, data)

    monkeypatch.setattr(Path, "write_text", write)
    assert not setup.run()
    saved = json.loads(setup.output.read_text())
    assert seen and saved["completed"] is False and saved["quality_passed"] is False
    assert saved["dispatch_passed"] is False and saved["accepted"] is False
    assert len(saved["efficient_cycle"]["rows"]) == 8
    assert "final write failed" in saved["error"]


@pytest.mark.parametrize("late", ["source", "precision"])
def test_late_failure_never_publishes_success(
    setup: Setup, monkeypatch: pytest.MonkeyPatch, late: str
) -> None:
    if late == "source":

        def changed(path: Path) -> str:
            return (
                "changed"
                if setup.calls and setup.calls[-1][0] == "profile_efficient"
                else "fixed"
            )

        monkeypatch.setattr(study, "digest", changed)
    else:

        @contextmanager
        def broken_restore() -> Iterator[None]:
            yield
            raise RuntimeError("precision restoration failed")

        monkeypatch.setattr(study, "ieee_precision", broken_restore)
    assert not setup.run()
    saved = json.loads(setup.output.read_text())
    assert saved["completed"] is False and saved["quality_passed"] is False
    assert saved["dispatch_passed"] is False and saved["accepted"] is False
    assert len(saved["efficient_profile_cycle"]["rows"]) == 8


def test_numerical_drift_aborts_before_profiles(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    def drifted(
        engine: BenchmarkEngine, cases: tuple[EvaluationCase, ...]
    ) -> BenchmarkCycle:
        actual = cycle(cases)
        if engine.name == "efficient":
            rows = tuple(
                replace(
                    row,
                    prediction=decision_from_logits(
                        case.request,
                        [
                            3.0 if candidate.id in case.expected else 0.0
                            for candidate in case.request.candidates
                        ],
                        MODEL_ID,
                    ),
                )
                for row, case in zip(actual.rows, cases, strict=True)
            )
            actual = replace(actual, rows=rows)
        return actual

    monkeypatch.setattr(study, "_run", drifted)
    assert not setup.run()
    saved = json.loads(setup.output.read_text())
    assert saved["efficient_drift"]["cpu_reference"]["passed"] is False
    assert len(saved["unknown_case_ids"]["efficient_profile"]) == 8
    assert "efficient_profile_cycle" not in saved


def test_missing_cycle_rows_rejected() -> None:
    full = tuple(
        case_from_dict(decode_json(line))
        for line in (ROOT / DATA).read_text().splitlines()
    )[:8]
    expected = cycle(full)
    with pytest.raises(ValueError, match="incomplete"):
        study.check_cycle(cycle(full[:7]), expected, expected)


def test_new_source_declaration_is_exact_and_rechecked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.py"
    source.write_text("frozen\n")
    declaration = tmp_path / "sources.json"
    declaration.write_text(json.dumps({"source.py": digest(source)}))
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(study, "SOURCES", (Path("source.py"),))
    pinned = digest(declaration)
    assert study.admit_sources(declaration, pinned) == {"source.py": digest(source)}
    source.write_text("changed\n")
    with pytest.raises(ValueError, match="bytes changed"):
        study.admit_sources(declaration, pinned)
    with pytest.raises(ValueError, match="reviewed identity"):
        study.admit_sources(declaration, "wrong")


def test_wrong_hf_policy_rejected_before_load(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sdpa_attention, "_is_torch_mps_available", True)
    assert not setup.run()
    saved = json.loads(setup.output.read_text())
    assert "Linux CUDA HF policy" in saved["error"]
    assert setup.calls == []


def test_parent_import_does_not_import_torch() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import benchmarks.analysis.cuda.attention.run; "
            "assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
