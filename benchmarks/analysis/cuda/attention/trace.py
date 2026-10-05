"""Diagnostic dispatch metadata only; never attach to ordinary inference."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Protocol, cast

import torch
from torch._C._autograd import DeviceType
from torch.profiler import ProfilerActivity, profile
from torch.utils._python_dispatch import TorchDispatchMode

from benchmarks.analysis.cuda.manifest import MODEL_ID
from wyat import Model
from wyat.benchmark import BenchmarkCycle, BenchmarkEngine, _run
from wyat.core import Scorer
from wyat.evaluation import EvaluationCase


class Operator(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> object: ...


def tensor_metadata(value: torch.Tensor) -> dict[str, object]:
    return {
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "dtype": str(value.dtype),
        "device": str(value.device),
    }


def argument(
    args: tuple[object, ...],
    kwargs: dict[str, object],
    position: int,
    name: str,
    default: object = None,
) -> object:
    return args[position] if len(args) > position else kwargs.get(name, default)


def call_flags(
    args: tuple[object, ...], kwargs: dict[str, object]
) -> dict[str, object]:
    """Pinned public SDPA schema; keep no tensor references in diagnostics."""
    values = {
        "dropout_p": argument(args, kwargs, 4, "dropout_p", 0.0),
        "is_causal": argument(args, kwargs, 5, "is_causal", False),
        "scale": kwargs.get("scale"),
        "enable_gqa": kwargs.get("enable_gqa", False),
    }
    return {
        name: value
        if value is None or isinstance(value, (bool, float, int))
        else f"unsupported scalar type: {type(value).__name__}"
        for name, value in values.items()
    }


# Upstream TorchDispatchMode.__init_subclass__ lacks annotations.
class Dispatch(TorchDispatchMode):  # type: ignore[no-untyped-call]
    """Keep primitives only; do not extend any attention tensor lifetime."""

    def __init__(self) -> None:
        super().__init__()  # type: ignore[no-untyped-call]  # upstream initializer
        self.events: list[dict[str, object]] = []

    def __torch_dispatch__(
        self,
        func: object,
        types: tuple[type[object], ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        name = str(func)
        options = {} if kwargs is None else kwargs
        if "scaled_dot_product" in name:
            inputs = [
                argument(args, options, index, field)
                for index, field in enumerate(("query", "key", "value", "attn_mask"))
            ]
            tensors = [value for value in inputs if isinstance(value, torch.Tensor)]
            event: dict[str, object] = {
                "operator": name,
                "tensors": [tensor_metadata(value) for value in tensors],
            }
            if name == "aten.scaled_dot_product_attention.default":
                flags = call_flags(args, options)
                event.update(
                    call_flags=flags,
                    selector_boundary="public_before_bias_conversion",
                    selector_scale_scope=(
                        "SDPAParams has no scale field; explicit scales are not queried"
                    ),
                    converted_bias_strides="unknown",
                )
                dropout = flags["dropout_p"]
                causal, gqa = flags["is_causal"], flags["enable_gqa"]
                if (
                    len(tensors) == 4
                    and type(dropout) in (float, int)
                    and isinstance(causal, bool)
                    and isinstance(gqa, bool)
                    and flags["scale"] is None
                ):
                    try:
                        params = torch.backends.cuda.SDPAParams(
                            tensors[0],
                            tensors[1],
                            tensors[2],
                            tensors[3],
                            float(cast(float, dropout)),
                            causal,
                            gqa,
                        )
                        event["efficient_eligible"] = (
                            torch.backends.cuda.can_use_efficient_attention(
                                params, debug=True
                            )
                        )
                    except (RuntimeError, ValueError) as error:
                        event["selector_error"] = str(error)
                else:
                    event["selector_error"] = (
                        "SDPAParams cannot represent this call's scalar arguments "
                        "or explicit scale"
                    )
            self.events.append(event)
        # Operator overloads are an untyped upstream variadic callable boundary.
        return cast(Operator, func)(*args, **options)


def profile_batch(
    scorer: Scorer,
    cases: tuple[EvaluationCase, ...],
    *,
    name: str,
) -> tuple[BenchmarkCycle, dict[str, object]]:
    observer = Dispatch()
    with (
        warnings.catch_warnings(record=True) as caught,
        profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
        ) as profiler,
        observer,
    ):
        cycle = _run(
            BenchmarkEngine(
                name, Model(scorer, model_id=MODEL_ID), batch_size=len(cases)
            ),
            cases,
        )
        torch.cuda.synchronize()
    events = profiler.key_averages(group_by_input_shape=True)
    kernels = [
        {
            "name": event.name,
            "duration_us": event.time_range.elapsed_us(),
            "device_index": event.device_index,
        }
        for event in profiler.events()
        if event.device_type == DeviceType.CUDA
    ]
    return cycle, {
        "case_ids": [case.id for case in cases],
        "dispatch": observer.events,
        "operators": [
            {
                "name": event.key,
                "count": event.count,
                "shapes": event.input_shapes,
                "self_cpu_time_us": event.self_cpu_time_total,
                "self_device_time_us": event.self_device_time_total,
            }
            for event in events
        ],
        "warnings": [str(item.message) for item in caught],
        "cuda_kernel_events": kernels,
        "cuda_kernel_names_available": bool(kernels),
        "timing_scope": "Intrusive profiler/dispatch diagnostics; no speed claim",
    }


def efficient_dispatch_gate(
    evidence: dict[str, object], *, batch_size: int, length: int
) -> bool:
    events = cast(Sequence[dict[str, object]], evidence.get("dispatch", ()))
    if len(events) != 28:
        return False
    expected = [batch_size, 16, length, 128]
    for event in events:
        flags = cast(dict[str, object], event.get("call_flags", {}))
        if (
            event.get("operator") != "aten.scaled_dot_product_attention.default"
            or event.get("selector_boundary") != "public_before_bias_conversion"
            or set(flags) != {"dropout_p", "is_causal", "enable_gqa", "scale"}
            or type(flags.get("dropout_p")) not in (float, int)
            or flags.get("dropout_p") != 0
            or flags.get("is_causal") is not False
            or flags.get("enable_gqa") is not False
            or (
                flags.get("scale") is not None
                and (
                    type(flags.get("scale")) is not float
                    or flags.get("scale") != 128**-0.5
                )
            )
        ):
            return False
        tensors = cast(list[dict[str, object]], event.get("tensors", []))
        if len(tensors) != 4:
            return False
        if any(
            value.get("shape") != expected
            or value.get("dtype") != "torch.float32"
            or not value.get("stride")
            or cast(list[int], value["stride"])[-1] != 1
            or value.get("device") != "cuda:0"
            for value in tensors[:3]
        ):
            return False
        # This is the actual public mask, preceding C++ bias conversion/padding.
        if (
            tensors[3].get("shape") != [1, 1, length, length]
            or tensors[3].get("dtype") != "torch.bool"
            or tensors[3].get("device") != "cuda:0"
            or tensors[3].get("stride") != [length * length, length * length, length, 1]
        ):
            return False
    operators = cast(Sequence[dict[str, object]], evidence.get("operators", ()))
    counts: dict[str, int] = {}
    for operator in operators:
        name, count = operator.get("name"), operator.get("count")
        if not isinstance(name, str) or type(count) is not int or count < 0:
            return False
        counts[name] = counts.get(name, 0) + count
    if (
        counts.get("aten::_scaled_dot_product_efficient_attention") != 28
        or counts.get("aten::_efficient_attention_forward") != 28
        or any(
            count > 0
            and any(
                fallback in name
                for fallback in ("attention_math", "flash_attention", "cudnn_attention")
            )
            for name, count in counts.items()
        )
    ):
        return False
    kernels = cast(Sequence[dict[str, object]], evidence.get("cuda_kernel_events", ()))
    attention_names = [
        name
        for event in kernels
        if isinstance(name := event.get("name"), str) and "fmha" in name.lower()
    ]
    # This exact runtime/device observed the SM80-specialized FP32 CUTLASS path.
    # Other implementations remain unsupported by this diagnostic declaration.
    return (
        evidence.get("cuda_kernel_names_available") is True
        and len(attention_names) == 28
        and all(
            name.startswith("fmha_cutlassF_f32_aligned_") and "_sm80(" in name
            for name in attention_names
        )
    )
