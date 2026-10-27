# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Stage exact FP32 Ouro weights from a pinned BF16 checkpoint one module at a time.

Only tensor descriptions survive validation. Staging owns temporary parameter
and buffer slots, and restores the caller's original objects on every exit.
This is an inference lifetime boundary, not a quantized or trainable model.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from safetensors import safe_open
from torch import nn

from wyat.resources import ResourceGuard, checkpoint


@dataclass(frozen=True, slots=True)
class _TensorSpec:
    shape: tuple[int, ...]
    elements: int


@dataclass(frozen=True, slots=True)
class _Slot:
    owner: nn.Module
    name: str
    full_name: str
    value: torch.Tensor
    parameter: bool


def _source_name(name: str) -> str:
    # StepLoRALinear owns the original projection as ``base``.
    return name.removeprefix("base.").replace(".base.", ".")


def _signature(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _slots(module: nn.Module, prefix: str) -> list[_Slot]:
    result: list[_Slot] = []
    for owner_name, owner in module.named_modules():
        stem = ".".join(part for part in (prefix, owner_name) if part)
        for parameter, entries in ((True, owner._parameters), (False, owner._buffers)):
            for name, value in entries.items():
                if value is not None:
                    full_name = ".".join(part for part in (stem, name) if part)
                    result.append(_Slot(owner, name, full_name, value, parameter))
    return result


class StreamedWeights:
    """Validate source weights, then materialize only a module's working set.

    ``expected_state`` is the unadapted backbone's state dictionary. It may
    contain meta tensors: validation keeps shapes and counts, never those
    tensors. Non-meta FP32 adapter parameters added later are staged from their
    existing values and restored unchanged. CPU and Apple MPS execute FP32 math.
    """

    def __init__(
        self,
        path: str | Path,
        expected_state: Mapping[str, torch.Tensor],
        device: torch.device,
        guard: ResourceGuard,
    ) -> None:
        self.path = Path(path).resolve()
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "mps"}:
            raise ValueError("streamed Ouro weights support CPU and MPS only")
        self.guard = guard
        self._specs = {
            name: _TensorSpec(tuple(value.shape), value.numel())
            for name, value in expected_state.items()
        }
        before = _signature(self.path)
        with (
            guard.activate(),
            safe_open(str(self.path), framework="pt", device="cpu") as handle,
        ):
            keys = set(handle.keys())
            missing = set(self._specs) - keys
            if missing - {"lm_head.weight"} or keys - set(self._specs):
                raise ValueError("Ouro base weights do not match native architecture")
            for key in handle.keys():
                checkpoint()
                spec = self._specs[key]
                if tuple(handle.get_slice(key).get_shape()) != spec.shape:
                    raise ValueError(f"Ouro base tensor {key} has unexpected shape")
                # Bound mapped source pages and finite-check intermediates.
                guard.check(
                    6 * spec.elements, operation="validate streamed Ouro weights"
                )
                source = handle.get_tensor(key)
                if source.dtype != torch.bfloat16:
                    raise ValueError(
                        f"Ouro base tensor {key} has dtype {source.dtype}, "
                        "expected torch.bfloat16"
                    )
                if not bool(torch.isfinite(source).all()):
                    raise ValueError(f"nonfinite Ouro base tensor: {key}")
                del source
        if _signature(self.path) != before:
            raise ValueError("Ouro base weights changed during validation")
        self._signature = before
        self._keys = keys
        layers: dict[str, int] = {}
        for name, spec in self._specs.items():
            parts = name.split(".")
            if "layers" in parts:
                index = parts.index("layers")
                if index + 1 < len(parts):
                    stem = ".".join(parts[: index + 2])
                    layers[stem] = layers.get(stem, 0) + 4 * spec.elements
        self.max_layer_bytes = max(layers.values(), default=0)
        self.embedding_bytes = sum(
            4 * spec.elements
            for name, spec in self._specs.items()
            if name.endswith("embed_tokens.weight")
        )
        largest_source = max(
            (2 * spec.elements for spec in self._specs.values()), default=0
        )
        self.peak_bytes = (
            max(self.max_layer_bytes, self.embedding_bytes) + largest_source
        )

    @contextmanager
    def stage(self, module: nn.Module, prefix: str) -> Iterator[None]:
        """Stage one module for a complete layer visit, then restore its slots.

        Keep the context around all context chunks for that layer. Staging each
        chunk separately would repeat checkpoint reads and transfers needlessly.
        """
        saved = _slots(module, prefix)
        for slot in saved:
            key = _source_name(slot.full_name)
            if key in self._keys and tuple(slot.value.shape) != self._specs[key].shape:
                raise ValueError(f"streamed module shape differs for {slot.full_name}")
        target_bytes = sum(
            slot.value.numel()
            * (
                4
                if _source_name(slot.full_name) in self._keys
                else slot.value.element_size()
            )
            for slot in saved
        )
        source_bytes = max(
            (
                2 * self._specs[_source_name(slot.full_name)].elements
                for slot in saved
                if _source_name(slot.full_name) in self._keys
            ),
            default=0,
        )
        with self.guard.activate():
            self.guard.check(
                target_bytes + source_bytes, operation="stage Ouro weights"
            )
            try:
                if _signature(self.path) != self._signature:
                    raise ValueError("Ouro base weights changed after validation")
                with safe_open(str(self.path), framework="pt", device="cpu") as handle:
                    for slot in saved:
                        checkpoint()
                        key = _source_name(slot.full_name)
                        if key in self._keys:
                            source = handle.get_tensor(key)
                            spec = self._specs[key]
                            if (
                                tuple(source.shape) != spec.shape
                                or source.dtype != torch.bfloat16
                            ):
                                raise ValueError(
                                    f"Ouro base tensor {key} changed after validation"
                                )
                            value = source.to(device=self.device, dtype=torch.float32)
                            del source
                        else:
                            if slot.value.is_meta:
                                raise ValueError(
                                    "no materialized value for streamed tensor "
                                    f"{slot.full_name}"
                                )
                            if slot.parameter and slot.value.dtype != torch.float32:
                                raise ValueError(
                                    f"streamed adapter {slot.full_name} "
                                    "must remain FP32"
                                )
                            value = slot.value.detach().to(
                                device=self.device, copy=True
                            )
                        if slot.parameter:
                            slot.owner._parameters[slot.name] = nn.Parameter(
                                value, requires_grad=False
                            )
                        else:
                            slot.owner._buffers[slot.name] = value
                        del value
                if _signature(self.path) != self._signature:
                    raise ValueError("Ouro base weights changed while staging")
                yield
            finally:
                # A submitted MPS command may still read staged parameters.
                # Fence before dropping them; restore even when the fence fails.
                active_error = sys.exc_info()[1]
                cleanup_error: RuntimeError | None = None
                try:
                    if self.device.type == "mps":
                        torch.mps.synchronize()
                except RuntimeError as error:
                    cleanup_error = error
                finally:
                    for slot in saved:
                        if slot.parameter:
                            slot.owner._parameters[slot.name] = cast(
                                nn.Parameter, slot.value
                            )
                        else:
                            slot.owner._buffers[slot.name] = slot.value
                    if self.device.type == "mps":
                        try:
                            torch.mps.empty_cache()
                        except RuntimeError as error:
                            cleanup_error = cleanup_error or error
                if cleanup_error is not None:
                    if active_error is None:
                        raise cleanup_error
                    active_error.add_note(
                        f"MPS staging cleanup also failed: {cleanup_error}"
                    )
