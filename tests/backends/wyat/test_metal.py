# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Lossless storage, optional admission and native FP32 projection contracts."""

from __future__ import annotations

from importlib.metadata import version
from typing import Any, cast

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx.nn.layers.base import Module  # noqa: E402
from mlx.nn.layers.linear import Linear  # noqa: E402
from mlx_lm.models.qwen3 import Qwen3Model  # noqa: E402

from wyat.backends.wyat import metal  # noqa: E402


def test_storage_preserves_zero_sign_and_subnormal_words() -> None:
    words = np.array(
        [0, 0x80000000, 0x00010000, 0x80010000, 0x3F800000, 0xBF800000],
        dtype=np.uint32,
    ).reshape(2, 3)
    with mx.stream(mx.cpu):
        weight = mx.array(words.view(np.float32))
        stored = metal._storage(weight)
        assert stored is not None
        assert stored.dtype == mx.bfloat16
        restored = metal._widen(stored).view(mx.uint32)
        np.testing.assert_array_equal(np.array(restored), words)
        np.testing.assert_array_equal(np.array(weight.view(mx.uint32)), words)


def test_storage_keeps_genuine_fp32_and_rejects_nonfinite() -> None:
    with mx.stream(mx.cpu):
        words = np.array([[0x3F800001, 0xBF800001]], dtype=np.uint32)
        assert metal._storage(mx.array(words.view(np.float32))) is None
        with pytest.raises(ValueError, match="Nonfinite"):
            metal._storage(mx.array([[float("nan")]], dtype=mx.float32))
        original = mx.ones((2, 3), dtype=mx.bfloat16)
        assert metal._storage(original) is original
        with pytest.raises(ValueError, match="Nonfinite"):
            metal._storage(mx.array([[float("inf")]], dtype=mx.bfloat16))


@pytest.mark.parametrize(
    "message",
    [
        "out of memory",
        "allocation failed",
        "resource exhausted",
        "std::bad_alloc",
        "cannot allocate memory",
        "failed to allocate memory",
        "numerical comparison failed",
    ],
)
def test_optional_compiler_handling_never_hides_runtime_failures(message: str) -> None:
    assert not metal._unavailable(RuntimeError(message))
    if message != "numerical comparison failed":
        assert not metal._unavailable(
            RuntimeError(
                "[metal::Device] Unable to build metal library from source "
                "'metal_tensor' file not found " + message
            )
        )


def test_missing_optional_header_is_unavailable() -> None:
    assert metal._unavailable(
        RuntimeError(
            "[metal::Device] Unable to build metal library from source "
            "'metal_tensor' file not found"
        )
    )


def test_cpu_and_compiled_resources_do_not_probe_compiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected() -> None:
        raise AssertionError("Optional compiler must not run")

    monkeypatch.setattr(metal, "_capability", unexpected)
    untouched = cast(Qwen3Model, None)
    assert not metal.optimize_projections(
        untouched, device=mx.Device(mx.cpu), compiled=False
    )
    assert not metal.optimize_projections(
        untouched, device=mx.Device(mx.gpu), compiled=True
    )


def test_unqualified_runtime_and_architecture_do_not_probe_compiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected() -> None:
        raise AssertionError("Optional compiler must not run")

    monkeypatch.setattr(metal, "_capability", unexpected)
    monkeypatch.setattr(metal, "version", lambda name: "0.32.2")
    assert metal._prepare(device=mx.Device(mx.gpu), compiled=False) is None
    monkeypatch.setattr(metal, "version", lambda name: "0.32.3")
    monkeypatch.setattr(
        mx, "device_info", lambda device: {"architecture": "applegpu_g17s"}
    )
    assert metal._prepare(device=mx.Device(mx.gpu), compiled=False) is None


def test_installer_preserves_ineligible_layers_and_updates_only_after_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not mx.metal.is_available():
        pytest.skip("Metal GPU required for installer validation")
    monkeypatch.setattr(metal, "version", lambda name: "0.32.3")
    monkeypatch.setattr(
        mx, "device_info", lambda device: {"architecture": "applegpu_g16s"}
    )
    monkeypatch.setattr(metal, "_capability", lambda: object())
    good = Linear(1024, 1024, bias=False)
    precise = Linear(1024, 1024, bias=False)
    biased = Linear(1024, 1024, bias=True)
    head = Linear(1024, 1024, bias=False)
    good.weight = mx.ones((1024, 1024), dtype=mx.float32)
    precise.weight = mx.full((1024, 1024), 1.0000001192092896, dtype=mx.float32)
    layers: list[tuple[str, Module]] = [
        ("layers.0.good", good),
        ("layers.0.precise", precise),
        ("layers.0.biased", biased),
        ("head.query", head),
    ]

    class Backbone:
        def __init__(self) -> None:
            self.updates: list[dict[str, Any]] = []

        def named_modules(self) -> list[tuple[str, Module]]:
            return layers

        def update_modules(self, replacements: dict[str, Any], *, strict: bool) -> None:
            assert strict
            self.updates.append(replacements)

    backbone = Backbone()
    model = cast(Qwen3Model, backbone)
    prepare = metal._checkpoint_weights(model)
    stored = mx.ones((1024, 1024), dtype=mx.bfloat16)
    for name in ("layers.0.good.weight", "layers.0.precise.weight"):
        prepared = prepare(name, stored)
        assert prepared is not stored
        assert prepared.dtype == mx.bfloat16
        np.testing.assert_array_equal(
            np.array(prepared.view(mx.uint16)), np.array(stored.view(mx.uint16))
        )
    words = np.array([0, 0x8000, 1, 0x8001, 0x3F80, 0xBF80], dtype=np.uint16)
    with mx.stream(mx.cpu):
        source = mx.array(words).view(mx.bfloat16)
        copied = prepare("layers.0.good.weight", source)
        assert copied is not source
        np.testing.assert_array_equal(np.array(copied.view(mx.uint16)), words)
        np.testing.assert_array_equal(np.array(source.view(mx.uint16)), words)
    assert prepare("layers.0.biased.weight", stored).dtype == mx.float32
    assert prepare("head.query.weight", stored).dtype == mx.float32
    assert metal.optimize_projections(model, device=mx.Device(mx.gpu), compiled=False)
    assert len(backbone.updates) == 1
    replacement = backbone.updates[0]["layers"][0]
    assert set(replacement) == {"good"}
    assert replacement["good"].weight.dtype == mx.bfloat16
    assert good.weight.dtype == precise.weight.dtype == mx.float32

    backbone.updates.clear()
    precise.weight = mx.full((1024, 1024), float("nan"), dtype=mx.float32)
    with pytest.raises(ValueError, match="Nonfinite"):
        metal.optimize_projections(model, device=mx.Device(mx.gpu), compiled=False)
    assert backbone.updates == []
    assert good.weight.dtype == mx.float32


@pytest.mark.parametrize(
    "rows,reduction,columns", [(17, 1024, 1024), (513, 3072, 1024), (17, 1024, 3072)]
)
def test_qualified_projection_preserves_native_scaled_fp32(
    rows: int, reduction: int, columns: int
) -> None:
    if (
        not mx.metal.is_available()
        or version("mlx") != "0.32.3"
        or mx.device_info(mx.gpu).get("architecture") != "applegpu_g16s"
    ):
        pytest.skip("Qualified M4 Pro architecture required")
    operation = metal._prepare(device=mx.Device(mx.gpu), compiled=False)
    if operation is None:
        pytest.skip("Qualified optional Metal compiler APIs required")
    rng = np.random.default_rng(481921)
    x_values = (rng.standard_normal((rows, reduction)) * 65536).astype(np.float32)
    source = (rng.standard_normal((columns, reduction)) / np.sqrt(reduction)).astype(
        np.float32
    )
    words = source.view(np.uint32) & np.uint32(0xFFFF0000)
    with mx.stream(mx.gpu):
        weight = mx.array((words >> 16).astype(np.uint16)).view(mx.bfloat16)
        x = mx.array(x_values)
        expected = mx.matmul(x, mx.array(words.view(np.float32)).T)
        actual = operation(x, weight)
        mx.eval(expected, actual)
        np.testing.assert_allclose(
            np.array(actual), np.array(expected), atol=1e-4, rtol=1e-4
        )
        np.testing.assert_array_equal(
            np.array(x.view(mx.uint32)), x_values.view(np.uint32)
        )
        np.testing.assert_array_equal(
            np.array(weight.view(mx.uint16)), (words >> 16).astype(np.uint16)
        )


def test_unsupported_geometry_and_fp32_weights_keep_native_values() -> None:
    if not mx.metal.is_available() or version("mlx") != "0.32.3":
        pytest.skip("Qualified MLX version and Metal GPU required")
    with mx.stream(mx.gpu):
        x = mx.arange(3 * 17, dtype=mx.float32).reshape(3, 17) / 13
        weight = mx.arange(5 * 17, dtype=mx.float32).reshape(5, 17) / 17
        operation = metal.MetalProjection()
        expected = mx.matmul(x, weight.T)
        actual = operation(x, weight)
        mx.eval(expected, actual)
        np.testing.assert_array_equal(np.array(actual), np.array(expected))
