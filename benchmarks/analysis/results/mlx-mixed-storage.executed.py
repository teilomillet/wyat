"""Frozen tiny CPU-only graph probe; no speed or model-quality claim."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from typing import cast

import mlx.core as mx

root = Path(__file__).resolve().parent
with mx.stream(mx.cpu):
    x = mx.arange(32, dtype=mx.float32).reshape(2, 16) / 32
    source = (
        (mx.arange(256, dtype=mx.float32).reshape(16, 16) / 256)
        .astype(mx.bfloat16)
        .astype(mx.float32)
    )
    half = source.astype(mx.float16)
    mixed = x @ half.T
    mx.export_to_dot(str(root / "mlx-mixed-storage.dot"), mixed, x=x, half=half)
    baseline = x @ source.T
    mx.eval(source, half, mixed, baseline)
    receipt: dict[str, object] = {
        "scope": "tiny CPU graph inspection only; no performance or checkpoint gate",
        "mlx": version("mlx"),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_dtype": str(x.dtype),
        "stored_weight_dtype": str(half.dtype),
        "output_dtype": str(mixed.dtype),
        "source_values_preserved": bool(
            mx.all(source == half.astype(mx.float32)).item()
        ),
        "max_abs_difference": float(
            cast(float, mx.max(mx.abs(mixed - baseline)).item())
        ),
        "graph_sha256": hashlib.sha256(
            (root / "mlx-mixed-storage.dot").read_bytes()
        ).hexdigest(),
    }
    (root / "mlx-mixed-storage.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))
