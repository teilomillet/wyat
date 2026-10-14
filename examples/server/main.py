# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""An ASGI factory: importing this module does not load model weights."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Literal, cast

import wyat

if TYPE_CHECKING:
    from fastapi import FastAPI


def app() -> FastAPI:
    path = os.environ.get("WYAT_MODEL")
    if not path or not path.strip():
        raise ValueError("Set WYAT_MODEL to a supported local model directory")
    backend = os.environ.get("WYAT_BACKEND", "torch")
    if backend not in {"torch", "mlx"}:
        raise ValueError("WYAT_BACKEND must be torch or mlx")

    from wyat.server import create_app

    model = wyat.load(path, backend=cast(Literal["torch", "mlx"], backend))
    return create_app(model)
