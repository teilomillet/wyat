# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Select a local artifact runtime without executing model code.

The only network path downloads recognized, pinned releases into a verified
local cache before loading. A release without pins fails before any request.
"""

import sys
from pathlib import Path
from typing import Literal

from wyat.core import Model, Scorer
from wyat.download import REPOSITORY


def _raise_missing_runtime(
    error: ModuleNotFoundError, backend: Literal["torch", "mlx"]
) -> None:
    """Explain an absent runtime; leave nested dependency failures untouched."""
    if error.name != backend:
        return
    extra = "inference" if backend == "torch" else "mlx"
    runtime = "PyTorch" if backend == "torch" else "MLX"
    platform = " MLX requires Apple Silicon macOS." if backend == "mlx" else ""
    message = (
        f"{runtime} is not installed; in your application's project, "
        f"run: python -m pip install 'wyat[{extra}]'."
        f"{platform} For a local wheel, install the same wheel with [{extra}]. "
        f"For a source checkout, run: uv add /absolute/path/to/wyat --extra {extra}. "
        f"From the Wyat checkout, run: uv sync --extra {extra}"
    )
    raise ModuleNotFoundError(message, name=backend) from error


def _local_directory(path: str | Path) -> str | Path:
    """Return a local directory, downloading the pinned release when it is named."""
    from wyat.download import RELEASES, download

    # Named releases have explicit identities; other strings remain local paths.
    repository = Path(path).as_posix()
    if repository not in RELEASES or Path(path).exists():
        return path

    def log(line: str) -> None:
        if not line.startswith("  ok "):
            print(line, file=sys.stderr)

    return download(repository=repository, log=log)


def load(
    path: str | Path = REPOSITORY,
    *,
    backend: Literal["torch", "mlx"] = "torch",
    device: str | None = None,
    max_length: int | None = None,
    token_cache_size: int = 0,
    prefix_cache_size: int | None = None,
    attention: str = "sdpa",
    compile: bool = False,
    responsive: bool = True,
    temperature: float = 1.0,
) -> Model:
    """Load a local model directory, or the pinned release by its Hub name.

    The default is ``teilomillet/wyat-1-800m``; it raises a clear error until a
    release is selected and pinned. Explicitly choose ``teilomillet/wyat-1-600m``
    to download and verify that release into the Wyat cache. A local
    directory with that relative path takes precedence. Torch defaults to CPU;
    MLX defaults to Apple GPU. Ouro decision artifacts declare their own
    execution policy: ``cpu-fp32`` defaults to CPU, ``mps-fp32`` to the Apple
    GPU on Apple Silicon, and ``cuda-bf16-autocast`` to CUDA when available.
    Compilation is an opt-in MLX experiment and must
    pass the caller's quality and performance gates. Native and option-fork MLX
    GPU inference pauses 1 ms after every fourth backbone layer so other
    applications can submit GPU work; ``responsive=False`` removes those pauses
    and is rejected by engines that do not pace.

    ``prefix_cache_size`` bounds a least-recently-used cache of exact token
    prefixes. ``None`` keeps the family defaults (native 16, option-fork 0);
    a nonnegative integer sets the entry count for the selected family, and
    ``0`` disables caching. Option-fork entries retain the state-block prefix
    plus attention and gated-delta-net state, so size them against memory.
    """
    if prefix_cache_size is not None and (
        type(prefix_cache_size) is not int or prefix_cache_size < 0
    ):
        raise ValueError("prefix_cache_size must be a nonnegative integer or None")
    if type(responsive) is not bool:
        raise ValueError("responsive must be a boolean")
    native_prefix_cache_size = 16 if prefix_cache_size is None else prefix_cache_size
    fork_prefix_cache_size = 0 if prefix_cache_size is None else prefix_cache_size
    ouro_prefix_cache_size = 0 if prefix_cache_size is None else prefix_cache_size
    path = _local_directory(path)
    is_native = (Path(path) / "native_manifest.json").is_file()
    is_fork = (Path(path) / "fork_manifest.json").is_file()
    is_ouro = (Path(path) / "ouro_manifest.json").is_file()
    if not responsive and not (backend == "mlx" and (is_native or is_fork)):
        raise ValueError(
            "responsive=False applies only to native MLX and option-fork MLX inference"
        )
    if is_ouro:
        if compile:
            raise ValueError("Ouro decision inference does not support compile")
        if backend != "torch":
            raise ValueError("Ouro decision inference supports the torch backend only")
        try:
            from wyat.loading.ouro import load_ouro
        except ModuleNotFoundError as exc:
            _raise_missing_runtime(exc, "torch")
            raise
        ouro_torch = load_ouro(
            path,
            # Ouro artifacts declare their own execution policy: cpu-fp32
            # defaults to CPU, mps-fp32 defaults to the Apple GPU, and
            # cuda-bf16-autocast defaults to CUDA when available.
            device=device,
            max_length=max_length,
            prefix_cache_size=ouro_prefix_cache_size,
            attention=attention,
        )
        return Model(ouro_torch, ouro_torch.model_id, temperature=temperature)
    if is_fork:
        # Option-fork artifacts (Qwen3.5 hybrid) retain bounded cross-call
        # state-block prefix caches; token caching stays question-local.
        if attention != "sdpa" or compile:
            raise ValueError("option-fork inference uses sdpa attention, uncompiled")
        fork: Scorer
        if backend == "torch":
            try:
                from wyat.loading.fork import load_fork
            except ModuleNotFoundError as exc:
                _raise_missing_runtime(exc, "torch")
                raise
            fork_torch = load_fork(
                path,
                device="cpu" if device is None else device,
                max_length=max_length,
                prefix_cache_size=fork_prefix_cache_size,
            )
            fork, model_id = fork_torch, fork_torch.model_id
        elif backend == "mlx":
            try:
                from wyat.loading.fork_mlx import load_fork_mlx
            except ModuleNotFoundError as exc:
                _raise_missing_runtime(exc, "mlx")
                raise
            fork_mlx = load_fork_mlx(
                path,
                device="gpu" if device is None else device,
                max_length=max_length,
                responsive=responsive,
                prefix_cache_size=fork_prefix_cache_size,
            )
            fork, model_id = fork_mlx, fork_mlx.model_id
        else:
            raise ValueError(f"unsupported backend: {backend}")
        return Model(fork, model_id, temperature=temperature)
    wyat_max_length = 2048 if max_length is None else max_length
    scorer: Scorer
    if backend == "torch":
        if compile:
            raise ValueError("compile is supported only by the MLX backend")
        if is_native:
            if attention != "sdpa":
                raise ValueError("native PyTorch inference requires sdpa attention")
            try:
                from wyat.loading.native import load_native
            except ModuleNotFoundError as exc:
                _raise_missing_runtime(exc, "torch")
                raise

            native_torch = load_native(
                path,
                device="cpu" if device is None else device,
                max_length=max_length,
                prefix_cache_size=native_prefix_cache_size,
            )
            scorer, model_id = native_torch, native_torch.model_id
        else:
            try:
                from wyat.backends.wyat.torch import WyatScorer
            except ModuleNotFoundError as exc:
                _raise_missing_runtime(exc, "torch")
                raise

            torch_scorer = WyatScorer.from_directory(
                path,
                device="cpu" if device is None else device,
                max_length=wyat_max_length,
                token_cache_size=token_cache_size,
                attention=attention,
            )
            scorer, model_id = torch_scorer, torch_scorer.model_id
    elif backend == "mlx":
        if attention != "sdpa":
            raise ValueError("MLX uses its native attention implementation")
        if is_native:
            if compile:
                raise ValueError("native MLX inference does not support compile")
            try:
                from wyat.loading.mlx import load_native_mlx
            except ModuleNotFoundError as exc:
                _raise_missing_runtime(exc, "mlx")
                raise

            native_mlx = load_native_mlx(
                path,
                device="gpu" if device is None else device,
                max_length=max_length,
                responsive=responsive,
                prefix_cache_size=native_prefix_cache_size,
            )
            scorer, model_id = native_mlx, native_mlx.model_id
        else:
            try:
                from wyat.backends.wyat.mlx import MLXWyatScorer
            except ModuleNotFoundError as exc:
                _raise_missing_runtime(exc, "mlx")
                raise

            mlx_scorer = MLXWyatScorer.from_directory(
                path,
                device="gpu" if device is None else device,
                max_length=wyat_max_length,
                token_cache_size=token_cache_size,
                compile=compile,
            )
            scorer, model_id = mlx_scorer, mlx_scorer.model_id
    else:
        raise ValueError(f"unsupported backend: {backend}")
    return Model(scorer, model_id, temperature=temperature)
