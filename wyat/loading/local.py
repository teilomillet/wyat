# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Select a local artifact runtime without executing model code.

The only network path downloads recognized, pinned releases into a verified
local cache before loading. A release without pins fails before any request.
"""

import sys
from collections.abc import Sequence
from functools import wraps
from pathlib import Path
from typing import Any, Literal, cast

from wyat.core import Model, Request, Scorer
from wyat.download import REPOSITORY
from wyat.resources import ResourceGuard, current_guard


class _GuardedScorer:
    """Retain a load policy across all public Model inference paths."""

    def __init__(self, scorer: Scorer, guard: ResourceGuard) -> None:
        self._scorer = scorer
        self._guard = guard

    def _run(self, method: Any, *args: Any, **kwargs: Any) -> Any:
        with self._guard.activate():
            self._guard.check(operation="inference")
            return method(*args, **kwargs)

    def score_many(self, requests: Sequence[Request]) -> Sequence[Sequence[float]]:
        return cast(
            Sequence[Sequence[float]], self._run(self._scorer.score_many, requests)
        )

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._scorer, name)
        if name in {"logits", "predict", "_records_logits"} and callable(value):

            @wraps(value)
            def guarded(*args: Any, **kwargs: Any) -> Any:
                return self._run(value, *args, **kwargs)

            return guarded
        return value


def _configure_current_runtime(backend: str, device: str | None) -> None:
    """Use a lazily imported runtime before it allocates model parameters."""
    guard = current_guard()
    if guard is None or not guard.enabled:
        return
    if backend == "torch":
        runtime = sys.modules.get("torch")
        if runtime is not None:
            guard.configure_torch(runtime, "cpu" if device is None else device)
    else:
        runtime = sys.modules.get("mlx.core")
        if runtime is not None:
            guard.configure_mlx(runtime)


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


def _load(
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
    memory_mode: Literal["resident", "streaming"] = "resident",
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
    if memory_mode != "resident" and not is_ouro:
        raise ValueError("streaming memory mode applies only to Ouro")
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
        _configure_current_runtime(backend, device)
        ouro_torch = load_ouro(
            path,
            # Ouro artifacts declare their own execution policy: cpu-fp32
            # defaults to CPU, mps-fp32 defaults to the Apple GPU, and
            # cuda-bf16-autocast defaults to CUDA when available.
            device=device,
            max_length=max_length,
            prefix_cache_size=ouro_prefix_cache_size,
            attention=attention,
            memory_mode=memory_mode,
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
            _configure_current_runtime(backend, device)
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
            _configure_current_runtime(backend, device)
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

            _configure_current_runtime(backend, device)
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

            _configure_current_runtime(backend, device)
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

            _configure_current_runtime(backend, device)
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

            _configure_current_runtime(backend, device)
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
    max_resource_fraction: float = 0.85,
    memory_mode: Literal["resident", "streaming"] = "resident",
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

    ``memory_mode`` is ``"resident"`` by default. For Ouro artifacts on macOS,
    ``"streaming"`` stages the original base weights one FP32 module at a time
    and spills the final context K/V bank to an anonymous uncached temporary
    file instead of retaining it in RAM. It requires CPU or Apple GPU execution,
    v2 chunked context execution, shared-context candidate batching and no
    prefix cache, and it makes the produced backbone scoring-only. Streaming
    replaces the three-times-on-disk loading estimate with the artifact's
    token-aware working-set estimate; the same fraction still bounds RAM and
    temporary disk. Non-macOS platforms reject streaming rather than run
    without the admission guard and uncached-file policy.

    ``max_resource_fraction`` defaults to 0.85 and may be set in (0, 1]. On
    macOS, admission accounts for other applications before model allocation
    and every inference call. Weight files in the root and recognized
    base/backbone directories receive a conservative three-times-on-disk loading
    estimate, covering dtype expansion and staging;
    this is an admission estimate, not a prediction of every runtime allocation.
    CPU workers and Apple GPU allocation limits are also bounded. These are
    cooperative safeguards, not a hard whole-machine CPU/GPU utilization cap:
    another app or an already queued kernel can change resource use between
    checks. The runtime settings affect the process, including other models.
    Other operating systems retain their previous resource behavior.
    """
    if memory_mode not in {"resident", "streaming"}:
        raise ValueError("memory_mode must be resident or streaming")
    guard = ResourceGuard(max_resource_fraction)
    with guard.activate():
        guard.check(operation="model loading")
        resolved = _local_directory(path)
        if guard.enabled and memory_mode == "resident":
            root = Path(resolved)
            weight_bytes = sum(
                file.stat().st_size
                for directory in (root, root / "base", root / "backbone")
                if directory.is_dir()
                for file in directory.iterdir()
                if file.is_file()
                and file.suffix in {".safetensors", ".bin", ".pt", ".pth"}
            )
            guard.check(3 * weight_bytes, operation="model loading")
        model = _load(
            resolved,
            backend=backend,
            device=device,
            max_length=max_length,
            token_cache_size=token_cache_size,
            prefix_cache_size=prefix_cache_size,
            attention=attention,
            compile=compile,
            responsive=responsive,
            temperature=temperature,
            memory_mode=memory_mode,
        )
        if guard.enabled:
            guard.check(operation="loaded model")
            if getattr(model._scorer, "resource_guard", None) is not guard:
                model._scorer = _GuardedScorer(model._scorer, guard)
        return model
