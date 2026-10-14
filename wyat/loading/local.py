"""Select a local artifact runtime without downloading or executing model code."""

from pathlib import Path
from typing import Literal

from wyat.core import Model, Scorer


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


def load(
    path: str | Path,
    *,
    backend: Literal["torch", "mlx"] = "torch",
    device: str | None = None,
    max_length: int | None = None,
    token_cache_size: int = 0,
    prefix_cache_size: int = 16,
    attention: str = "sdpa",
    compile: bool = False,
    temperature: float = 1.0,
) -> Model:
    """Load local Wyat or native weights; select the runtime explicitly.

    Torch defaults to CPU; MLX defaults to Apple GPU. Compilation is an opt-in
    MLX experiment and must pass the caller's quality and performance gates.
    """
    if type(prefix_cache_size) is not int or prefix_cache_size < 0:
        raise ValueError("prefix_cache_size must be a nonnegative integer")
    is_native = (Path(path) / "native_manifest.json").is_file()
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
                prefix_cache_size=prefix_cache_size,
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
                prefix_cache_size=prefix_cache_size,
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
