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
    max_length: int = 2048,
    token_cache_size: int = 0,
    attention: str = "sdpa",
    compile: bool = False,
) -> Model:
    """Load local Kai weights. Select the runtime explicitly; never download code.

    Torch defaults to CPU; MLX defaults to Apple GPU. Compilation is an opt-in
    MLX experiment and must pass the caller's quality and performance gates.
    """
    scorer: Scorer
    if backend == "torch":
        if compile:
            raise ValueError("compile is supported only by the MLX backend")
        try:
            from wyat.backends.kai.torch import KaiScorer
        except ModuleNotFoundError as exc:
            _raise_missing_runtime(exc, "torch")
            raise

        torch_scorer = KaiScorer.from_directory(
            path,
            device="cpu" if device is None else device,
            max_length=max_length,
            token_cache_size=token_cache_size,
            attention=attention,
        )
        scorer, model_id = torch_scorer, torch_scorer.model_id
    elif backend == "mlx":
        if attention != "sdpa":
            raise ValueError("MLX uses its native attention implementation")
        try:
            from wyat.backends.kai.mlx import MLXKaiScorer
        except ModuleNotFoundError as exc:
            _raise_missing_runtime(exc, "mlx")
            raise

        mlx_scorer = MLXKaiScorer.from_directory(
            path,
            device="gpu" if device is None else device,
            max_length=max_length,
            token_cache_size=token_cache_size,
            compile=compile,
        )
        scorer, model_id = mlx_scorer, mlx_scorer.model_id
    else:
        raise ValueError(f"unsupported backend: {backend}")
    return Model(scorer, model_id)
