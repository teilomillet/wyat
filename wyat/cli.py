"""UNIX entry point: JSONL on stdin/stdout; diagnostics on stderr."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from typing import Literal

from . import load
from .core import Request, request_from_dict
from .loading.local import _raise_missing_runtime
from .wire import JsonValue, decode_json


class Arguments(argparse.Namespace):
    command: Literal["decide", "evaluate", "serve", "benchmark"]
    model: str
    backend: Literal["torch", "mlx"]
    device: str | None
    max_length: int
    token_cache_size: int
    attention: str
    compile: bool
    torch_threads: int | None
    mlx_cache_limit_mib: int | None
    batch_size: int
    baseline_backend: Literal["torch", "mlx"]
    baseline_device: str | None
    baseline_batch_size: int
    repeats: int
    warmup: int
    host: str
    port: int
    max_body_bytes: int


def _rows() -> Iterator[tuple[int, JsonValue]]:
    for number, line in enumerate(sys.stdin, 1):
        if not line.strip():
            continue
        try:
            yield number, decode_json(line)
        except (ValueError, UnicodeError, RecursionError) as exc:
            raise ValueError(f"line {number}: invalid JSON") from exc


def _write(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")))


def main(argv: list[str] | None = None) -> int:
    """Run a command; explicitly requested runtime settings persist in this process."""
    parser = argparse.ArgumentParser(
        prog="wyat", description="Load, evaluate and serve local decision models."
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    descriptions = {
        "decide": "Read request JSONL from stdin; write decision JSONL to stdout.",
        "evaluate": "Read labeled case JSONL from stdin; write one evaluation report.",
        "serve": "Serve the local model over HTTP; no stdin input required.",
        "benchmark": (
            "Read labeled case JSONL from stdin; compare paired baseline and "
            "candidate timings. Report passed and exit status 0 indicate quality "
            "gates passed, not a guaranteed speed improvement."
        ),
    }
    for name, description in descriptions.items():
        command = subcommands.add_parser(
            name,
            help=description,
            description=description,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        command.add_argument("model", help="local Kai artifact directory")
        command.add_argument(
            "--backend", choices=("torch", "mlx"), default="torch", help="runtime"
        )
        command.add_argument(
            "--device",
            help="torch: cpu/cuda; MLX: cpu/gpu; omitted selects CPU/Apple GPU",
        )
        command.add_argument(
            "--max-length", type=int, default=2048, help="maximum rendered input tokens"
        )
        command.add_argument(
            "--token-cache-size",
            type=int,
            default=0,
            help="tokenized option cache entries; 0 disables caching",
        )
        command.add_argument(
            "--attention",
            choices=("sdpa", "eager"),
            default="sdpa",
            help="PyTorch attention; MLX requires sdpa and uses native attention",
        )
        command.add_argument(
            "--compile", action="store_true", help="opt-in MLX compile"
        )
        command.add_argument(
            "--torch-threads",
            type=int,
            help="process-wide PyTorch CPU threads; default leaves runtime unchanged",
        )
        command.add_argument(
            "--mlx-cache-limit-mib",
            type=int,
            help=(
                "process-wide Metal inactive-buffer cache in MiB; 0 disables caching; "
                "omitted leaves runtime unchanged"
            ),
        )
        if name in ("decide", "benchmark"):
            command.add_argument(
                "--batch-size", type=int, default=1, help="requests per batch"
            )
        if name == "benchmark":
            command.add_argument(
                "--baseline-backend",
                choices=("torch", "mlx"),
                default="torch",
                help="baseline runtime",
            )
            command.add_argument(
                "--baseline-device",
                default="cpu",
                help="baseline device; defaults to CPU for both runtimes",
            )
            command.add_argument(
                "--baseline-batch-size",
                type=int,
                default=1,
                help="baseline requests per batch",
            )
            command.add_argument(
                "--repeats",
                type=int,
                default=5,
                help="measured alternating pairs; minimum 3",
            )
            command.add_argument(
                "--warmup", type=int, default=1, help="warmup pairs; minimum 1"
            )
        if name == "serve":
            command.add_argument(
                "--host", default="127.0.0.1", help="HTTP bind address"
            )
            command.add_argument("--port", type=int, default=8000, help="HTTP port")
            command.add_argument(
                "--max-body-bytes",
                type=int,
                default=1_048_576,
                help="maximum HTTP request body bytes",
            )
    args = Arguments()
    parser.parse_args(argv, namespace=args)
    if args.command in ("decide", "benchmark") and args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.command == "benchmark":
        if args.baseline_batch_size <= 0 or args.repeats < 3 or args.warmup < 1:
            parser.error(
                "positive batch sizes, at least 3 repeats and 1 warmup required"
            )
    if args.torch_threads is not None and args.torch_threads <= 0:
        parser.error("--torch-threads must be positive")
    if args.mlx_cache_limit_mib is not None and not (
        0 <= args.mlx_cache_limit_mib <= sys.maxsize >> 20
    ):
        parser.error("--mlx-cache-limit-mib must fit a nonnegative byte count")
    if args.command == "serve":
        if not 1 <= args.port <= 65535:
            parser.error("--port must be between 1 and 65535")
        if args.max_body_bytes <= 0:
            parser.error("--max-body-bytes must be positive")
    try:
        if args.torch_threads is not None:
            if args.backend != "torch" and (
                args.command != "benchmark" or args.baseline_backend != "torch"
            ):
                raise ValueError("--torch-threads requires a PyTorch engine")
            try:
                import torch
            except ModuleNotFoundError as exc:
                _raise_missing_runtime(exc, "torch")
                raise

            torch.set_num_threads(args.torch_threads)
        if args.mlx_cache_limit_mib is not None:
            metal_candidate = args.backend == "mlx" and args.device in (None, "gpu")
            metal_baseline = (
                args.command == "benchmark"
                and args.baseline_backend == "mlx"
                and args.baseline_device == "gpu"
            )
            if not (metal_candidate or metal_baseline):
                raise ValueError("--mlx-cache-limit-mib requires a Metal MLX engine")
            try:
                import mlx.core as mx
            except ModuleNotFoundError as exc:
                _raise_missing_runtime(exc, "mlx")
                raise

            if not mx.metal.is_available():
                raise ValueError("--mlx-cache-limit-mib requires available Apple Metal")
            mx.set_cache_limit(args.mlx_cache_limit_mib * (1 << 20))
        model = load(
            args.model,
            backend=args.backend,
            device=args.device,
            max_length=args.max_length,
            token_cache_size=args.token_cache_size,
            attention=args.attention,
            compile=args.compile,
        )
        if args.command == "decide":
            pending: list[Request] = []
            for number, value in _rows():
                try:
                    pending.append(request_from_dict(value))
                except ValueError as exc:
                    raise ValueError(f"line {number}: {exc}") from exc
                if len(pending) == args.batch_size:
                    for result in model.decide_many(pending):
                        _write(result.to_dict())
                    pending.clear()
            for result in model.decide_many(pending):
                _write(result.to_dict())
        elif args.command in ("evaluate", "benchmark"):
            from .evaluation import case_from_dict, evaluate

            cases = []
            for number, value in _rows():
                try:
                    cases.append(case_from_dict(value))
                except ValueError as exc:
                    raise ValueError(f"line {number}: {exc}") from exc
            if args.command == "evaluate":
                report = evaluate(model, cases)
                _write(report.to_dict())
                return 1 if report.failed else 0
            from .benchmark import BenchmarkEngine, BenchmarkProtocol, benchmark_pair

            baseline = load(
                args.model,
                backend=args.baseline_backend,
                device=args.baseline_device,
                max_length=args.max_length,
                token_cache_size=0,
            )
            benchmark = benchmark_pair(
                BenchmarkEngine(
                    "baseline",
                    baseline,
                    args.baseline_batch_size,
                    conditions={
                        "backend": args.baseline_backend,
                        "device": str(args.baseline_device),
                        "torch_threads": str(args.torch_threads),
                        "max_length": str(args.max_length),
                        "mlx_cache_limit_mib": str(args.mlx_cache_limit_mib),
                        "token_cache_size": "0",
                        "compile": "False",
                        "attention": "sdpa"
                        if args.baseline_backend == "torch"
                        else "native",
                    },
                ),
                BenchmarkEngine(
                    "candidate",
                    model,
                    args.batch_size,
                    conditions={
                        "backend": args.backend,
                        "device": args.device
                        or ("cpu" if args.backend == "torch" else "gpu"),
                        "compile": str(args.compile),
                        "token_cache_size": str(args.token_cache_size),
                        "attention": args.attention
                        if args.backend == "torch"
                        else "native",
                        "torch_threads": str(args.torch_threads),
                        "max_length": str(args.max_length),
                        "mlx_cache_limit_mib": str(args.mlx_cache_limit_mib),
                    },
                ),
                cases,
                protocol=BenchmarkProtocol(repeats=args.repeats, warmup=args.warmup),
            )
            _write(benchmark.to_dict())
            return 0 if benchmark.passed else 1
        else:
            import uvicorn

            from .server import create_app

            uvicorn.run(
                create_app(model, max_body_bytes=args.max_body_bytes),
                host=args.host,
                port=args.port,
            )
    except BrokenPipeError:
        return 0
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f"wyat: {exc}", file=sys.stderr)
        return 1
    return 0
