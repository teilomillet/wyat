"""UNIX entry point: JSONL on stdin/stdout; diagnostics on stderr."""

from __future__ import annotations

import argparse
import io
import json
import sys
from collections.abc import Iterator, Sequence
from typing import Literal

from . import load
from .core import Decision, Model, Request, request_from_dict
from .loading.local import _raise_missing_runtime
from .wire import JsonValue, decode_json


class Arguments(argparse.Namespace):
    command: Literal["decide", "evaluate", "serve"]
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


def _decide_batch(
    model: Model, rows: Sequence[tuple[int, Request]]
) -> tuple[Decision, ...]:
    try:
        return model.decide_many([request for _, request in rows])
    except BrokenPipeError:
        raise
    except Exception:
        if rows:
            first, last = rows[0][0], rows[-1][0]
            location = f"line {first}" if first == last else f"lines {first}..{last}"
            print(
                f"wyat: inference failed for input {location}; "
                "no decisions emitted for this batch",
                file=sys.stderr,
            )
        raise


def main(argv: list[str] | None = None) -> int:
    """Run with UTF-8 standard streams; encoding and runtime settings persist."""
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper) and (
            stream.encoding.lower().replace("-", "") != "utf8"
            or stream.errors != "strict"
        ):
            stream.reconfigure(encoding="utf-8", errors="strict")
    parser = argparse.ArgumentParser(
        prog="wyat",
        description="Load, evaluate and serve local decision models.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    descriptions = {
        "decide": "Read request JSONL from stdin; write decision JSONL to stdout.",
        "evaluate": "Read labeled case JSONL from stdin; write one evaluation report.",
        "serve": "Serve the local model over HTTP; no stdin input required.",
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
        if name == "decide":
            command.add_argument(
                "--batch-size", type=int, default=1, help="requests per batch"
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
    if args.command == "decide" and args.batch_size <= 0:
        parser.error("--batch-size must be positive")
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
            if args.backend != "torch":
                raise ValueError("--torch-threads requires a PyTorch engine")
            try:
                import torch
            except ModuleNotFoundError as exc:
                _raise_missing_runtime(exc, "torch")
                raise

            torch.set_num_threads(args.torch_threads)
        if args.mlx_cache_limit_mib is not None:
            metal_candidate = args.backend == "mlx" and args.device in (None, "gpu")
            if not metal_candidate:
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
            pending: list[tuple[int, Request]] = []
            for number, value in _rows():
                try:
                    pending.append((number, request_from_dict(value)))
                except ValueError as exc:
                    raise ValueError(f"line {number}: {exc}") from exc
                if len(pending) == args.batch_size:
                    for result in _decide_batch(model, pending):
                        _write(result.to_dict())
                    pending.clear()
            for result in _decide_batch(model, pending):
                _write(result.to_dict())
        elif args.command == "evaluate":
            from .evaluation import case_from_dict, evaluate

            cases = []
            for number, value in _rows():
                try:
                    cases.append(case_from_dict(value))
                except ValueError as exc:
                    raise ValueError(f"line {number}: {exc}") from exc
            report = evaluate(model, cases)
            _write(report.to_dict())
            return 1 if report.failed else 0
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
