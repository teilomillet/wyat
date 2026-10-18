# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""UNIX entry point: JSONL on stdin/stdout; diagnostics on stderr."""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import sys
from collections.abc import Iterator, Sequence
from typing import Literal, cast

from . import load
from .core import Decision, Model, Request, request_from_dict
from .download import RELEASES, REPOSITORY, DownloadError, download
from .loading.local import _raise_missing_runtime
from .wire import JsonValue, decode_json


class Arguments(argparse.Namespace):
    command: Literal["decide", "evaluate", "serve", "download"]
    model: str
    directory: str
    backend: Literal["torch", "mlx"]
    device: str | None
    max_length: int
    token_cache_size: int
    prefix_cache_size: int
    attention: str
    compile: bool
    responsive: bool
    temperature: float
    torch_threads: int | None
    mlx_cache_limit_mib: int | None
    batch_size: int
    host: str
    port: int
    max_body_bytes: int
    max_queue_size: int
    queue_timeout: float
    log_access: bool


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


def _show_access_log(access_logger: logging.Logger) -> None:
    """Print access records to stderr unless the process already handles them."""
    if access_logger.hasHandlers():
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s:     %(message)s"))
    access_logger.addHandler(handler)
    access_logger.setLevel(logging.INFO)
    access_logger.propagate = False


def main(argv: list[str] | None = None) -> int:
    """Run with UTF-8 standard streams; encoding and runtime settings persist."""
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper) and (
            stream.encoding.lower().replace("-", "") != "utf8"
            or stream.errors != "strict"
        ):
            stream.reconfigure(encoding="utf-8", errors="strict")
    import importlib.util

    if argv is None:
        argv = sys.argv[1:]
    if (
        argv
        and argv[0] in ("bench", "data")
        and importlib.util.find_spec(f"wyat.{argv[0]}.cli") is not None
    ):
        import importlib

        cli_module = importlib.import_module(f"wyat.{argv[0]}.cli")
        return cast(int, cli_module.main(argv[1:]))
    parser = argparse.ArgumentParser(
        prog="wyat",
        description="Download, load, evaluate and serve local decision models.",
        epilog=(
            "Use 'wyat bench --help' for benchmark and evaluation commands."
            if importlib.util.find_spec("wyat.bench") is not None
            else None
        ),
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
        command.add_argument(
            "model",
            help=f"local model directory, or a named release such as {REPOSITORY}",
        )
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
            "--prefix-cache-size",
            type=int,
            default=16,
            help="native prefix KV cache entries; 0 disables caching",
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
            "--responsive",
            action=argparse.BooleanOptionalAction,
            default=True,
            help=(
                "native MLX GPU: pause 1 ms after every fourth layer so other "
                "applications can submit GPU work; --no-responsive removes the pauses"
            ),
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
        command.add_argument(
            "--temperature",
            type=float,
            default=1.0,
            help="softmax temperature scaling (> 0)",
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
            command.add_argument(
                "--max-queue-size",
                type=int,
                default=16,
                help="maximum queued requests when busy; 0 rejects immediately",
            )
            command.add_argument(
                "--queue-timeout",
                type=float,
                default=30.0,
                help="maximum seconds a queued request waits before 503 (> 0)",
            )
            command.add_argument(
                "--access-log",
                dest="log_access",
                action=argparse.BooleanOptionalAction,
                default=True,
                help=(
                    "print one access log line per request to stderr with "
                    "X-Request-ID and latency; never logs request text or "
                    "client addresses (default: on)"
                ),
            )
    download_help = "Download and verify the pinned model release; print its path."
    download_command = subcommands.add_parser(
        "download",
        help=download_help,
        description=(
            download_help + " Rerunning resumes an interrupted download or "
            "re-verifies a complete one."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    download_command.add_argument(
        "directory",
        nargs="?",
        help=(
            "where to place the release (default: ~/.cache/wyat/<model>); "
            "its model/ subdirectory is printed"
        ),
    )
    download_command.add_argument(
        "--model",
        choices=tuple(RELEASES),
        default=REPOSITORY,
        help="named model release; the 800M default is not released yet",
    )
    args = Arguments()
    parser.parse_args(argv, namespace=args)
    if args.command == "download":
        try:
            model_path = download(args.directory, repository=args.model)
        except (DownloadError, OSError) as exc:
            print(f"wyat: {exc}", file=sys.stderr)
            return 1
        print(
            f'Next: wyat.load("{model_path}") in Python, '
            f"or 'wyat serve {model_path}'.",
            file=sys.stderr,
        )
        print(model_path)
        return 0
    if args.temperature <= 0 or not math.isfinite(args.temperature):
        parser.error("--temperature must be a positive finite number")
    if args.prefix_cache_size < 0:
        parser.error("--prefix-cache-size must be nonnegative")
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
        if args.max_queue_size < 0:
            parser.error("--max-queue-size must be nonnegative")
        if args.queue_timeout <= 0 or not math.isfinite(args.queue_timeout):
            parser.error("--queue-timeout must be a positive finite number")
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
        load_kwargs: dict[str, object] = {
            "backend": args.backend,
            "device": args.device,
            "max_length": args.max_length,
            "token_cache_size": args.token_cache_size,
            "attention": args.attention,
            "compile": args.compile,
            "temperature": args.temperature,
        }
        if args.prefix_cache_size != 16:
            load_kwargs["prefix_cache_size"] = args.prefix_cache_size
        if not args.responsive:
            load_kwargs["responsive"] = False
        model = load(
            args.model,
            **load_kwargs,  # type: ignore[arg-type]
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
                    sys.stdout.flush()
                    pending.clear()
            if pending:
                for result in _decide_batch(model, pending):
                    _write(result.to_dict())
                sys.stdout.flush()
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

            from .server import access_logger, create_app

            if args.log_access:
                _show_access_log(access_logger)
            uvicorn.run(
                create_app(
                    model,
                    max_body_bytes=args.max_body_bytes,
                    max_queue_size=args.max_queue_size,
                    queue_timeout=args.queue_timeout,
                    log_access=args.log_access,
                ),
                host=args.host,
                port=args.port,
                # Wyat's record replaces uvicorn's line, which adds the client address.
                access_log=False,
            )
    except BrokenPipeError:
        return 0
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f"wyat: {exc}", file=sys.stderr)
        return 1
    return 0
