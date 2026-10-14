from __future__ import annotations

import importlib
import io
import json
import math
import selectors
import subprocess
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import pytest

from wyat import cli
from wyat.core import Decision, Model, Request


@pytest.mark.parametrize(
    "command,phrases",
    [
        (
            "decide",
            ("request JSONL from stdin", "maximum rendered input tokens", "2048"),
        ),
        ("evaluate", ("labeled case JSONL from stdin", "evaluation report")),
        ("serve", ("HTTP", "1048576", "body bytes", "127.0.0.1", "8000")),
    ],
)
def test_command_help_explains_input_and_operational_defaults(
    capsys: pytest.CaptureFixture[str], command: str, phrases: tuple[str, ...]
) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main([command, "--help"])
    assert error.value.code == 0
    output = capsys.readouterr()
    help_text = " ".join(output.out.split())
    assert output.err == ""
    for phrase in phrases:
        assert phrase in help_text


def test_dependency_free_cli_help_and_missing_runtime_diagnostic() -> None:
    root = Path(__file__).resolve().parents[1]
    help_result = subprocess.run(
        [sys.executable, "-S", "-m", "wyat", "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert help_result.returncode == 0
    assert "Read request JSONL" in help_result.stdout
    assert help_result.stderr == ""
    missing = subprocess.run(
        [sys.executable, "-S", "-m", "wyat", "decide", "unused-local-artifact"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert missing.returncode == 1
    assert missing.stdout == ""
    assert "uv sync --extra inference" in missing.stderr
    assert "Traceback" not in missing.stderr


BODY = {
    "query": "Question",
    "candidates": [{"id": "a", "text": "First"}, {"id": "b", "text": "Second"}],
}


class Scorer:
    def __init__(self) -> None:
        self.calls: list[tuple[Request, ...]] = []

    def score_many(self, requests: Sequence[Request]) -> Sequence[Sequence[float]]:
        self.calls.append(tuple(requests))
        if any(request.query == "failure" for request in requests):
            raise RuntimeError("fixture failure")
        return [(1.0, 2.0) for _ in requests]


def install_model(monkeypatch: pytest.MonkeyPatch) -> Scorer:
    scorer = Scorer()
    monkeypatch.setattr(cli, "load", lambda *args, **kwargs: Model(scorer, "fixture"))
    return scorer


@contextmanager
def byte_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[io.BytesIO, io.BytesIO, io.BytesIO]]:
    buffers = (io.BytesIO(), io.BytesIO(), io.BytesIO())
    with (
        io.TextIOWrapper(buffers[0], encoding="cp1252", errors="replace") as stdin,
        io.TextIOWrapper(buffers[1], encoding="cp1252", errors="replace") as stdout,
        io.TextIOWrapper(buffers[2], encoding="cp1252", errors="replace") as stderr,
        monkeypatch.context() as patch,
    ):
        patch.setattr(sys, "stdin", stdin)
        patch.setattr(sys, "stdout", stdout)
        patch.setattr(sys, "stderr", stderr)
        yield buffers


@pytest.mark.parametrize("command", ["decide", "evaluate"])
def test_cli_uses_strict_utf8_bytes_for_multilingual_jsonl(
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    scorer = install_model(monkeypatch)
    candidates = [
        {"id": "é", "text": "First"},
        {"id": "e\u0301-日本-العربية-🧪", "text": "Second"},
    ]
    selected_id = candidates[1]["id"]
    body = {
        "query": "Choisir café / 選択 / اختر 👩🏽\u200d💻",
        "context": "é e\u0301 日本語 العربية",
        "candidates": candidates,
    }
    row = (
        {
            "id": "é-日本-العربية-🧪",
            "request": body,
            "expected": [selected_id],
        }
        if command == "evaluate"
        else body
    )
    with byte_streams(monkeypatch) as (stdin, stdout, stderr):
        stdin.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
        stdin.seek(0)
        streams = (sys.stdin, sys.stdout, sys.stderr)
        assert cli.main([command, "local-fixture"]) == 0
        sys.stdout.flush()
        sys.stderr.flush()
        assert (sys.stdin, sys.stdout, sys.stderr) == streams
        assert all(
            stream.encoding == "utf-8" and stream.errors == "strict"
            for stream in streams
        )
        assert not any(stream.closed for stream in streams)
        assert scorer.calls[0][0].query == body["query"]
        assert scorer.calls[0][0].context == body["context"]
        output = stdout.getvalue()
        assert selected_id.encode("utf-8") in output
        report = json.loads(output.decode("utf-8"))
        if command == "evaluate":
            assert report["rows"][0]["id"] == row["id"]
            assert report["accuracy"] == 1.0
            decision = report["rows"][0]["prediction"]
        else:
            decision = report
        assert decision["selected"] == selected_id
        assert stderr.getvalue() == b""


def test_invalid_utf8_input_is_rejected_without_replacement_or_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scorer = install_model(monkeypatch)
    with byte_streams(monkeypatch) as (stdin, stdout, stderr):
        stdin.write(json.dumps(BODY).encode("utf-8").replace(b"Question", b"caf\xe9"))
        stdin.seek(0)
        assert cli.main(["decide", "local-fixture"]) == 1
        sys.stdout.flush()
        sys.stderr.flush()
        assert stdout.getvalue() == b""
        diagnostic = stderr.getvalue().decode("utf-8")
        assert "utf-8" in diagnostic and "decode" in diagnostic
        assert "�" not in diagnostic
        assert scorer.calls == []


def test_import_does_not_reconfigure_standard_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with byte_streams(monkeypatch):
        streams = (sys.stdin, sys.stdout, sys.stderr)
        importlib.reload(cli)
        assert (sys.stdin, sys.stdout, sys.stderr) == streams
        assert all(
            stream.encoding == "cp1252" and stream.errors == "replace"
            for stream in streams
        )


def test_main_preserves_custom_injected_text_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CustomTextStream(io.StringIO):
        def reconfigure(self, **kwargs: object) -> None:
            raise AssertionError("custom streams own their encoding")

    install_model(monkeypatch)
    streams = tuple(CustomTextStream() for _ in range(3))
    for name, stream in zip(("stdin", "stdout", "stderr"), streams, strict=True):
        monkeypatch.setattr(sys, name, stream)
    assert cli.main(["decide", "local-fixture"]) == 0
    assert (sys.stdin, sys.stdout, sys.stderr) == streams


def test_decide_streams_jsonl_with_explicit_batches(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    scorer = install_model(monkeypatch)
    rows = [{**BODY, "query": f"Question {index}"} for index in range(3)]
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n".join(map(json.dumps, rows))))
    assert cli.main(["decide", "local-fixture", "--batch-size", "2"]) == 0
    output = capsys.readouterr()
    assert output.err == ""
    decisions = [json.loads(line) for line in output.out.splitlines()]
    assert len(decisions) == 3
    assert all(decision["selected"] == "b" for decision in decisions)
    assert [[request.query for request in batch] for batch in scorer.calls] == [
        ["Question 0", "Question 1"],
        ["Question 2"],
    ]


@pytest.mark.parametrize("batch_size,count", [(1, 0), (1, 3), (2, 4), (2, 3)])
def test_decide_only_calls_nonempty_batches_and_flushes_each_once(
    monkeypatch: pytest.MonkeyPatch, batch_size: int, count: int
) -> None:
    install_model(monkeypatch)
    calls: list[int] = []
    decide_many = Model.decide_many

    def observe(
        model: Model, requests: Sequence[Request], *, temperature: float | None = None
    ) -> tuple[Decision, ...]:
        calls.append(len(requests))
        return decide_many(model, requests, temperature=temperature)

    class Output(io.StringIO):
        def __init__(self) -> None:
            super().__init__()
            self.flushed_rows: list[int] = []

        def flush(self) -> None:
            self.flushed_rows.append(len(self.getvalue().splitlines()))
            super().flush()

    output = Output()
    monkeypatch.setattr(Model, "decide_many", observe)
    monkeypatch.setattr(sys, "stdin", io.StringIO((json.dumps(BODY) + "\n") * count))
    monkeypatch.setattr(sys, "stdout", output)
    args = ["decide", "local-fixture"]
    if batch_size != 1:
        args.extend(["--batch-size", str(batch_size)])
    assert cli.main(args) == 0
    expected_sizes = [batch_size] * (count // batch_size)
    if count % batch_size:
        expected_sizes.append(count % batch_size)
    assert calls == expected_sizes
    assert output.flushed_rows == [
        sum(expected_sizes[: i + 1]) for i in range(len(calls))
    ]
    assert len(output.getvalue().splitlines()) == count


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX subprocess pipe readiness")
@pytest.mark.parametrize("batch_size", [1, 2])
def test_decide_delivers_completed_batches_through_pipe_before_eof(
    batch_size: int,
) -> None:
    source = """
import sys
from wyat import cli
from wyat.core import Model
class Scorer:
    def score_many(self, requests):
        return [(1.0, 2.0) for _ in requests]
cli.load = lambda *args, **kwargs: Model(Scorer(), 'fixture')
raise SystemExit(cli.main(['decide', 'fixture', '--batch-size', sys.argv[1]]))
"""
    root = Path(__file__).resolve().parents[1]
    # -E ignores PYTHONUNBUFFERED: the child must exercise buffered pipe output.
    child = subprocess.Popen(
        [sys.executable, "-E", "-S", "-c", source, str(batch_size)],
        cwd=root,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert (
        child.stdin is not None
        and child.stdout is not None
        and child.stderr is not None
    )
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(child.stdout, selectors.EVENT_READ)
            for _ in range(2):
                child.stdin.write((json.dumps(BODY) + "\n").encode() * batch_size)
                child.stdin.flush()
                assert selector.select(timeout=5), (
                    "completed batch was buffered until EOF"
                )
                for _ in range(batch_size):
                    assert json.loads(child.stdout.readline())["selected"] == "b"
                assert child.poll() is None
        child.stdin.close()
        assert child.wait(timeout=5) == 0
        assert child.stdout.read() == b""
        assert child.stderr.read() == b""
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)
        child.stdin.close()
        child.stdout.close()
        child.stderr.close()


def test_broken_pipe_during_batch_flush_is_quiet_success(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    scorer = install_model(monkeypatch)

    class ClosedPipe(io.StringIO):
        def flush(self) -> None:
            raise BrokenPipeError("downstream closed pipe")

    monkeypatch.setattr(sys, "stdin", io.StringIO((json.dumps(BODY) + "\n") * 2))
    with monkeypatch.context() as patch:
        output = ClosedPipe()
        patch.setattr(sys, "stdout", output)
        assert cli.main(["decide", "local-fixture"]) == 0
        assert len(output.getvalue().splitlines()) == 1
    assert len(scorer.calls) == 1
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(
    "line", ["{", '{"query":NaN}', '{"query":1e999}', '{"query":"a","query":"b"}']
)
def test_malformed_json_writes_diagnostic_only(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], line: str
) -> None:
    scorer = install_model(monkeypatch)
    monkeypatch.setattr(sys, "stdin", io.StringIO(line))
    assert cli.main(["decide", "local-fixture"]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "line 1: invalid JSON" in output.err
    assert scorer.calls == []


def test_evaluate_retains_failure_in_report_and_exit_status(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    install_model(monkeypatch)
    cases = [
        {"id": "ok", "request": BODY, "expected": ["b"]},
        {"id": "failed", "request": {**BODY, "query": "failure"}, "expected": ["b"]},
    ]
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n".join(map(json.dumps, cases))))
    assert cli.main(["evaluate", "local-fixture"]) == 1
    output = capsys.readouterr()
    report = json.loads(output.out)
    assert output.err == ""
    assert report["total"] == 2
    assert report["accuracy"] == 0.5
    assert report["rows"][1]["status"] == "error"


def test_broken_pipe_is_quiet_success(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    install_model(monkeypatch)
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(BODY)))

    def closed_pipe(value: object) -> None:
        raise BrokenPipeError("downstream closed pipe")

    monkeypatch.setattr(cli, "_write", closed_pipe)
    assert cli.main(["decide", "local-fixture"]) == 0
    assert capsys.readouterr().err == ""


def install_cache_runtime(
    monkeypatch: pytest.MonkeyPatch, *, available: bool = True
) -> list[int]:
    """Fake the optional runtime so CLI ownership needs no native allocation."""
    limits: list[int] = []

    def set_cache_limit(limit: int) -> int:
        limits.append(limit)
        return 4096

    parent, core, metal = (ModuleType(name) for name in ("mlx", "mlx.core", "metal"))
    monkeypatch.setattr(core, "set_cache_limit", set_cache_limit, raising=False)
    monkeypatch.setattr(metal, "is_available", lambda: available, raising=False)
    monkeypatch.setattr(core, "metal", metal, raising=False)
    monkeypatch.setattr(parent, "core", core, raising=False)
    monkeypatch.setitem(sys.modules, "mlx", parent)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    return limits


@pytest.mark.parametrize("mib", [0, 512])
def test_metal_cache_control_converts_units_before_loading(
    monkeypatch: pytest.MonkeyPatch, mib: int
) -> None:
    limits = install_cache_runtime(monkeypatch)
    scorer = Scorer()

    def load_after_setting(*args: object, **kwargs: object) -> Model:
        assert limits == [mib * (1 << 20)]
        return Model(scorer, "fixture")

    monkeypatch.setattr(cli, "load", load_after_setting)
    monkeypatch.setattr(sys, "stdin", io.StringIO())
    assert (
        cli.main(
            ["decide", "local", "--backend", "mlx", "--mlx-cache-limit-mib", str(mib)]
        )
        == 0
    )


@pytest.mark.parametrize("backend,device", [("torch", "cpu"), ("mlx", "cpu")])
def test_cache_control_rejects_non_metal_engine_before_loading(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    backend: str,
    device: str,
) -> None:
    scorer = install_model(monkeypatch)
    limits = install_cache_runtime(monkeypatch)
    assert (
        cli.main(
            [
                "decide",
                "local",
                "--backend",
                backend,
                "--device",
                device,
                "--mlx-cache-limit-mib",
                "512",
            ]
        )
        == 1
    )
    assert "requires a Metal MLX engine" in capsys.readouterr().err
    assert limits == [] and scorer.calls == []


def test_cache_control_requires_available_metal(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    install_model(monkeypatch)
    limits = install_cache_runtime(monkeypatch, available=False)
    assert (
        cli.main(
            ["decide", "local", "--backend", "mlx", "--mlx-cache-limit-mib", "512"]
        )
        == 1
    )
    assert "available Apple Metal" in capsys.readouterr().err
    assert limits == []


@pytest.mark.parametrize("mib", ["-1", str((sys.maxsize >> 20) + 1)])
def test_cache_control_rejects_out_of_range_before_runtime_import(
    monkeypatch: pytest.MonkeyPatch, mib: str
) -> None:
    monkeypatch.setitem(sys.modules, "mlx", None)
    with pytest.raises(SystemExit) as error:
        cli.main(["decide", "local", "--mlx-cache-limit-mib", mib])
    assert error.value.code == 2


def test_default_cli_does_not_import_mlx_or_change_cache_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_model(monkeypatch)
    monkeypatch.setitem(sys.modules, "mlx", None)
    monkeypatch.setattr(sys, "stdin", io.StringIO())
    assert cli.main(["decide", "local"]) == 0


@pytest.mark.parametrize(
    "backend,option,extra",
    [
        ("torch", "--torch-threads", "inference"),
        ("mlx", "--mlx-cache-limit-mib", "mlx"),
    ],
)
def test_explicit_runtime_controls_explain_missing_primary_dependency(
    backend: str,
    option: str,
    extra: str,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-m",
            "wyat",
            "decide",
            "unused-local-artifact",
            "--backend",
            backend,
            option,
            "8",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 1
    assert result.stdout == ""
    assert f"uv sync --extra {extra}" in result.stderr
    assert "Traceback" not in result.stderr


def test_help_preserves_already_read_strict_utf8_stdin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with io.TextIOWrapper(io.BytesIO(b"already read"), encoding="utf-8") as stdin:
        assert stdin.read(1) == "a"
        monkeypatch.setattr(sys, "stdin", stdin)
        monkeypatch.setattr(sys, "stdout", io.StringIO())
        monkeypatch.setattr(sys, "stderr", io.StringIO())
        with pytest.raises(SystemExit) as error:
            cli.main(["--help"])
        assert error.value.code == 0
        assert stdin.read() == "lready read"


@pytest.mark.parametrize(
    "invalid", ["0", "-1.5", "-0.01", "nan", "inf", "-inf", "not-a-number"]
)
def test_cli_rejects_invalid_temperature(
    capsys: pytest.CaptureFixture[str], invalid: str
) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(["decide", "fixture", "--temperature", invalid])
    assert error.value.code == 2
    output = capsys.readouterr()
    assert "--temperature" in output.err


def test_cli_decide_passes_temperature(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class FixedScorer:
        def score_many(self, requests: Sequence[Request]) -> Sequence[Sequence[float]]:
            return [(2.0, 0.0) for _ in requests]

    recorded_temp: list[float] = []

    def mock_load(path: str, **kwargs: object) -> Model:
        temp = float(kwargs.get("temperature", 1.0))  # type: ignore[arg-type]
        recorded_temp.append(temp)
        return Model(FixedScorer(), "fixture", temperature=temp)

    monkeypatch.setattr(cli, "load", mock_load)
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(BODY) + "\n"))

    result = cli.main(["decide", "local-path", "--temperature", "2.0"])
    assert result == 0
    assert recorded_temp == [2.0]
    out = capsys.readouterr().out
    decision = json.loads(out)
    expected_p0 = math.exp(1.0) / (math.exp(1.0) + math.exp(0.0))
    assert decision["probabilities"][0] == pytest.approx(expected_p0)


def test_cli_forwards_prefix_cache_size_and_serve_queue_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_kwargs: list[dict[str, object]] = []
    app_kwargs: list[dict[str, object]] = []

    def mock_load(path: str, **kwargs: object) -> Model:
        load_kwargs.append(dict(kwargs))
        return Model(Scorer(), "fixture")

    monkeypatch.setattr(cli, "load", mock_load)

    server_mod = ModuleType("wyat.server")

    def fake_create_app(model: Model, **kwargs: object) -> object:
        app_kwargs.append(dict(kwargs))
        return "fake-app"

    monkeypatch.setattr(server_mod, "create_app", fake_create_app, raising=False)
    monkeypatch.setitem(sys.modules, "wyat.server", server_mod)

    uvicorn_mod = ModuleType("uvicorn")
    monkeypatch.setattr(uvicorn_mod, "run", lambda app, **kwargs: None, raising=False)
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn_mod)

    assert (
        cli.main(
            [
                "serve",
                "local-path",
                "--prefix-cache-size",
                "24",
                "--max-queue-size",
                "12",
                "--queue-timeout",
                "15.5",
            ]
        )
        == 0
    )
    assert load_kwargs[0]["prefix_cache_size"] == 24
    assert app_kwargs[0] == {
        "max_body_bytes": 1_048_576,
        "max_queue_size": 12,
        "queue_timeout": 15.5,
        "log_access": True,
    }

    assert cli.main(["serve", "local-path", "--no-access-log"]) == 0
    assert app_kwargs[1]["log_access"] is False

    for bad_args in (
        ["decide", "local", "--prefix-cache-size", "-1"],
        ["serve", "local", "--max-queue-size", "-1"],
        ["serve", "local", "--queue-timeout", "0"],
    ):
        with pytest.raises(SystemExit) as error:
            cli.main(bad_args)
        assert error.value.code == 2
