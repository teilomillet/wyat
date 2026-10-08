from __future__ import annotations

import importlib
import io
import json
import subprocess
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import pytest

from wyat import cli
from wyat.core import Model, Request


@pytest.mark.parametrize(
    "command,phrases",
    [
        (
            "decide",
            ("request JSONL from stdin", "maximum rendered input tokens", "2048"),
        ),
        ("evaluate", ("labeled case JSONL from stdin", "evaluation report")),
        ("serve", ("HTTP", "1048576", "body bytes", "127.0.0.1", "8000")),
        (
            "benchmark",
            (
                "labeled case JSONL from stdin",
                "exit status 0 indicate quality",
                "not a guaranteed speed improvement",
                "CPU for both runtimes",
                "measured alternating pairs",
                "minimum 3",
            ),
        ),
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


def test_benchmark_loads_two_engines_and_keeps_all_paired_rows(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    scorer = install_model(monkeypatch)
    rows = [
        {"id": f"case-{index}", "request": BODY, "expected": ["b"]}
        for index in range(3)
    ]
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n".join(map(json.dumps, rows))))
    assert cli.main(["benchmark", "local", "--batch-size", "2", "--repeats", "3"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] is True
    assert report["quality"]["total"] == 9
    assert report["baseline"]["batch_size"] == 1
    assert report["candidate"]["batch_size"] == 2
    assert len(scorer.calls) == 20  # 4 cycles, each containing 3 + 2 batches


def test_benchmark_quality_failure_returns_nonzero_with_receipt(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    install_model(monkeypatch)
    row = {"id": "failed", "request": {**BODY, "query": "failure"}, "expected": ["b"]}
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(row)))
    assert cli.main(["benchmark", "local", "--repeats", "3"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] is False
    assert report["baseline"]["failed"] == 3
    assert report["candidate"]["failed"] == 3


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


def test_cache_setting_is_shared_and_recorded_for_benchmark_baseline(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    install_model(monkeypatch)
    limits = install_cache_runtime(monkeypatch)
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(json.dumps({"id": "case", "request": BODY, "expected": ["b"]})),
    )
    assert (
        cli.main(
            [
                "benchmark",
                "local",
                "--baseline-backend",
                "mlx",
                "--baseline-device",
                "gpu",
                "--mlx-cache-limit-mib",
                "512",
                "--repeats",
                "3",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert limits == [512 * (1 << 20)]
    for engine in ("baseline", "candidate"):
        assert report[engine]["conditions"]["mlx_cache_limit_mib"] == "512"


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
