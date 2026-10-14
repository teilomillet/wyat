# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Exercise user examples with local seeded weights, not learned-quality claims."""

from __future__ import annotations

import json
import math
import runpy
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Literal, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fixtures.wyat import create_wyat_bundle
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import wyat
from wyat.wire import decode_json, read_object

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def factory() -> Callable[[], FastAPI]:
    values = runpy.run_path(str(EXAMPLES / "server" / "main.py"))
    return cast(Callable[[], FastAPI], values["app"])


@pytest.mark.parametrize("name", ["quickstart", "server"])
def test_import_does_not_load_weights(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_load(*args: object, **kwargs: object) -> wyat.Model:
        raise AssertionError("Importing an example must not load a model")

    monkeypatch.setattr(wyat, "load", unexpected_load)
    monkeypatch.delenv("WYAT_MODEL", raising=False)
    runpy.run_path(str(EXAMPLES / name / "main.py"))


def test_cpu_model_and_server_examples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle = create_wyat_bundle(tmp_path / "wyat")
    # This rotary-position fixture needs a cap that admits the public load default.
    config_path = bundle / "backbone" / "config.json"
    config = read_object(config_path)
    config["max_position_embeddings"] = 2048
    config_path.write_text(json.dumps(config), encoding="utf-8")
    calls: list[tuple[str | Path, str]] = []
    original_load = wyat.load

    def recorded_load(
        path: str | Path, *, backend: Literal["torch", "mlx"]
    ) -> wyat.Model:
        calls.append((path, backend))
        return original_load(path, backend=backend)

    monkeypatch.setattr(wyat, "load", recorded_load)
    request = wyat.Request(
        query="Choose the requested action",
        context="Cancel my booking",
        candidates=(
            wyat.Candidate(id="cancel", text="Cancel the booking"),
            wyat.Candidate(id="create", text="Create a booking"),
        ),
    )
    decision = decode_json(
        json.dumps(wyat.load(bundle, backend="torch").decide(request).to_dict())
    )
    assert isinstance(decision, dict)
    ranking = decision["ranking"]
    assert ranking in [["cancel", "create"], ["create", "cancel"]]
    assert isinstance(ranking, list)
    scores = decision["scores"]
    assert isinstance(scores, list) and len(scores) == 2
    for score in scores:
        assert isinstance(score, float) and math.isfinite(score)
    probabilities = decision["probabilities"]
    assert isinstance(probabilities, list)
    assert all(isinstance(value, float) for value in probabilities)
    assert math.fsum(cast(list[float], probabilities)) == pytest.approx(1.0)
    assert decision["selected"] == ranking[0]
    assert calls == [(bundle, "torch")]

    monkeypatch.setattr(sys, "argv", ["main.py", "--model", str(bundle)])
    runpy.run_path(str(EXAMPLES / "quickstart" / "main.py"), run_name="__main__")
    output = capsys.readouterr().out
    assert "Message: Please cancel my booking." in output
    assert output.splitlines()[-1] in {"Selected: create", "Selected: cancel"}
    assert "Demo mode" not in output
    assert calls == [(bundle, "torch"), (bundle, "torch")]

    monkeypatch.setenv("WYAT_MODEL", str(bundle))
    monkeypatch.delenv("WYAT_BACKEND", raising=False)
    with TestClient(factory()()) as client:
        assert client.get("/health").json()["model_id"] == decision["model_id"]
        for _ in range(2):
            response = client.post(
                "/v1/decide",
                json={
                    "query": "Choose the requested action",
                    "context": "Cancel my booking",
                    "candidates": [
                        {"id": "cancel", "text": "Cancel the booking"},
                        {"id": "create", "text": "Create a booking"},
                    ],
                },
            )
            assert response.status_code == 200
            assert response.json() == decision
    assert calls == [(bundle, "torch"), (bundle, "torch"), (str(bundle), "torch")]


def test_quickstart_without_weights(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def unexpected_load(*args: object, **kwargs: object) -> wyat.Model:
        raise AssertionError("Demo mode must not load weights")

    monkeypatch.setattr(wyat, "load", unexpected_load)
    monkeypatch.setattr(sys, "argv", ["main.py", "--demo"])
    runpy.run_path(str(EXAMPLES / "quickstart" / "main.py"), run_name="__main__")
    assert capsys.readouterr().out == (
        "Demo mode: fixed scores, no trained model.\n"
        "Message: Please cancel my booking.\n"
        "Choices:\n"
        "  create: Create a booking\n"
        "  cancel: Cancel the booking\n"
        "Selected: cancel\n"
    )


@pytest.mark.parametrize("path", [None, "", "   "])
def test_server_requires_model_path(
    path: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    if path is None:
        monkeypatch.delenv("WYAT_MODEL", raising=False)
    else:
        monkeypatch.setenv("WYAT_MODEL", path)
    with pytest.raises(ValueError, match="Set WYAT_MODEL"):
        factory()()


@given(
    backend=st.text(
        alphabet=st.characters(
            blacklist_categories=["Cs"], blacklist_characters="\x00"
        ),
        max_size=48,
    ).filter(lambda value: value not in {"torch", "mlx"})
)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_server_rejects_unknown_backend_before_loading(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_load(*args: object, **kwargs: object) -> wyat.Model:
        raise AssertionError("Invalid backend must fail before loading")

    monkeypatch.setattr(wyat, "load", unexpected_load)
    monkeypatch.setenv("WYAT_MODEL", "/nonexistent/local/model")
    monkeypatch.setenv("WYAT_BACKEND", backend)
    with pytest.raises(ValueError, match="WYAT_BACKEND must be torch or mlx"):
        factory()()
