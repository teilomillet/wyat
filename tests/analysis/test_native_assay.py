"""Pure report checks preserve capability failures and complete denominators."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import NoReturn, cast

import pytest

from benchmarks.real.dmbench import native
from benchmarks.real.dmbench.native import (
    RELEASE,
    UNSUPPORTED,
    check_reports,
    object_value,
    read_assay,
)


def reports() -> tuple[dict[str, object], dict[str, tuple[str, ...]]]:
    choices: dict[str, tuple[str, ...]] = {
        identifier: ("a", "b")
        for identifier in (*UNSUPPORTED, *(f"authored-{i}" for i in range(24)))
    }
    rows: list[dict[str, object]] = []
    for identifier in choices:
        failed = identifier in UNSUPPORTED
        rows.append(
            {
                "id": identifier,
                "answer": None if failed else "a",
                "error": (
                    "ValueError: 2683 tokens exceeds max_length=2048; no truncation"
                )
                if failed
                else None,
                "probabilities": None if failed else {"a": 0.6, "b": 0.4},
            }
        )
    return {"count": 30, "predictions": rows}, choices


def row(report: dict[str, object], index: int) -> dict[str, object]:
    return object_value(cast(list[object], report["predictions"])[index])


def test_expected_unsupported_is_a_supported_assay_outcome() -> None:
    report, choices = reports()
    checks = check_reports(report, deepcopy(report), choices)
    assert checks["passed"] is True
    assert checks["total"] == 30
    assert checks["expected_unsupported"] == 6
    assert checks["admitted"] == checks["selected_equal"] == 24


@pytest.mark.parametrize(
    "change", ["status", "selection", "probabilities", "capability"]
)
def test_drift_or_unexpected_failure_rejects_the_assay(change: str) -> None:
    report, choices = reports()
    candidate = deepcopy(report)
    if change == "status":
        row(candidate, 6)["error"] = "RuntimeError: GPU unavailable"
    elif change == "selection":
        row(candidate, 6)["answer"] = "b"
    elif change == "probabilities":
        row(candidate, 6)["probabilities"] = {"a": 0.6002, "b": 0.3998}
    else:
        row(candidate, 0)["error"] = "RuntimeError: unexpected allocator failure"
    assert check_reports(report, candidate, choices)["passed"] is False


@pytest.mark.parametrize("change", ["count", "missing", "order", "nonfinite"])
def test_missing_or_invalid_rows_cannot_escape_the_denominator(change: str) -> None:
    report, choices = reports()
    candidate = deepcopy(report)
    if change == "count":
        candidate["count"] = 24
    elif change == "missing":
        cast(list[object], candidate["predictions"]).pop()
    elif change == "order":
        cast(list[object], candidate["predictions"]).reverse()
    else:
        row(candidate, 6)["probabilities"] = {"a": float("nan"), "b": 0.4}
    with pytest.raises(ValueError):
        check_reports(report, candidate, choices)


def test_actual_peer_loader_preserves_the_frozen_assay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    peer = Path("/Users/teilomillet/Code/dmbench")
    if not (peer / "reports/wyat-native-dev-assay-contract.json").is_file():
        pytest.skip("Peer's predeclared assay is unavailable")
    monkeypatch.syspath_prepend(str(peer / "src"))
    contract, cases = read_assay(peer)
    assert object_value(contract["manifest"])["release_id"] == RELEASE
    assert len(cases) == 30
    assert tuple(case.request.id for case in cases[:6]) == UNSUPPORTED
    assert [len(case.request.choices) for case in cases[:6]] == [151] * 6
    assert sorted({len(case.request.choices) for case in cases[6:]}) == [3, 6]


def test_unexpected_preflight_failure_retains_an_incomplete_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    peer = Path("/Users/teilomillet/Code/dmbench")
    if not (peer / "reports/wyat-native-dev-assay-contract.json").is_file():
        pytest.skip("Peer's predeclared assay is unavailable")
    monkeypatch.syspath_prepend(str(peer / "src"))

    def fail_read(path: Path) -> NoReturn:
        raise RuntimeError("injected preflight failure")

    monkeypatch.setattr(native, "read_assay", fail_read)
    output = tmp_path / "receipt.json"
    result = native.assay(tmp_path / "unused-model", peer, output)
    assert result["completed"] is result["passed"] is False
    assert result["error"] == "RuntimeError: injected preflight failure"
    assert object_value(result["source_sha256"])
    assert output.is_file()
