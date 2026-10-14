from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

pytest.importorskip("mlx.core", exc_type=ImportError)
pytest.importorskip("mlx_lm", exc_type=ImportError)

import mlx.core as mx

from benchmarks.analysis.mlx import cache
from wyat.benchmark import BenchmarkCycle
from wyat.core import Request
from wyat.wire import read_object


@given(st.integers(1, 20))
def test_two_alternating_comparisons_balance_both_arm_positions(repeats: int) -> None:
    first = cache.paired_orders(repeats, reverse=False)
    second = cache.paired_orders(repeats, reverse=True)
    assert len(first) == len(second) == repeats
    for order in first + second:
        assert set(order) == {"default", "capped"}
    assert sum(order[0] == "default" for order in first + second) == repeats
    assert all(left != right for left, right in zip(first, first[1:], strict=False))


@pytest.mark.parametrize("phase", ["load", "sample"])
def test_original_allocator_cap_restored_after_failure(
    phase: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current_cap = [2048]
    setters: list[int] = []

    def set_cap(value: int) -> int:
        previous = current_cap[0]
        current_cap[0] = value
        setters.append(value)
        return previous

    def memory() -> dict[str, int]:
        return {"active": 10, "cache": 20, "active_plus_cache": 30, "peak_active": 10}

    class Scorer:
        model_id = "fake"

        @classmethod
        def from_directory(cls, *args: object, **kwargs: object) -> Scorer:
            if phase == "load":
                raise RuntimeError("model load failure")
            return cls()

        def score_many(self, requests: Sequence[Request]) -> list[list[float]]:
            raise AssertionError("No actual scoring is authorized in this test")

    def reference(*args: object, **kwargs: object) -> BenchmarkCycle:
        return BenchmarkCycle(0, (), 0, 0)

    def sample(*args: object, **kwargs: object) -> None:
        set_cap(100)
        raise RuntimeError("sample failure")

    monkeypatch.setattr(mx, "set_cache_limit", set_cap)
    monkeypatch.setattr(cache, "memory", memory)
    monkeypatch.setattr(cache, "MLXWyatScorer", Scorer)
    monkeypatch.setattr(cache, "read_reference", reference)
    monkeypatch.setattr(cache, "sample", sample)
    output = tmp_path / "failure.json"
    with pytest.raises(
        RuntimeError, match=f"{'model load' if phase == 'load' else phase} failure"
    ):
        cache.study(tmp_path, output)
    assert current_cap[0] == 2048
    assert setters[-1] == 2048
    receipt = read_object(output)
    assert receipt["passed"] is False and receipt["completed"] is False
    assert receipt["restored_cache_cap_bytes"] == 2048
    assert "failure" in str(receipt["error"])
    assert isinstance(receipt["source_archive"], dict)


def test_failed_numerical_gates_produce_nonzero_cli_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failed(*args: object, **kwargs: object) -> dict[str, object]:
        return {"passed": False, "completed": True}

    monkeypatch.setattr(cache, "study", failed)
    monkeypatch.setattr("sys.argv", ["cache-study", "model-directory"])
    with pytest.raises(SystemExit) as error:
        cache.main()
    assert error.value.code == 1
