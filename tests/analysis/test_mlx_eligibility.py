from __future__ import annotations

import json
import math
import struct
import subprocess
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from benchmarks.analysis.mlx.eligibility import (
    TensorEligibility,
    TensorSpan,
    arm_results,
    declared_arms,
    launch,
    projection_spans,
    scan_tensor,
    study,
)


def write_bits(path: Path, bits: list[int]) -> TensorSpan:
    path.write_bytes(struct.pack(f"<{len(bits)}H", *bits))
    return TensorSpan("probe", (1, len(bits)), 0, len(bits))


def test_exact_range_and_bf16_exponent_counterexamples(tmp_path: Path) -> None:
    # ±1, signed zeros and the smallest half subnormal are exact. BF16 has
    # smaller subnormals and larger finite values that half cannot preserve.
    bits = [0x3F80, 0xBF80, 0, 0x8000, 0x3380, 0x0001, 0x7F7F]
    path = tmp_path / "bits"
    result = scan_tensor(path, write_bits(path, bits), chunk_elements=2)
    assert result.elements == 7
    assert result.source_nonfinite == 0
    assert result.mismatches == 2
    assert result.underflow_to_zero == 1
    assert result.source_subnormal == 1
    assert result.overflow == 1
    assert not result.eligible


def test_half_subnormal_grid_rounding_is_not_zero_underflow(tmp_path: Path) -> None:
    path = tmp_path / "bits"
    # 2**-24 + 2**-31 is BF16-representable but not a half subnormal grid point.
    result = scan_tensor(path, write_bits(path, [0x3381]))
    assert result.mismatches == 1
    assert result.underflow_to_zero == result.overflow == 0
    assert result.max_finite_absolute_error == 2**-31


def test_nonfinite_source_fails_even_if_infinity_roundtrips(tmp_path: Path) -> None:
    path = tmp_path / "bits"
    result = scan_tensor(path, write_bits(path, [0x7F80, 0xFF80, 0x7FC1]))
    assert result.source_nonfinite == 3
    assert result.overflow == 0
    assert not result.eligible


@given(st.lists(st.integers(0, 65535), min_size=1, max_size=64))
def test_chunked_scan_matches_independent_python_half_oracle(bits: list[int]) -> None:
    # NamedTemporaryFile avoids Hypothesis retaining a pytest tmp_path fixture.
    from tempfile import TemporaryDirectory

    expected_mismatch = expected_nonfinite = 0
    for word in bits:
        value = struct.unpack("<f", struct.pack("<I", word << 16))[0]
        expected_nonfinite += not math.isfinite(value)
        try:
            restored = struct.unpack("<e", struct.pack("<e", value))[0]
        except OverflowError:
            restored = math.copysign(math.inf, value)
        expected_mismatch += value != restored
    with TemporaryDirectory() as directory:
        path = Path(directory) / "bits"
        span = write_bits(path, bits)
        result = scan_tensor(path, span, chunk_elements=3)
        assert result.mismatches == expected_mismatch
        assert result.source_nonfinite == expected_nonfinite
        assert result.eligible == (expected_mismatch == expected_nonfinite == 0)


def fixture_file(path: Path, *, change: str = "") -> None:
    header: dict[str, object] = {}
    for index, role in enumerate(("gate_proj", "up_proj", "down_proj")):
        shape = [2, 1] if role != "down_proj" else [1, 2]
        header[f"layers.0.mlp.{role}.weight"] = {
            "dtype": "BF16",
            "shape": shape,
            "data_offsets": [4 * index, 4 * index + 4],
        }
    first = header["layers.0.mlp.gate_proj.weight"]
    assert isinstance(first, dict)
    if change == "dtype":
        first["dtype"] = "F16"
    elif change == "shape":
        first["shape"] = [True, 2]
    elif change == "outside":
        first["data_offsets"] = [100, 104]
    elif change == "overlap":
        first["data_offsets"] = [4, 8]
    elif change == "short":
        first["data_offsets"] = [0, 2]
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(12))


def test_only_declared_mlp_projections_and_shapes_are_selected(tmp_path: Path) -> None:
    path = tmp_path / "fixture"
    fixture_file(path)
    spans = projection_spans(path, layers=1, hidden=1, intermediate=2)
    assert len(spans) == 3 and sum(item.elements for item in spans) == 6
    assert spans[-1].shape == (1, 2)
    assert all(scan_tensor(path, span).eligible for span in spans)


@pytest.mark.parametrize("change", ["dtype", "shape", "outside", "overlap", "short"])
def test_malformed_selected_projection_fails_closed(
    tmp_path: Path, change: str
) -> None:
    path = tmp_path / "fixture"
    fixture_file(path, change=change)
    with pytest.raises(ValueError):
        projection_spans(path, layers=1, hidden=1, intermediate=2)


def test_truncated_stream_never_counts_as_exact(tmp_path: Path) -> None:
    path = tmp_path / "bits"
    path.write_bytes(bytes(2))
    with pytest.raises(ValueError, match="Truncated"):
        scan_tensor(path, TensorSpan("probe", (1, 2), 0, 2))


def test_arm_eligibility_uses_exact_role_and_layer_intersections() -> None:
    arms = declared_arms(8)
    tensors = tuple(
        TensorEligibility(
            name,
            (1, 1),
            1,
            0,
            int(name == "layers.0.mlp.gate_proj.weight"),
            0,
            0,
            0,
            0.0,
        )
        for name in arms[2].names()
    )
    results = arm_results(arms, tensors)
    assert [item["eligible"] for item in results] == [
        True,
        False,
        False,
        False,
        True,
        True,
    ]
    assert results[-1]["diagnostic_only"] is True
    with pytest.raises(KeyError):
        arm_results(arms, tensors[:-1])
    with pytest.raises(ValueError, match="Duplicate"):
        arm_results(arms, (*tensors, tensors[0]))


def test_parent_timeout_preserves_partial_child_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "result.json"

    def timeout(command: list[str], *, check: bool, timeout: int) -> None:
        output.write_text('{"complete": false}\n')
        raise subprocess.TimeoutExpired(command, timeout)

    monkeypatch.setattr(subprocess, "run", timeout)
    assert launch(tmp_path / "model", output) == 124
    wall = json.loads(output.with_suffix(".wall.json").read_text())
    assert wall["timed_out"] is True and wall["completed"] is False
    assert json.loads(output.read_text())["complete"] is False
    with pytest.raises(FileExistsError):
        launch(tmp_path / "model", output)


def test_unexpected_artifact_failure_cannot_become_pass(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    assert study(tmp_path / "absent-model", output) == 1
    result = json.loads(output.read_text())
    assert result["complete"] is False and result["any_eligible"] is False
    assert result["error"].startswith("FileNotFoundError")
