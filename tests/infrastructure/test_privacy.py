# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Publication scans reject disclosure controls without revealing their values."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import pytest

from scripts.package.privacy import check_files, check_text


def selected(tmp_path: Path, name: str, raw: bytes) -> dict[str, Path]:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {name: path}


@pytest.mark.parametrize(
    "secret",
    [
        "gh" + "p_" + "a" * 36,
        "github" + "_pat_" + "b" * 40,
        "h" + "f_" + "c" * 35,
        "AK" + "IA" + "A" * 16,
        "sk" + "-proj-" + "d" * 30,
        "xo" + "xb-" + "1234567890-" * 3,
        "ey" + "J" + "a" * 20 + "." + "b" * 20 + "." + "c" * 20,
        "Bea" + "rer " + "a" * 24,
        "https://account:" + "secret-value" + "@service.invalid",
        "-----BEGIN " + "PRIVATE KEY-----",
        "api_" + 'key = "' + "non-placeholder-key" + '"',
        "pass" + 'word = "' + "correct horse ! staple" + '"',
    ],
)
def test_credentials_fail_without_echoing_content(secret: str) -> None:
    with pytest.raises(ValueError) as failure:
        check_text("README.md", ("public heading\n" + secret).encode())
    assert secret not in str(failure.value)
    assert "README.md:2" in str(failure.value)
    assert "credential" in str(failure.value)


@pytest.mark.parametrize(
    "path",
    [
        "/" + "Users/alice/private.txt",
        "/" + "home/alice/private.txt",
        "/" + "root/private.txt",
        "C:" + "\\Users\\alice\\private.txt",
        "artifacts/" + "run-20260101/private.json",
    ],
)
def test_private_paths_are_redacted(path: str) -> None:
    with pytest.raises(ValueError) as failure:
        check_text("verification/BUILD.json", path.encode())
    assert path not in str(failure.value)
    assert "path" in str(failure.value)


@pytest.mark.parametrize(
    "text",
    [
        "Copyright Teïlo Millet - teilomillet",
        "Contact public-author@example.com",
        "/absolute/path/to/model",
        "/" + "home/user/model",
        "api_" + 'key = "your_api_key"',
        "pass" + 'word = "<your-password>"',
        "Bea" + "rer YOUR_TOKEN",
        "https://username:your_password@service.invalid",
        r"re.compile(r'/[A-Za-z]+/[a-z]+')",
    ],
)
def test_public_attribution_placeholders_and_patterns_are_allowed(text: str) -> None:
    check_text("README.md", text.encode())


def test_binary_source_is_not_silently_skipped(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-text"):
        check_files(selected(tmp_path, "README.md", b"\xff\xfe"))


@pytest.mark.parametrize(
    "name",
    [
        "training.csv",
        "predictions.json",
        "credentials.json",
        ".cache/session.json",
        "verification/screenshot.png",
        "model/training.pt",
        "runtime/arbitrary.py",
    ],
)
def test_model_artifact_names_require_explicit_admission(
    tmp_path: Path, name: str
) -> None:
    with pytest.raises(ValueError, match="unadmitted model artifact"):
        check_files(selected(tmp_path, name, b"{}"), profile="model")


@pytest.mark.parametrize(
    "field",
    [
        "per_class_f1",
        "gold_label_share",
        "pred_label_share",
        "over_picked",
        "under_picked",
        "by_gold_category",
        "gold_slot_share",
        "by_length_bucket",
        "row_ids_sha256",
        "question_keys_sha256",
        "candidate",
        "raw_predictions",
    ],
)
def test_benchmark_fields_cannot_disguise_records_as_aggregates(
    tmp_path: Path, field: str
) -> None:
    raw = json.dumps({"data": [{"nested": {field: {"private answer": 1}}}]}).encode()
    with pytest.raises(ValueError, match="record field") as failure:
        check_files(
            selected(tmp_path, "benchmarks/comparison.json", raw), profile="model"
        )
    assert "private answer" not in str(failure.value)


def test_json_dictionary_keys_receive_content_scan(tmp_path: Path) -> None:
    raw = json.dumps({"h" + "f_" + "A" * 35: "summary"}).encode()
    with pytest.raises(ValueError, match="credential"):
        check_files(
            selected(tmp_path, "benchmarks/comparison.json", raw), profile="model"
        )


def test_json_escaping_does_not_hide_a_credential(tmp_path: Path) -> None:
    raw = ('{"note":"h' + r"\u0066_" + "A" * 35 + '"}').encode()
    with pytest.raises(ValueError, match="credential"):
        check_files(
            selected(tmp_path, "benchmarks/comparison.json", raw), profile="model"
        )


@pytest.mark.parametrize("suffix", [".json", ".jsonl"])
@pytest.mark.parametrize("as_key", [False, True])
@pytest.mark.parametrize("unicode_escape", [False, True])
def test_decoded_windows_paths_fail_in_json_keys_and_values(
    tmp_path: Path, suffix: str, as_key: bool, unicode_escape: bool
) -> None:
    private_path = "C:" + "\\Users\\alice\\private.txt"
    content = {private_path: "summary"} if as_key else {"path": private_path}
    raw = json.dumps({"nested": [content]}).encode()
    if unicode_escape:
        raw = raw.replace(b"Users", b"\\u0055sers")
    name = "examples/fixture" + suffix
    with pytest.raises(ValueError, match="private filesystem path") as failure:
        check_files(selected(tmp_path, name, raw))
    assert private_path not in str(failure.value)


@pytest.mark.parametrize("as_key", [False, True])
def test_decoded_unicode_windows_paths_fail_in_model_receipts(
    tmp_path: Path, as_key: bool
) -> None:
    private_path = "C:" + "\\Users\\alice\\private.txt"
    content = {private_path: "summary"} if as_key else {"path": private_path}
    raw = json.dumps(content).encode().replace(b"Users", b"\\u0055sers")
    with pytest.raises(ValueError, match="private filesystem path"):
        check_files(selected(tmp_path, "verification/BUILD.json", raw), profile="model")


@pytest.mark.parametrize(
    "count_key", ["n", "rows", "questions", "n_questions", "n_requests"]
)
def test_small_aggregate_cells_require_count_only_suppression(
    tmp_path: Path, count_key: str
) -> None:
    raw = json.dumps({"data": {count_key: 1, "accuracy": 1.0}}).encode()
    with pytest.raises(ValueError, match="small aggregate cell"):
        check_files(
            selected(tmp_path, "benchmarks/comparison.json", raw), profile="model"
        )
    raw = json.dumps({"data": {count_key: 1, "suppressed": True}}).encode()
    check_files(selected(tmp_path, "benchmarks/comparison.json", raw), profile="model")


@pytest.mark.parametrize("suffix", [".json", ".jsonl"])
def test_standalone_json_scan_rejects_duplicate_keys_before_value_loss(
    suffix: str,
) -> None:
    raw = ('{"duplicate":"' + r"\u0068f_" + "A" * 32 + '","duplicate":"ok"}').encode()
    with pytest.raises(ValueError, match="invalid or ambiguous JSON"):
        check_text("coverage" + suffix, raw)


@pytest.mark.parametrize("suffix", [".json", ".jsonl"])
def test_standalone_json_scan_rejects_truncated_escaped_content(suffix: str) -> None:
    raw = ('{"note":"' + r"\u0068f_" + "A" * 32).encode()
    with pytest.raises(ValueError, match="invalid or ambiguous JSON"):
        check_text("coverage" + suffix, raw)


def test_blank_jsonl_records_are_not_silently_skipped() -> None:
    with pytest.raises(ValueError, match="invalid or ambiguous JSON"):
        check_text("examples/fixture.jsonl", b"{}\n\n{}\n")


def test_empty_jsonl_collection_is_permitted() -> None:
    check_text("examples/fixture.jsonl", b"")


@pytest.mark.parametrize(
    "count_key", ["n", "rows", "questions", "n_questions", "n_requests"]
)
@pytest.mark.parametrize(
    "outcome", [{"correct": 1}, {"accuracy": 1.0}, {"metrics": {"accuracy": 1.0}}]
)
def test_suppression_flag_cannot_preserve_individual_outcomes(
    tmp_path: Path, count_key: str, outcome: dict[str, Any]
) -> None:
    raw = json.dumps({"data": {count_key: 1, "suppressed": True, **outcome}}).encode()
    with pytest.raises(ValueError, match="small aggregate cell"):
        check_files(
            selected(tmp_path, "benchmarks/comparison.json", raw), profile="model"
        )


@pytest.mark.parametrize(
    "count_key", ["n", "rows", "questions", "n_questions", "n_requests"]
)
@pytest.mark.parametrize("count", ["1", 1.0, True, -1, None])
def test_aggregate_counts_require_nonnegative_integers(
    tmp_path: Path, count_key: str, count: Any
) -> None:
    raw = json.dumps({"data": {count_key: count, "accuracy": 1.0}}).encode()
    with pytest.raises(ValueError, match="invalid aggregate count"):
        check_files(
            selected(tmp_path, "benchmarks/comparison.json", raw), profile="model"
        )


def test_serialized_candidate_as_dictionary_key_is_rejected(tmp_path: Path) -> None:
    raw = json.dumps({json.dumps({"action": "private operation"}): 0.5}).encode()
    with pytest.raises(ValueError, match="input-shaped"):
        check_files(
            selected(tmp_path, "benchmarks/comparison.json", raw), profile="model"
        )


def test_aggregate_counts_and_public_task_names_are_permitted(tmp_path: Path) -> None:
    raw = json.dumps(
        {
            "data": {
                "tasks": {
                    "public-evaluation": {
                        "questions": 100,
                        "correct": 70,
                        "accuracy": 0.7,
                        "failures": 2,
                    }
                }
            }
        }
    ).encode()
    check_files(selected(tmp_path, "benchmarks/comparison.json", raw), profile="model")


@pytest.mark.parametrize("documented", [False, "true", None])
def test_verification_cannot_claim_real_data_is_synthetic(
    tmp_path: Path, documented: Any
) -> None:
    raw = json.dumps(
        {"synthetic_fixture": documented, "state": {"invoice": 1}}
    ).encode()
    with pytest.raises(ValueError, match="undocumented verification fixture"):
        check_files(
            selected(tmp_path, "verification/VERIFICATION.json", raw), profile="model"
        )


def test_documented_synthetic_fixture_is_permitted(tmp_path: Path) -> None:
    raw = json.dumps({"synthetic_fixture": True, "state": {"invoice": 1}}).encode()
    check_files(
        selected(tmp_path, "verification/VERIFICATION.json", raw), profile="model"
    )


def tensor_bytes(header: dict[str, Any], payload: bytes = b"\x00" * 4) -> bytes:
    encoded = json.dumps(header).encode()
    return struct.pack("<Q", len(encoded)) + encoded + payload


def tensor_header() -> dict[str, Any]:
    return {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}


def test_tensor_payload_is_structurally_checked_not_text_scanned(
    tmp_path: Path,
) -> None:
    raw = tensor_bytes(tensor_header(), b"\xff\xfe\xfd\xfc")
    check_files(
        selected(tmp_path, "model/native-adapters.safetensors", raw), profile="model"
    )


@pytest.mark.parametrize(
    ("header", "payload", "category"),
    [
        ({"__metadata__": {}}, b"", "metadata"),
        (
            {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [1, 5]}},
            b"0" * 5,
            "gaps",
        ),
        (
            {"weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}},
            b"0" * 4,
            "dimensions",
        ),
        (
            {"weight": {"dtype": "UNKNOWN", "shape": [1], "data_offsets": [0, 4]}},
            b"0" * 4,
            "dimensions",
        ),
        (tensor_header(), b"0" * 5, "unreferenced"),
    ],
)
def test_tensor_metadata_gaps_and_trailers_fail(
    tmp_path: Path, header: dict[str, Any], payload: bytes, category: str
) -> None:
    raw = tensor_bytes(header, payload)
    with pytest.raises(ValueError, match=category):
        check_files(
            selected(tmp_path, "model/native-adapters.safetensors", raw),
            profile="model",
        )


def test_duplicate_json_keys_cannot_override_reviewed_content(tmp_path: Path) -> None:
    raw = b'{"data": 1, "data": 2}'
    with pytest.raises(ValueError, match="ambiguous JSON"):
        check_files(
            selected(tmp_path, "benchmarks/comparison.json", raw), profile="model"
        )


def test_symlink_is_rejected_before_read(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("public")
    link = tmp_path / "README.md"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="unsafe publication path"):
        check_files({"README.md": link})


def test_unknown_profile_fails(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown publication privacy profile"):
        check_files({}, profile="unknown")  # type: ignore[arg-type]
