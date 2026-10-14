"""Tiny common-path, storage-alias, admission and failure-lifecycle challenges."""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import cast

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from torch import nn
from transformers import Qwen3Model
from transformers.cache_utils import DynamicLayer

from benchmarks.analysis.torch.prefix.cache import PrefixScorer
from benchmarks.analysis.torch.prefix.memory import study as runner
from benchmarks.analysis.torch.prefix.memory.scorer import (
    Arm,
    MemoryScorer,
    StorageObservation,
    storage_bytes,
)
from benchmarks.analysis.torch.prefix.pairs import (
    admit_history,
)
from benchmarks.analysis.torch.prefix.pairs import (
    digest as source_digest,
)
from benchmarks.fixtures.wyat import create_wyat_bundle
from wyat import Candidate, Request
from wyat.backends.wyat.encoding import EncodedRequest
from wyat.backends.wyat.torch import WyatScorer
from wyat.wire import read_object


@pytest.fixture
def base(tmp_path: Path) -> WyatScorer:
    original = WyatScorer.from_directory(
        create_wyat_bundle(tmp_path / "wyat"), max_length=512
    )
    config = copy.deepcopy(original.backbone.config)
    config.num_hidden_layers = 2
    config.layer_types = ["full_attention"] * 2
    with torch.random.fork_rng():
        torch.manual_seed(19)
        backbone = Qwen3Model(config)
    cast(nn.Module, backbone).eval()
    return WyatScorer(
        backbone,
        original.head,
        original.tokenizer,
        device=original.device,
        max_length=512,
        max_options=original.max_options,
        token_cache_size=0,
        model_id="tiny2",
    )


def request(query: str = "read?", count: int = 2) -> Request:
    return Request(
        query,
        tuple(Candidate(str(i), "read " * (i + 1)) for i in range(count)),
        context="Bonjour",
    )


@settings(
    max_examples=8,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(repeats=st.integers(1, 6), count=st.integers(2, 4))
def test_both_common_paths_match_independent_full_and_original_stock(
    base: WyatScorer,
    repeats: int,
    count: int,
) -> None:
    requests = (request(), request("write? " * repeats, count))
    prefix = base.encode_request(requests[0]).ids[:2]
    legacy = PrefixScorer(base, prefix)
    full = base.score_many(requests)
    stock = legacy.score_many(requests)
    for arm in ("retained", "one-use"):
        scorer = MemoryScorer(base, prefix, arm)
        snapshots = tuple((k.clone(), v.clone()) for k, v in scorer._states)
        first = scorer.score_many(requests)
        scorer.score_many(requests[::-1])
        again = scorer.score_many(requests)
        observed: list[StorageObservation] = []
        instrumented = scorer.score_observed(requests, observed.append)
        for actual in (first, again, instrumented):
            for a, f, s in zip(actual, full, stock, strict=True):
                torch.testing.assert_close(
                    torch.tensor(a), torch.tensor(f), atol=1e-6, rtol=1e-5
                )
                torch.testing.assert_close(
                    torch.tensor(a), torch.tensor(s), atol=1e-6, rtol=1e-5
                )
        assert first == again and scorer.score_many(()) == []
        assert [item.layer for item in observed] == [None, 0, 1]
        for state, expected in zip(scorer._states, snapshots, strict=True):
            assert torch.equal(state[0], expected[0]) and torch.equal(
                state[1], expected[1]
            )
        # Stored evidence has primitive values only, never Tensor/Storage owners.
        assert all(
            all(value is None or type(value) is int for value in asdict(item).values())
            for item in observed
        )
    assert base.backbone.config.use_cache is False


def test_storage_inventory_counts_backing_allocation_once() -> None:
    tensor = torch.ones(2, 3, 7)
    alias = tensor[:, :, :1]
    assert (
        storage_bytes((tensor, alias, tensor.expand(4, -1, -1, -1)))
        == tensor.numel() * 4
    )
    assert storage_bytes((tensor, tensor.clone())) == tensor.numel() * 8


def test_one_use_counts_returned_storage_after_reset_and_lowers_selected_pool(
    base: WyatScorer,
) -> None:
    requests = (request("read " * 10), request("write " * 20))
    prefix = base.encode_request(requests[0]).ids[:2]
    rows: dict[str, list[StorageObservation]] = {}
    for arm in ("retained", "one-use"):
        rows[arm] = []
        MemoryScorer(base, prefix, arm).score_observed(requests, rows[arm].append)
    retained, one = rows["retained"], rows["one-use"]
    assert retained[0] == one[0]
    assert one[-1].cache_owned_bytes == 0 and one[-1].returned_logical_bytes > 0
    assert one[-1].combined_unique_storage_bytes > one[-1].returned_logical_bytes
    assert retained[-1].cache_owned_bytes == retained[-1].returned_logical_bytes * 2
    assert max(item.combined_unique_storage_bytes for item in one) < max(
        item.combined_unique_storage_bytes for item in retained
    )


@pytest.mark.parametrize("arm", ["retained", "one-use"])
@pytest.mark.parametrize("kind", ["mismatch", "candidate", "query", "empty"])
def test_whole_batch_fallback_keeps_original_admission(
    base: WyatScorer,
    monkeypatch: pytest.MonkeyPatch,
    arm: Arm,
    kind: str,
) -> None:
    requests = (request(), request("write?"))
    row = base.encode_request(requests[0])
    scorer = MemoryScorer(base, row.ids[:2], arm)
    invalid = {
        "mismatch": EncodedRequest(
            (row.ids[0] + 1, *row.ids[1:]), row.candidate_positions, row.query_position
        ),
        "candidate": EncodedRequest(
            row.ids, (1, row.candidate_positions[1]), row.query_position
        ),
        "query": EncodedRequest(row.ids, row.candidate_positions, 1),
        "empty": EncodedRequest(scorer.prefix, (0, 1), 1),
    }[kind]
    calls: list[tuple[Request, ...]] = []

    def encode(value: Request) -> EncodedRequest:
        return row if value is requests[0] else invalid

    def fallback(values: Sequence[Request]) -> list[list[float]]:
        calls.append(tuple(values))
        return [[1.0, 2.0], [3.0, 4.0]]

    observed: list[StorageObservation] = []
    monkeypatch.setattr(base, "encode_request", encode)
    monkeypatch.setattr(base, "score_many", fallback)
    assert scorer.score_observed(requests, observed.append) == [[1.0, 2.0], [3.0, 4.0]]
    assert calls == [requests] and observed == []


@pytest.mark.parametrize("arm", ["retained", "one-use"])
def test_original_input_errors_and_fresh_owner_scope(base: WyatScorer, arm: Arm) -> None:
    scorer = MemoryScorer(base, base.encode_request(request()).ids[:2], arm)
    with pytest.raises(ValueError, match="exceeds"):
        scorer.score_many((request(), request("read " * 600)))
    with pytest.raises(ValueError, match="choice"):
        scorer.score_many((Request("read?", request().candidates, task="score"),))
    with pytest.raises(ValueError, match="positive"):
        scorer.fork(True)
    with torch.inference_mode():
        a, b = scorer.fork(2), scorer.fork(2)
        for left, right, seed in zip(a.layers, b.layers, scorer._states, strict=True):
            assert isinstance(left, DynamicLayer) and isinstance(right, DynamicLayer)
            assert left.keys is not None and right.keys is not None
            assert left.keys.data_ptr() != right.keys.data_ptr() != seed[0].data_ptr()


@pytest.mark.parametrize(
    "arm,repeat", [("unknown", 0), ("retained", True), ("one-use", 3)]
)
def test_invalid_study_identity_before_effects(
    tmp_path: Path, arm: str, repeat: int
) -> None:
    with pytest.raises(ValueError, match="requires"):
        runner.study(
            tmp_path / "model", tmp_path / "out.json", arm=cast(Arm, arm), repeat=repeat
        )
    assert not (tmp_path / "out.json").exists()


def test_history_failure_retained_before_model_load_and_threads_restored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous = torch.get_num_threads()

    def fail() -> dict[str, object]:
        raise ValueError("history identity failed")

    monkeypatch.setattr(runner, "admit_history", fail)
    output = tmp_path / "out.json"
    with pytest.raises(ValueError, match="history identity"):
        runner.study(tmp_path / "model", output, arm="retained", repeat=0)
    result = read_object(output)
    assert result["completed"] is result["screen_passed"] is result["accepted"] is False
    assert torch.get_num_threads() == previous


def test_dispatch_requires_gqa_exact_offset_mask_and_28_calls() -> None:
    event: dict[str, object] = {
        "operator": "aten::_scaled_dot_product_flash_attention_for_cpu",
        "count": 28,
        "input_shapes": [
            [8, 16, 167, 128],
            [8, 8, 192, 128],
            [8, 8, 192, 128],
            [],
            [],
            [8, 1, 167, 192],
            [],
        ],
    }
    assert runner.dispatch([event], 167)["passed"] is True
    assert runner.dispatch([{**event, "count": 27}], 167)["passed"] is False
    shapes = copy.deepcopy(cast(list[list[int]], event["input_shapes"]))
    shapes[1][1] = shapes[2][1] = 16
    assert runner.dispatch([{**event, "input_shapes": shapes}], 167)["passed"] is False
    shapes[5] = []
    assert runner.dispatch([{**event, "input_shapes": shapes}], 167)["passed"] is False
    assert (
        runner.dispatch(
            [
                event,
                {"operator": "aten::_scaled_dot_product_attention_math", "count": 1},
            ],
            167,
        )["passed"]
        is False
    )


def test_cli_parent_does_not_import_torch() -> None:
    # Child import inventory is observed in a clean interpreter, not inferred
    # from a monkeypatched current test interpreter that already imports Torch.
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import benchmarks.analysis.torch.prefix.memory.run; "
            "assert 'torch' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture
def fake_study(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pure fake logits/observers test journal and failure policy, never real RSS."""
    from types import TracebackType

    from transformers.integrations import sdpa_attention

    from benchmarks.analysis.torch.prefix.memory.measure import MemoryReport
    from wyat.benchmark import BenchmarkCycle
    from wyat.core import decision_from_logits
    from wyat.evaluation import EvaluationCase, EvaluationRow

    history = admit_history()
    identifier = cast(str, history["model_id"])
    prefix = tuple(cast(list[int], history["prefix_tokens"]))

    class FakeBase:
        model_id = identifier

        def encode_request(self, value: Request) -> EncodedRequest:
            width = len(value.candidates)
            return EncodedRequest(
                prefix + tuple(range(width + 2)),
                tuple(range(25, 25 + width)),
                26 + width,
            )

        def score_many(self, values: Sequence[Request]) -> list[list[float]]:
            return [
                [float(-i) for i in range(len(value.candidates))] for value in values
            ]

    class FakeMemory(FakeBase):
        prefix_state_bytes = 123

        def __init__(self, base: object, tokens: tuple[int, ...], arm: Arm) -> None:
            assert tokens == prefix

        def admits(self, row: EncodedRequest) -> bool:
            return row.ids[:25] == prefix

        def score_observed(
            self, values: Sequence[Request], observer: object
        ) -> list[list[float]]:
            from benchmarks.analysis.torch.prefix.memory.scorer import Observer

            callback = cast(Observer, observer)
            callback(StorageObservation(None, 8 * 123, 0, 9 * 123))
            for layer in range(28):
                callback(StorageObservation(layer, 100, 200, 300))
            return self.score_many(values)

    class FakeSampler:
        def __enter__(self) -> FakeSampler:
            return self

        def __exit__(
            self,
            kind: type[BaseException] | None,
            error: BaseException | None,
            traceback: TracebackType | None,
        ) -> None:
            return None

        def report(self) -> MemoryReport:
            return MemoryReport(
                True,
                20_000_000,
                512,
                30.0,
                False,
                False,
                1,
                (20_000_000,) * 4,
                0,
                80_000_000,
                3,
                (20_000_000,) * 4,
                (),
                None,
                1000,
                (),
            )

    def reference(
        path: Path,
        cases: tuple[EvaluationCase, ...],
        encoder: object,
        *,
        dataset_sha256: str,
    ) -> BenchmarkCycle:
        rows = tuple(
            EvaluationRow(
                case.id,
                case.expected,
                "ok",
                decision,
                decision.selected in case.expected,
                0,
                None,
            )
            for case in cases
            for decision in [
                decision_from_logits(
                    case.request,
                    tuple(float(-i) for i in range(len(case.request.candidates))),
                    identifier,
                )
            ]
        )
        return BenchmarkCycle(0, rows, 0, 0)

    def load(*args: object, **kwargs: object) -> FakeBase:
        return FakeBase()

    monkeypatch.setattr(WyatScorer, "from_directory", load)
    monkeypatch.setattr(runner, "MemoryScorer", FakeMemory)
    monkeypatch.setattr(runner, "MemorySampler", FakeSampler)
    monkeypatch.setattr(runner, "read_reference", reference)
    monkeypatch.setattr(runner, "dispatch", lambda events, length: {"passed": True})
    monkeypatch.setattr(sdpa_attention, "_is_torch_mps_available", True)
    monkeypatch.setattr(sdpa_attention, "_is_torch_greater_or_equal_than_2_13", True)


def test_complete_fake_cycles_and_history_keep_all_raw_gates(
    tmp_path: Path,
    fake_study: None,
) -> None:
    output = tmp_path / "out.json"
    assert runner.study(tmp_path / "model", output, arm="retained", repeat=0)
    result = read_object(output)
    cycles = cast(list[dict[str, object]], result["cycles"])
    journal = [
        json.loads(line)
        for line in output.with_suffix(".cycles.jsonl").read_text().splitlines()
    ]
    assert (
        cycles == journal
        and [row["stage"] for row in cycles] == runner.POLICY["cycles"]
    )
    assert all(row["passed"] is True for row in cycles)
    assert (
        sum(
            len(cast(list[object], cast(dict[str, object], row["cycle"])["rows"]))
            for row in cycles
        )
        == 48
    )
    assert (
        result["completed"]
        is result["quality_passed"]
        is result["screen_passed"]
        is True
    )
    assert result["accepted"] is False
    assert cast(dict[str, object], result["full77_admission"])["raw_outputs"] == 2156


def test_late_write_failure_resets_flags_and_preserves_every_raw_cycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_study: None,
) -> None:
    original = Path.write_text

    def write(path: Path, text: str) -> int:
        if path.name == "out.json" and json.loads(text).get("completed") is True:
            raise OSError("final write failed deliberately")
        return original(path, text)

    monkeypatch.setattr(Path, "write_text", write)
    output = tmp_path / "out.json"
    with pytest.raises(OSError, match="final write"):
        runner.study(tmp_path / "model", output, arm="one-use", repeat=1)
    result = read_object(output)
    assert (
        result["completed"]
        is result["quality_passed"]
        is result["screen_passed"]
        is result["accepted"]
        is False
    )
    assert len(cast(list[object], result["cycles"])) == 6
    assert len(output.with_suffix(".cycles.jsonl").read_text().splitlines()) == 6


@pytest.mark.parametrize("evidence", ["receipt", "journal", "wall", "source_archive"])
def test_confirmed_full77_history_tamper_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    evidence: str,
) -> None:
    original = source_digest
    review = read_object(
        runner.ROOT / "benchmarks/analysis/results/torch-prefix77-review.json"
    )
    row = cast(list[dict[str, object]], review["shape_results"])[0]
    target = runner.evidence_path(
        cast(dict[str, dict[str, str]], row["evidence"])[evidence]["path"]
    )

    def digest(path: Path) -> str:
        return "0" * 64 if path == target else original(path)

    monkeypatch.setattr(runner, "digest", digest)
    with pytest.raises(ValueError, match="confirmed history"):
        runner.admit_confirmation(admit_history())


@pytest.mark.parametrize(
    "change", ["missing", "duplicate", "seed-alias", "zero-returned"]
)
def test_inventory_requires_complete_order_and_private_seed_geometry(
    change: str,
) -> None:
    rows = [StorageObservation(None, 80, 0, 90)] + [
        StorageObservation(i, 80, 20, 100) for i in range(28)
    ]
    assert runner.inventory_complete(rows, 10)
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows[-1] = rows[-2]
    elif change == "seed-alias":
        rows[0] = StorageObservation(None, 10, 0, 10)
    else:
        rows[-1] = StorageObservation(27, 0, 0, 10)
    assert not runner.inventory_complete(rows, 10)
    assert not runner.inventory_complete([], 10)


def test_incomplete_inventory_is_unsupported_with_raw_outputs_retained(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_study: None,
) -> None:
    from benchmarks.analysis.torch.prefix.memory.scorer import Observer

    def incomplete(
        self: object, requests: Sequence[Request], observer: Observer
    ) -> list[list[float]]:
        observer(StorageObservation(None, 8 * 123, 0, 9 * 123))
        return [[float(-i) for i in range(len(value.candidates))] for value in requests]

    monkeypatch.setattr(
        cast(type[MemoryScorer], vars(runner)["MemoryScorer"]),
        "score_observed",
        incomplete,
    )
    output = tmp_path / "out.json"
    assert not runner.study(tmp_path / "model", output, arm="one-use", repeat=2)
    result = read_object(output)
    assert result["completed"] is result["quality_passed"] is True
    assert result["screen_passed"] is result["accepted"] is False
    memory = cast(list[dict[str, object]], result["cycles"])[4]
    assert memory["selected_live_storage_peak_bytes"] is None
    assert "Selected KV inventory missing/duplicated/invalid" in cast(
        list[str], result["unsupported_reasons"]
    )
    assert len(cast(list[object], result["cycles"])) == 6
