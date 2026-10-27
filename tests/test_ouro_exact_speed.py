# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Bit-exact CPU gate for wyat-2 exact-speed runtime changes.

Goldens were recorded from the unmodified release runtime at HEAD ``ea16d44a``
(``wyat/loading/ouro.py`` blob 7b52b0b6, sha256 e514d89d...) on the host below.
Exact-by-construction speed changes may alter when, where or how often work
happens; they may never change FP32 values, op shapes/dtypes, or value-changing
op order. ``_GOLDEN`` holds the raw uint32 bit patterns of every candidate logit
and is compared exactly - no tolerance, NaN and signed-zero bits included.

The grid covers the released ``wyat-ouro-decision-v2`` execution contract on a
tiny random artifact: exact single context pass at/below
``single_chunk_max_tokens``, chunked flash and lean context, shared-context
batches with unequal and equal candidate lengths, a forced multi-batch split,
the per-candidate path, and several record rows. CPU kernels are host-specific:
on another interpreter/BLAS the test skips loudly (set
``WYAT_OURO_EXACT_STRICT=1`` to fail instead). Receipt:
``.keme/maintenance/wyat2-exact-speed-20261027/verify/golden-v2-cpu.json``.
"""

from __future__ import annotations

import os
import platform
import random
import sys
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from test_ouro_artifact import (  # noqa: E402
    CONTEXT_PREFILL,
    NUMERICS,
    _build_ouro_artifact,
)

from wyat import tasks  # noqa: E402
from wyat.loading import ouro  # noqa: E402
from wyat.loading.ouro import OuroScorer  # noqa: E402
from wyat.loading.ouro_artifact import CandidateBatch  # noqa: E402

_VOCAB = 320
_STRICT_ENV = "WYAT_OURO_EXACT_STRICT"
# The released wyat-2 shape: exact single pass up to 1024 context tokens, fused
# flash chunks above it. The lean artifact uses the fixture's CONTEXT_PREFILL.
_FLASH_PREFILL = {
    **CONTEXT_PREFILL,
    "chunk": 128,
    "single_chunk_max_tokens": 1024,
    "chunked_attention": "flash",
}
_RECORD = {
    "state": {"text": "Bonjour 世界 é", "count": 7},
    "questions": {
        "choice": {
            "type": "choice",
            "instructions": "Which applies?",
            "criteria": {"a": "short", "b": "a much longer option " * 3, "c": "x"},
        },
        "score": {
            "type": "score",
            "instructions": "Rate.",
            "criteria": ["no", "maybe", "yes"],
        },
    },
}
_GOLDEN_ENVIRONMENT = {
    "torch": "2.14.1",
    "python": "3.12.11",
    "system": "darwin",
    "machine": "arm64",
}


def _skip_unless_golden_environment() -> None:
    observed = {
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "system": sys.platform,
        "machine": platform.machine(),
    }
    if observed != _GOLDEN_ENVIRONMENT:
        message = f"goldens recorded on {_GOLDEN_ENVIRONMENT}, host is {observed}"
        if os.environ.get(_STRICT_ENV):
            pytest.fail(message)
        pytest.skip(message)


def _ids(seed: int, length: int) -> tuple[int, ...]:
    """Stable token IDs (CPython Mersenne Twister, independent of torch)."""
    generator = random.Random(seed)
    return tuple(generator.randrange(_VOCAB) for _ in range(length))


def _row_of(
    seed: int, context_length: int, candidates: tuple[tuple[int, int], ...]
) -> CandidateBatch:
    question = tasks.Question(
        id="grid",
        type="choice",
        instructions="Choose.",
        options=tuple(
            (f"o{index}", f"option {index}") for index in range(len(candidates))
        ),
    )
    return CandidateBatch(
        question=question,
        context_ids=_ids(seed, context_length),
        candidate_ids=tuple(_ids(s, n) for s, n in candidates),
    )


def _logits(scorer: OuroScorer, row: CandidateBatch) -> torch.Tensor:
    return torch.tensor(scorer.question_logits(row), dtype=torch.float32)


_UNEQUAL = ((201, 5), (202, 2), (203, 7), (204, 2), (205, 4))
_EQUAL = ((211, 6), (212, 6), (213, 6), (214, 6))


def _multi_batch_scores(scorer: OuroScorer) -> torch.Tensor:
    """One candidate list forced across several shared-context batches."""
    with torch.inference_mode():
        cache = ouro.prefill_context(
            scorer.backbone,
            torch.tensor([_ids(301, 40)], dtype=torch.long),
            policy="last",
            chunk=64,
            attention="lean",
        )
        scores = ouro.score_candidates_shared_context(
            scorer.backbone,
            scorer.head,
            cache,
            [
                torch.tensor([_ids(seed, length)], dtype=torch.long)
                for seed, length in _UNEQUAL
            ],
            score_budget_bytes=12_800,
        )
    return scores.detach().float()


def _scorer_case(
    position: str,
    seed: int,
    context_length: int,
    candidates: tuple[tuple[int, int], ...],
) -> Callable[[dict[str, OuroScorer]], torch.Tensor]:
    def case(scorers: dict[str, OuroScorer]) -> torch.Tensor:
        return _logits(scorers[position], _row_of(seed, context_length, candidates))

    return case


def _record_rows(scorers: dict[str, OuroScorer]) -> torch.Tensor:
    scorer = scorers["lean:shared_context"]
    return torch.cat([_logits(scorer, row) for row in scorer.encode_record(_RECORD)])


_GRID: tuple[tuple[str, Callable[[dict[str, OuroScorer]], torch.Tensor]], ...] = (
    ("lean-single-shared", _scorer_case("lean:shared_context", 101, 40, _UNEQUAL)),
    ("lean-chunked-shared", _scorer_case("lean:shared_context", 102, 200, _UNEQUAL)),
    (
        "lean-chunked-shared-equal",
        _scorer_case("lean:shared_context", 103, 200, _EQUAL),
    ),
    (
        "lean-single-per-candidate",
        _scorer_case("lean:per_candidate", 104, 40, _UNEQUAL),
    ),
    ("flash-single-pass", _scorer_case("flash:shared_context", 105, 1000, _UNEQUAL)),
    ("flash-chunked", _scorer_case("flash:shared_context", 106, 1100, _UNEQUAL)),
    ("record-rows", _record_rows),
    ("shared-multi-batch", lambda s: _multi_batch_scores(s["lean:shared_context"])),
)

# Space-separated uint32 hex bit patterns of the FP32 logits, recorded from HEAD.
_GOLDEN: dict[str, str] = {
    "lean-single-shared": "bd6ea6f7 bcb70c59 bd9bf44e bcb1cac4 bd4ca384",
    "lean-chunked-shared": "3daf9466 3dc3341b 3da43ca6 3dc3ae60 3db43d98",
    "lean-chunked-shared-equal": "3e9238c0 3e91e1fb 3e926133 3e9297b9",
    "lean-single-per-candidate": "3e3b2a40 3e48168c 3e2e8e26 3e48d9cc 3e40dfaa",
    "flash-single-pass": "3e8e18e8 3e8b7945 3e8edd9b 3e8c23e1 3e8ce6f5",
    "flash-chunked": "3df7e1b7 3df5de90 3df6b2cc 3df4716e 3df56759",
    "record-rows": "3e53cf4d 3e876226 3e547a9e 3e55983f 3e5893da 3e55d045",
    "shared-multi-batch": "3e4ae7d8 3e44b0ff 3e4ee2e0 3e46d260 3e43fbfa",
}


def _bits(values: torch.Tensor) -> tuple[int, ...]:
    return tuple(value & 0xFFFFFFFF for value in values.view(torch.int32).tolist())


def _assert_golden(name: str, observed: torch.Tensor) -> None:
    golden = tuple(int(value, 16) for value in _GOLDEN[name].split())
    if _bits(observed) != golden:
        pytest.fail(f"{name}: {_bits(observed)} != golden {golden}")


@pytest.fixture(scope="module", autouse=True)
def _golden_process_state() -> Iterator[None]:
    """Pin CPU threads and restore the process numerics a v2 artifact applies."""
    saved = (
        torch.get_num_threads(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.get_float32_matmul_precision(),
        torch.are_deterministic_algorithms_enabled(),
        torch.utils.deterministic.fill_uninitialized_memory,
    )
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(saved[0])
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved[1:3]
    torch.set_float32_matmul_precision(saved[3])
    torch.use_deterministic_algorithms(saved[4])
    torch.utils.deterministic.fill_uninitialized_memory = saved[5]  # type: ignore[attr-defined]


def _load_scorer_grid(root: Path) -> dict[str, OuroScorer]:
    scorers: dict[str, OuroScorer] = {}
    for kind, prefill in (("lean", CONTEXT_PREFILL), ("flash", _FLASH_PREFILL)):
        directory = root / kind
        directory.mkdir(parents=True)
        artifact = _build_ouro_artifact(
            directory, cache_entries=0, context_prefill=prefill, numerics=NUMERICS
        )
        for mode in ("shared_context", "per_candidate"):
            scorer = OuroScorer.from_directory(
                artifact, device="cpu", candidate_batching=mode
            )
            # Memory admission is not arithmetic; a bitwise gate must not be
            # flaky under host memory pressure.
            scorer.resource_guard.enabled = False
            scorers[f"{kind}:{mode}"] = scorer
    return scorers


@pytest.fixture(scope="module")
def scorer_grid(tmp_path_factory: pytest.TempPathFactory) -> dict[str, OuroScorer]:
    return _load_scorer_grid(tmp_path_factory.mktemp("ouro-exact-speed"))


def _observed(
    scorers: dict[str, OuroScorer], *, reverse: bool = False
) -> dict[str, torch.Tensor]:
    cases = reversed(_GRID) if reverse else _GRID
    return {name: case(scorers) for name, case in cases}


def test_grid_matches_head_golden(scorer_grid: dict[str, OuroScorer]) -> None:
    _skip_unless_golden_environment()
    assert torch.get_num_threads() == 1
    observed = _observed(scorer_grid)
    assert set(observed) == set(_GOLDEN)
    for name in _GOLDEN:
        _assert_golden(name, observed[name])
    lean = scorer_grid["lean:shared_context"]
    flash = scorer_grid["flash:shared_context"]
    assert lean._context_execution(40) == (None, "native", None)
    assert lean._context_execution(200) == (64, "lean", None)
    assert flash._context_execution(1000) == (None, "native", None)
    assert flash._context_execution(1100) == (128, "flash", None)
    assert len(ouro.shared_context_batches([5, 2, 7, 2, 4], 40, 4, 12_800)) > 1


def test_grid_is_deterministic(scorer_grid: dict[str, OuroScorer]) -> None:
    _skip_unless_golden_environment()
    first = _observed(scorer_grid)
    second = _observed(scorer_grid, reverse=True)
    for name in _GOLDEN:
        assert _bits(first[name]) == _bits(second[name]), f"{name} not deterministic"
        _assert_golden(name, second[name])


@pytest.mark.parametrize("mode", ["shared_context", "per_candidate"])
def test_out_of_range_candidate_refused(
    scorer_grid: dict[str, OuroScorer], mode: str
) -> None:
    row = _row_of(401, 40, ((402, 4), (403, 2), (404, 2)))
    row = replace(
        row,
        candidate_ids=(row.candidate_ids[0], (7, 11, _VOCAB, 3), row.candidate_ids[2]),
    )
    with pytest.raises(ValueError, match="outside the vocabulary"):
        scorer_grid[f"lean:{mode}"].question_logits(row)
