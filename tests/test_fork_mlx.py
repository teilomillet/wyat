# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Option-fork artifacts on MLX reproduce the PyTorch runtime's logits."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
torch = pytest.importorskip("torch")
from test_fork_artifact import RECORD, _artifact, _config  # noqa: E402

import wyat  # noqa: E402
from wyat.loading import fork_artifact  # noqa: E402
from wyat.loading.fork import ForkScorer  # noqa: E402
from wyat.loading.fork_mlx import load_fork_mlx  # noqa: E402

DEVICES = [
    "cpu",
    pytest.param(
        "gpu",
        marks=pytest.mark.skipif(
            not mx.metal.is_available(), reason="requires Apple Metal"
        ),
    ),
]
MANY = {
    "state": "x" * 40,
    "questions": {
        "q": {
            "type": "choice",
            "instructions": "Pick.",
            "criteria": {f"o{i}": "w" * (1 + i % 9) for i in range(45)},
        }
    },
}


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    # The Metal gated-delta kernel needs key heads at least 32 wide.
    config = _config(257)
    config.linear_key_head_dim = config.linear_value_head_dim = 32
    return _artifact(tmp_path_factory.mktemp("fork-mlx"), config=config)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("responsive", [True, False])
def test_mlx_logits_match_pytorch(
    artifact: Path, device: str, responsive: bool
) -> None:
    expected = ForkScorer.from_directory(artifact).logits(RECORD)
    actual = load_fork_mlx(artifact, device=device, responsive=responsive).logits(
        RECORD
    )
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            torch.tensor(left), torch.tensor(right), atol=1e-5, rtol=1e-4
        )


@pytest.mark.parametrize("device", DEVICES)
def test_mlx_branch_groups_match_pytorch(artifact: Path, device: str) -> None:
    reference = ForkScorer.from_directory(artifact)
    scorer = load_fork_mlx(artifact, device=device)
    reference.max_batch_tokens = scorer.max_batch_tokens = 600
    (row,) = scorer.encode_record(MANY)
    groups = fork_artifact.branch_groups(row.option_ids, len(row.prefix_ids), 600)
    assert len(list(groups)) > 2
    torch.testing.assert_close(
        torch.tensor(scorer.question_logits(row)),
        torch.tensor(reference.question_logits(row)),
        atol=1e-5,
        rtol=1e-4,
    )


def test_public_load_decides_on_mlx(artifact: Path) -> None:
    model = wyat.load(artifact, backend="mlx", device="cpu")
    reference = wyat.load(artifact)
    assert model.model_id == reference.model_id
    request = wyat.Request(
        query="Which applies?",
        context="Bonjour 世界 é",
        candidates=(wyat.Candidate("a", "short"), wyat.Candidate("b", "longer")),
    )
    decision = model.decide(request)
    expected = reference.decide(request)
    assert decision.selected == expected.selected
    torch.testing.assert_close(
        torch.tensor(decision.scores),
        torch.tensor(expected.scores),
        atol=1e-5,
        rtol=1e-4,
    )
    with pytest.raises(wyat.ModelInputError) as refused:
        wyat.load(artifact, backend="mlx", device="cpu", max_length=16).decide(request)
    assert refused.value.reason == "input-length"


def test_mlx_needs_no_pytorch(artifact: Path) -> None:
    expected = ForkScorer.from_directory(artifact).logits(RECORD)
    script = (
        "import json, sys; sys.modules['torch'] = None\n"
        "from wyat.loading.fork_mlx import load_fork_mlx\n"
        "scorer = load_fork_mlx(sys.argv[1], device='cpu')\n"
        "print(json.dumps(scorer.logits(json.loads(sys.argv[2]))))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(artifact), json.dumps(RECORD)],
        check=True,
        capture_output=True,
        text=True,
    )
    actual = json.loads(completed.stdout.strip().splitlines()[-1])
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            torch.tensor(left), torch.tensor(right), atol=1e-5, rtol=1e-4
        )


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Apple Metal")
def test_gpu_refuses_key_heads_the_kernel_cannot_run(tmp_path: Path) -> None:
    narrow = _artifact(tmp_path)
    with pytest.raises(ValueError, match="divisible by 32"):
        load_fork_mlx(narrow, device="gpu")
    assert len(load_fork_mlx(narrow, device="cpu").logits(RECORD)) == 3


def test_invalid_mlx_options_fail_before_loading(artifact: Path) -> None:
    with pytest.raises(ValueError, match="cpu or gpu"):
        load_fork_mlx(artifact, device="cuda")
    with pytest.raises(ValueError, match="artifact's limit"):
        load_fork_mlx(artifact, device="cpu", max_length=1 << 20)
