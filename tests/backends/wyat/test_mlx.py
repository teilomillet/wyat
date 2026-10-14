# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Cross-runtime oracles; seeded fixtures establish mechanics, not quality."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
torch = pytest.importorskip("torch")

from fixtures.wyat import create_wyat_bundle  # noqa: E402

from wyat import Candidate, Request  # noqa: E402
from wyat.backends.wyat.encoding import EncodedRequest  # noqa: E402
from wyat.backends.wyat.mlx import (  # noqa: E402
    MLXCandidateHead,
    MLXWyatScorer,
    _batch_groups,
)
from wyat.backends.wyat.torch import CandidateHead, WyatScorer  # noqa: E402


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    return create_wyat_bundle(tmp_path / "wyat")


@pytest.fixture(params=["cpu", "gpu"])
def native(bundle: Path, request: pytest.FixtureRequest) -> MLXWyatScorer:
    device = str(request.param)
    if device == "gpu" and not mx.metal.is_available():
        pytest.skip("Metal GPU is unavailable")
    return MLXWyatScorer.from_directory(bundle, max_length=512, device=device)


def request(texts: Sequence[str], context: str = "Bonjour") -> Request:
    return Request(
        query="read?",
        context=context,
        candidates=tuple(
            Candidate(str(index), text) for index, text in enumerate(texts)
        ),
    )


@given(rows=st.integers(1, 3), choices=st.integers(2, 6), seed=st.integers(0, 30))
@settings(max_examples=12, deadline=None)
def test_fp32_head_matches_torch(rows: int, choices: int, seed: int) -> None:
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        reference = CandidateHead(16, 8).eval()
        candidates = torch.randn(rows, choices, 16)
        query = torch.randn(rows, 16)
    with mx.stream(mx.cpu):
        head = MLXCandidateHead(16, 8)
        head.load_weights(
            [
                (name, mx.array(value.numpy()))
                for name, value in reference.state_dict().items()
            ],
            strict=True,
        )
        actual = head(mx.array(candidates.numpy()), mx.array(query.numpy()))
        mx.eval(actual)
        values = actual.tolist()
    with torch.inference_mode():
        expected = reference(candidates, query)
    torch.testing.assert_close(torch.tensor(values), expected, atol=1e-6, rtol=1e-5)


def test_local_bundle_matches_torch_tokenizer_and_logits(
    native: MLXWyatScorer,
    bundle: Path,
) -> None:
    reference = WyatScorer.from_directory(bundle, max_length=512)
    values = [request(("yes", "no")), request(("世界", "é", "a"), "write " * 7)]
    assert native.model_id == reference.model_id
    assert [native.encode_request(value) for value in values] == [
        reference.encode_request(value) for value in values
    ]
    for actual, expected in zip(
        native.score_many(values), reference.score_many(values), strict=True
    ):
        assert actual == pytest.approx(expected, abs=2e-6, rel=1e-5)
    assert native.score_many([]) == []


@pytest.mark.parametrize(
    ("lengths", "expected"),
    [
        ([], ()),
        ([8], ((0,),)),
        ([8, 16], ((0, 1),)),
        ([9, 9, 9, 32], ((0, 1, 2, 3),)),
        ([8, 8, 8, 24], ((0, 1, 2), (3,))),
        ([32, 8, 8, 8], ((1, 2, 3), (0,))),
    ],
)
def test_batch_groups_respect_rounded_work_threshold(
    lengths: list[int], expected: tuple[tuple[int, ...], ...]
) -> None:
    assert _batch_groups(lengths) == expected


@given(lengths=st.lists(st.integers(1, 2048), max_size=64))
def test_batch_groups_preserve_rows_and_halve_padding(lengths: list[int]) -> None:
    groups = _batch_groups(lengths)
    assert sorted(index for group in groups for index in group) == list(
        range(len(lengths))
    )
    if len(groups) <= 1:
        assert groups == ((tuple(range(len(lengths))),) if lengths else ())
        return
    original = len(lengths) * (-(-max(lengths) // 8) * 8)
    grouped = sum(
        len(group) * (-(-max(lengths[index] for index in group) // 8) * 8)
        for group in groups
    )
    assert grouped * 2 <= original
    for group in groups:
        assert max(lengths[index] for index in group) < 2 * min(
            lengths[index] for index in group
        )


@pytest.mark.parametrize("compile_forward", [False, True])
def test_length_split_preserves_request_and_candidate_order(
    native: MLXWyatScorer,
    bundle: Path,
    monkeypatch: pytest.MonkeyPatch,
    compile_forward: bool,
) -> None:
    native = MLXWyatScorer(
        native.backbone,
        native.head,
        native.encoder,
        model_id=native.model_id,
        device=native.device,
        compile=compile_forward,
    )
    short = request(("世界", "é", "a"))
    long = request(("yes", "no"), "read " * 180)
    values = [short, short, long, short, short, short, short, short]
    assert (
        len(_batch_groups([len(native.encode_request(row).ids) for row in values])) > 1
    )
    batches: list[int] = []
    score = native._score_encoded

    def observe(encoded: Sequence[EncodedRequest]) -> list[list[float]]:
        batches.append(len(encoded))
        return score(encoded)

    monkeypatch.setattr(native, "_score_encoded", observe)
    reference = WyatScorer.from_directory(bundle, max_length=512)
    for batch in (values, values[::-1], values):
        batches.clear()
        actual = native.score_many(batch)
        assert batches == ([7, 1] if native.device.type == mx.gpu else [8])
        for row, expected in zip(actual, reference.score_many(batch), strict=True):
            assert row == pytest.approx(expected, abs=2e-6, rel=1e-5)


@given(
    texts=st.lists(
        st.text(alphabet='abc 世界é\n<>/"\\', min_size=1, max_size=20).filter(
            str.strip
        ),
        min_size=2,
        max_size=5,
    ),
)
@settings(
    max_examples=10,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_causal_right_padding_preserves_single_request_scores(
    native: MLXWyatScorer,
    texts: list[str],
) -> None:
    values = [request(texts), request(("yes", "no"), "read " * 30)]
    batched = native.score_many(values)
    for value, actual in zip(values, batched, strict=True):
        assert actual == pytest.approx(
            native.score_many([value])[0], abs=2e-6, rel=1e-5
        )


def test_compiled_forward_preserves_scores(bundle: Path) -> None:
    eager = MLXWyatScorer.from_directory(bundle, max_length=512, device="cpu")
    compiled = MLXWyatScorer.from_directory(
        bundle, max_length=512, device="cpu", compile=True
    )
    for values in [
        [request(("yes", "no"))],
        [request(("a", "b", "c")), request(("yes", "no"))],
    ]:
        for actual, expected in zip(
            compiled.score_many(values), eager.score_many(values), strict=True
        ):
            assert actual == pytest.approx(expected, abs=2e-6, rel=1e-5)


@pytest.mark.parametrize("device", ["cuda", "mps", "other"])
def test_rejects_wrong_device(bundle: Path, device: str) -> None:
    with pytest.raises(ValueError, match="device"):
        MLXWyatScorer.from_directory(bundle, max_length=512, device=device)


def test_rejects_missing_weight(bundle: Path) -> None:
    path = bundle / "decision_head.safetensors"
    weights = mx.load(str(path))
    weights.pop("scalar.weight")
    mx.save_safetensors(str(path), weights)
    with pytest.raises(ValueError):
        MLXWyatScorer.from_directory(bundle, max_length=512, device="cpu")


@pytest.mark.parametrize(
    "name", ["decision_head.safetensors", "backbone/model.safetensors"]
)
def test_rejects_nonfinite_weight(bundle: Path, name: str) -> None:
    path = bundle / name
    weights = mx.load(str(path))
    key = next(iter(weights))
    weights[key] = mx.full(weights[key].shape, float("nan"), dtype=mx.float32)
    mx.save_safetensors(str(path), weights)
    with pytest.raises(ValueError, match="Nonfinite"):
        MLXWyatScorer.from_directory(bundle, max_length=512, device="cpu")


def test_local_mlx_load_in_fresh_process(bundle: Path) -> None:
    code = (
        "from wyat.backends.wyat.mlx import MLXWyatScorer; "
        "import sys; "
        "MLXWyatScorer.from_directory(sys.argv[1], max_length=512, device='cpu')"
    )
    subprocess.run(
        [sys.executable, "-c", code, str(bundle)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_prefix_cache_preserves_scores_and_frozen_state(native: MLXWyatScorer) -> None:
    from wyat.backends.wyat.prefix import PrefixScorer

    values = [request(("yes", "no")), request(("世界", "é", "a"), "Bonjour " * 7)]
    cached = PrefixScorer(native, native.encode_request(values[0]).ids[:3])
    with mx.stream(native.device):
        frozen = [(keys.tolist(), vals.tolist()) for keys, vals in cached._states]
    for batch in (values, values[:1], values[::-1], values):
        for actual, expected in zip(
            cached.score_many(batch), native.score_many(batch), strict=True
        ):
            assert actual == pytest.approx(expected, abs=2e-6, rel=1e-5)
    # Appending the suffix must allocate only the exact logical length and
    # preserve the shared snapshots, including broadcast batch views.
    with mx.stream(native.device):
        for cache in cached._fork(3):
            keys, vals = cache.state
            suffix = mx.ones((3, *keys.shape[1:2], 5, keys.shape[-1]))
            actual_keys, actual_values = cache.update_and_fetch(suffix, suffix)
            mx.eval(actual_keys, actual_values)
            assert cache.offset == len(cached.prefix) + 5
            assert actual_keys.shape[2] == actual_values.shape[2] == cache.offset
        assert frozen == [
            (keys.tolist(), vals.tolist()) for keys, vals in cached._states
        ]
    mismatch = request(("yes", "no"), "different context")
    assert not cached._admits(native.encode_request(mismatch))
    assert cached.score_many([values[0], mismatch]) == native.score_many(
        [values[0], mismatch]
    )
