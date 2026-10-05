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

from benchmarks.fixtures.kai import create_kai_bundle  # noqa: E402
from wyat import Candidate, Request  # noqa: E402
from wyat.backends.kai.mlx import MLXCandidateHead, MLXKaiScorer  # noqa: E402
from wyat.backends.kai.torch import CandidateHead, KaiScorer  # noqa: E402


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    return create_kai_bundle(tmp_path / "kai")


@pytest.fixture(params=["cpu", "gpu"])
def native(bundle: Path, request: pytest.FixtureRequest) -> MLXKaiScorer:
    device = str(request.param)
    if device == "gpu" and not mx.metal.is_available():
        pytest.skip("Metal GPU is unavailable")
    return MLXKaiScorer.from_directory(bundle, max_length=512, device=device)


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
    native: MLXKaiScorer,
    bundle: Path,
) -> None:
    reference = KaiScorer.from_directory(bundle, max_length=512)
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
    native: MLXKaiScorer,
    texts: list[str],
) -> None:
    values = [request(texts), request(("yes", "no"), "read " * 30)]
    batched = native.score_many(values)
    for value, actual in zip(values, batched, strict=True):
        assert actual == pytest.approx(
            native.score_many([value])[0], abs=2e-6, rel=1e-5
        )


def test_compiled_forward_preserves_scores(bundle: Path) -> None:
    eager = MLXKaiScorer.from_directory(bundle, max_length=512, device="cpu")
    compiled = MLXKaiScorer.from_directory(
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
        MLXKaiScorer.from_directory(bundle, max_length=512, device=device)


def test_rejects_missing_weight(bundle: Path) -> None:
    path = bundle / "decision_head.safetensors"
    weights = mx.load(str(path))
    weights.pop("scalar.weight")
    mx.save_safetensors(str(path), weights)
    with pytest.raises(ValueError):
        MLXKaiScorer.from_directory(bundle, max_length=512, device="cpu")


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
        MLXKaiScorer.from_directory(bundle, max_length=512, device="cpu")


def test_local_mlx_load_in_fresh_process(bundle: Path) -> None:
    code = (
        "from wyat.backends.kai.mlx import MLXKaiScorer; "
        "import sys; "
        "MLXKaiScorer.from_directory(sys.argv[1], max_length=512, device='cpu')"
    )
    subprocess.run(
        [sys.executable, "-c", code, str(bundle)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
