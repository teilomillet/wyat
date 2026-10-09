"""Local typed Qwen inference and the canonical full-artifact contract.

Axes writes this format; Wyat owns validation, execution and typed readouts.
Adapters remain separate: merging would change the trained floating-point math.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F
from transformers import Qwen3Config, Qwen3Model

from wyat import tasks
from wyat.backends.kai.encoding import load_tokenizer
from wyat.backends.kai.torch import CandidateHead

FORMAT = "wyat-native-full-v1"
MANIFEST = "native_manifest.json"
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
)
DOCUMENTATION_FILES = frozenset(
    {"README.md", "LICENSE", "NOTICE", "ATTRIBUTION.md", ".gitattributes"}
)
DOWNLOAD_METADATA = ".cache/huggingface/"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_contract() -> dict[str, object]:
    """Bind encoding/readout and shared-head code, without recording local paths."""
    import inspect

    return {
        "recipe": tasks.RECIPE,
        "tasks_sha256": file_sha256(Path(tasks.__file__)),
        "predictor_sha256": hashlib.sha256(
            inspect.getsource(CandidateHead).encode("utf-8")
        ).hexdigest(),
        "adapter_math": "base(x) + 2 * linear(linear(float32(x), down), up)",
        "independent_question_rows": True,
        "padding_attention_mask": False,
        "backbone_storage": ["torch.float32", "torch.bfloat16"],
        "cpu_bf16_policy": "lossless-fp32-promotion",
    }


def _positive_integer(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_artifact(directory: str | Path) -> dict[str, Any]:
    """Reject missing, changed and undeclared artifact files before tensor loading."""
    root = Path(directory).resolve()
    if (root / MANIFEST).is_symlink():
        raise ValueError("native manifest must be a local regular file")
    manifest = json.loads((root / MANIFEST).read_text())
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise ValueError("unsupported native artifact format")
    if (
        manifest.get("production_kai_compatible") is not False
        or manifest.get("resume_supported") is not False
    ):
        raise ValueError(
            "native artifact must declare distinct inference and no resume"
        )
    if not isinstance(manifest.get("model_id"), str) or not manifest[
        "model_id"
    ].startswith("native-"):
        raise ValueError("native artifact requires a model identity")
    if not isinstance(manifest.get("provenance"), dict):
        raise ValueError("native artifact provenance must be an object")
    if manifest.get("contract") != artifact_contract():
        raise ValueError("native artifact task/predictor contract differs")
    if manifest.get("adapter") != {
        "rank": 16,
        "alpha": 32,
        "targets": ["q_proj", "v_proj"],
        "merged": False,
    }:
        raise ValueError("unsupported native adapter contract")
    maximum = _positive_integer(manifest.get("max_length"), "max_length")
    budget = _positive_integer(manifest.get("max_batch_tokens"), "max_batch_tokens")
    if budget < maximum:
        raise ValueError("batch token budget must admit one complete row")
    _positive_integer(manifest.get("head_dim"), "head_dim")
    files = manifest.get("files_sha256")
    required = {
        "backbone/config.json",
        "backbone/model.safetensors",
        "decision_head.safetensors",
        "native-adapters.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    if not isinstance(files, dict) or not required <= files.keys():
        raise ValueError("native artifact lacks required file identities")
    identity = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    if manifest["model_id"] != "native-" + identity:
        raise ValueError("native model identity differs from artifact files")
    permitted = required | set(TOKENIZER_FILES)
    if not set(files) <= permitted:
        raise ValueError("native artifact contains unsupported files")
    actual = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    # Published repositories carry documentation beside the weights, and
    # `hf download --local-dir` writes its own metadata. Neither is ever read.
    actual = {
        name
        for name in actual
        if name not in DOCUMENTATION_FILES and not name.startswith(DOWNLOAD_METADATA)
    }
    if actual != set(files) | {MANIFEST}:
        raise ValueError("native artifact file inventory differs")
    for name, expected in files.items():
        path = root / name
        if (
            path.is_symlink()
            or not path.is_file()
            or path.resolve().parent not in {root, root / "backbone"}
        ):
            raise ValueError("native artifact files must be local regular files")
        if file_sha256(path) != expected:
            raise ValueError("native artifact file identity differs: " + name)
    return manifest


def _finite_state(state: dict[str, torch.Tensor]) -> None:
    for name, value in state.items():
        if not value.is_floating_point():
            raise ValueError("native weights must be floating point: " + name)
        flattened = value.reshape(-1)
        for start in range(0, flattened.numel(), 1 << 20):
            if not bool(torch.isfinite(flattened[start : start + (1 << 20)]).all()):
                raise ValueError("nonfinite native weight: " + name)


class _Adapter(nn.Module):
    """Execute the exact unmerged FP32 rank-16 training projection."""

    def __init__(self, base: nn.Linear) -> None:
        super().__init__()
        self.base = base
        self.down = nn.Parameter(torch.empty(16, base.in_features))
        self.up = nn.Parameter(torch.empty(base.out_features, 16))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return cast(
            torch.Tensor,
            self.base(value)
            + 2 * F.linear(F.linear(value.float(), self.down), self.up),
        )


class NativeModel:
    """One shared predictor with Wyat's Choice, Score and Noul readouts."""

    def __init__(
        self,
        backbone: Qwen3Model,
        head: CandidateHead,
        tokenizer: Any,
        manifest: dict[str, Any],
        device: torch.device,
        max_length: int,
    ) -> None:
        self.backbone = backbone
        cast(nn.Module, backbone).eval()
        self.head = head.eval()
        self.tokenizer = tokenizer
        self.manifest = manifest
        self.device = device
        self.max_length = max_length
        self.max_batch_tokens = manifest["max_batch_tokens"]
        self.model_id = manifest["model_id"]
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = tokenizer.eos_token_id
        if type(self.pad_id) is not int or self.pad_id < 0:
            raise ValueError("tokenizer requires a nonnegative pad or EOS token")

    def encode_record(self, record: object) -> tuple[tasks.EncodedQuestion, ...]:
        return tasks.encode_record(self.tokenizer, record, max_length=self.max_length)

    def _logits(self, encoded: Sequence[tasks.EncodedQuestion]) -> list[list[float]]:
        length = max(len(row.ids) for row in encoded)
        if length * len(encoded) > self.max_batch_tokens:
            raise ValueError("complete padded question batch exceeds explicit budget")
        ids = torch.full(
            (len(encoded), length), self.pad_id, dtype=torch.long, device=self.device
        )
        for index, row in enumerate(encoded):
            ids[index, : len(row.ids)] = torch.tensor(row.ids, device=self.device)
        autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            hidden = self.backbone(input_ids=ids, use_cache=False).last_hidden_state
            result = []
            for index, row in enumerate(encoded):
                candidates = hidden[index : index + 1, row.candidate_positions, :]
                query = hidden[index : index + 1, row.query_position, :]
                result.append(self.head(candidates, query)[0].float().cpu().tolist())
        return result

    def logits(self, record: object) -> list[list[float]]:
        """Return one complete ordered FP32 logit vector per public question."""
        return self._logits(self.encode_record(record))

    def predict(
        self, record: object
    ) -> dict[str, tasks.ChoiceAnswer | tasks.ScoreAnswer | tasks.NoulAnswer]:
        encoded = self.encode_record(record)
        return {
            row.question.id: tasks.interpret(row.question, logits)
            for row, logits in zip(encoded, self._logits(encoded), strict=True)
        }


def load_native(
    directory: str | Path, *, device: str = "cpu", max_length: int | None = None
) -> NativeModel:
    """Load an Axes export entirely locally; no remote code, downloads or resume."""
    root = Path(directory).resolve()
    manifest = validate_artifact(root)
    target = torch.device(device)
    if target.type not in {"cpu", "cuda"}:
        raise ValueError("native inference supports cpu and cuda")
    if target.type == "cuda" and (
        not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("native CUDA inference requires BF16 support")
    maximum = (
        manifest["max_length"]
        if max_length is None
        else _positive_integer(max_length, "max_length")
    )
    if maximum > manifest["max_length"]:
        raise ValueError("max_length exceeds the artifact contract")
    raw = json.loads((root / "backbone/config.json").read_text())
    if not isinstance(raw, dict) or raw.get("model_type") != "qwen3":
        raise ValueError("native artifact requires a Qwen3 backbone")
    config = Qwen3Config.from_dict(raw)
    config._attn_implementation = "sdpa"
    backbone = Qwen3Model(config)
    base = load_file(str(root / "backbone/model.safetensors"))
    _finite_state(base)
    if any(
        value.dtype not in {torch.float32, torch.bfloat16} for value in base.values()
    ):
        raise ValueError("native backbone requires FP32 or BF16 storage")
    # CUDA preserves the historical mixed BF16/FP32 storage. CPU promotes BF16
    # values losslessly, as the released Kai CPU loader does; GPU arithmetic
    # equivalence is a separate validation and is never implied by this load.
    if target.type == "cpu":
        base = {name: value.float() for name, value in base.items()}
    backbone.load_state_dict(base, strict=True, assign=True)
    for layer in backbone.layers:
        for name in ("q_proj", "v_proj"):
            setattr(layer.self_attn, name, _Adapter(getattr(layer.self_attn, name)))
    state = load_file(str(root / "native-adapters.safetensors"))
    _finite_state(state)
    adapters = {
        name: parameter
        for name, parameter in backbone.named_parameters()
        if name.endswith((".down", ".up"))
    }
    if set(state) != set(adapters):
        raise ValueError("native adapter tensor inventory differs")
    with torch.no_grad():
        for name, value in state.items():
            if value.dtype != torch.float32 or value.shape != adapters[name].shape:
                raise ValueError("invalid native adapter tensor: " + name)
            adapters[name].copy_(value)
    head = CandidateHead(config.hidden_size, manifest["head_dim"])
    head_state = load_file(str(root / "decision_head.safetensors"))
    _finite_state(head_state)
    if any(value.dtype != torch.float32 for value in head_state.values()):
        raise ValueError("native shared head requires FP32 weights")
    head.load_state_dict(head_state, strict=True)
    backbone.config.use_cache = False
    backbone.requires_grad_(False)
    head.requires_grad_(False)
    return NativeModel(
        cast(Qwen3Model, cast(nn.Module, backbone).to(target)),
        head.to(target),
        load_tokenizer(root),
        manifest,
        target,
        maximum,
    )
