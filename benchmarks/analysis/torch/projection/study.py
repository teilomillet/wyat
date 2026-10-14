"""Authentic layer-zero operands; no backbone construction or model speed claim."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import time
from importlib.metadata import version
from pathlib import Path
from typing import Protocol, cast

import torch
from safetensors import safe_open
from torch import nn
from transformers.models.qwen3 import modeling_qwen3

from benchmarks.analysis.evidence import freeze_sources
from benchmarks.analysis.torch.autotune import retain_tensors
from benchmarks.analysis.torch.prefix.production import study as original
from wyat.evaluation import percentile
from wyat.wire import read_object

from .qkv import Arm, Outputs, PackedQKV, quality

ROOT = Path(__file__).resolve().parents[4]
ARMS: tuple[Arm, ...] = ("separate", "kv", "qkv")
ORDERS = tuple(itertools.permutations(ARMS))
PROFILES = (
    ("scalar192", (7,)),
    ("scalar197", (18,)),
    ("scalar214", (75,)),
    ("batch8", tuple(range(8))),
)


class TensorSlice(Protocol):
    def __getitem__(self, rows: slice) -> torch.Tensor: ...


def tensor_sha(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().numpy().tobytes()).hexdigest()


def source_paths() -> dict[str, Path]:
    paths = {str(p.relative_to(ROOT)): p for p in Path(__file__).parent.glob("*.py")}
    for name in (
        "tests/analysis/cpu/projection/test_qkv.py",
        "docs/analysis/cpu/projection.md",
        "benchmarks/analysis/evidence.py",
        "benchmarks/analysis/torch/autotune.py",
        "benchmarks/analysis/torch/process.py",
        "benchmarks/analysis/torch/prefix/production/study.py",
        "wyat/evaluation.py",
        "wyat/wire.py",
        "wyat/backends/wyat/encoding.py",
        "wyat/backends/wyat/torch.py",
    ):
        paths[name] = ROOT / name
    paths["maintained_qwen3"] = Path(modeling_qwen3.__file__)
    paths["maintained_functional"] = Path(torch.nn.functional.__file__)
    return paths


def input_vectors() -> list[dict[str, object]]:
    original.verify_files(
        {original.DATA: original.DATA_SHA, original.REFERENCE: original.REFERENCE_SHA}
    )
    cases = [json.loads(line) for line in original.DATA.read_text().splitlines()]
    encoded = json.dumps(
        cases,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    if len(cases) != 77 or hashlib.sha256(encoded).hexdigest() != original.INPUT_SHA:
        raise ValueError("Exact ordered77 canonical requests required")
    reference = read_object(original.REFERENCE)
    rows = cast(list[dict[str, object]], reference["rows"])
    if len(rows) != 77 or reference["model_id"] != original.MODEL_ID:
        raise ValueError("Frozen model/reference identity differs")
    prefix = cast(list[int], rows[0]["token_ids"])[:25]
    for case, row in zip(cases, rows, strict=True):
        tokens = cast(list[int], row["token_ids"])
        positions = cast(list[int], row["candidate_positions"])
        ids = [item["id"] for item in case["request"]["candidates"]]
        if (
            row["id"] != case["id"]
            or row["candidate_ids"] != ids
            or len(ids) != 6
            or not 25 < len(tokens) <= 512
            or any(type(token) is not int for token in tokens)
            or tokens[:25] != prefix
            or row["query_position"] != len(tokens) - 1
            or len(positions) != 6
            or any(type(p) is not int or not 25 <= p < len(tokens) for p in positions)
        ):
            raise ValueError("Frozen token/readout/candidate geometry differs")
    if tuple(len(cast(list[int], rows[i]["token_ids"])) for i in (7, 18, 75)) != (
        192,
        197,
        214,
    ):
        raise ValueError("Predeclared real operand lengths differ")
    return rows


def padding_id(model: Path) -> int:
    config = read_object(model / "tokenizer_config.json")
    tokenizer = read_object(model / "tokenizer.json")
    name = config["pad_token"] or config["eos_token"]
    added = cast(list[dict[str, object]], tokenizer["added_tokens"])
    matches = [item["id"] for item in added if item["content"] == name]
    if matches != [151643]:
        raise ValueError("Released pad/EOS token identity differs")
    return 151643


def operands(
    model: Path, rows: list[dict[str, object]], pad: int
) -> tuple[
    PackedQKV, list[tuple[str, torch.Tensor, dict[str, object]]], dict[str, object]
]:
    config = read_object(model / "backbone/config.json")
    if (
        config["hidden_size"],
        config["num_attention_heads"],
        config["num_key_value_heads"],
        config["head_dim"],
        config["rms_norm_eps"],
    ) != (1024, 16, 8, 128, 1e-6):
        raise ValueError("Released projection/normalization geometry differs")
    result: list[tuple[str, torch.Tensor, dict[str, object]]] = []
    identities: dict[str, object] = {}
    with safe_open(
        model / "backbone/model.safetensors", framework="pt", device="cpu"
    ) as reader:
        source = tuple(
            reader.get_tensor(f"layers.0.self_attn.{name}_proj.weight").float()
            for name in ("q", "k", "v")
        )
        for name, value in zip(("q", "k", "v"), source, strict=True):
            identities[name] = {
                "shape": list(value.shape),
                "sha256": tensor_sha(value),
                "dtype": str(value.dtype),
            }
        packed = PackedQKV.from_weights(*source)
        if not all(
            torch.equal(a, b) for a, b in zip(source, packed.views, strict=True)
        ):
            raise ValueError("Packing changed a weight bit")
        del source
        norm = modeling_qwen3.Qwen3RMSNorm(1024, eps=1e-6)
        norm.weight = nn.Parameter(
            reader.get_tensor("layers.0.input_layernorm.weight").float(),
            requires_grad=False,
        )
        norm.eval()
        identities["norm"] = {
            "sha256": tensor_sha(norm.weight),
            "epsilon": norm.variance_epsilon,
        }
        embedding = cast(TensorSlice, reader.get_slice("embed_tokens.weight"))
        for label, indices in PROFILES:
            tokens = [cast(list[int], rows[i]["token_ids"]) for i in indices]
            length = max(map(len, tokens))
            ids = torch.full((len(indices), length), pad, dtype=torch.long)
            for index, values in enumerate(tokens):
                ids[index, : len(values)] = torch.tensor(values)
            # Read only selected embedding rows; never materialize the vocabulary.
            unique = {
                int(token): embedding[int(token) : int(token) + 1].float()
                for token in set(int(t) for t in ids.flatten())
            }
            full = torch.cat([unique[int(token)] for token in ids.flatten()]).reshape(
                len(indices), length, 1024
            )
            value = cast(torch.Tensor, norm(full))[:, 25:].contiguous()
            if not torch.isfinite(value).all().item():
                raise ValueError("Authentic normalized operand is nonfinite")
            result.append(
                (
                    label,
                    value,
                    {
                        "case_offsets": list(indices),
                        "case_ids": [rows[i]["id"] for i in indices],
                        "full_lengths": [len(t) for t in tokens],
                        "padded_full_length": length,
                        "pad_id": pad,
                        "prefix_length": 25,
                        "input_ids_sha256": tensor_sha(ids),
                        "token_ids": [list(t) for t in tokens],
                        "shape": list(value.shape),
                        "strides": list(value.stride()),
                        "activation_sha256": tensor_sha(value),
                    },
                )
            )
    return packed, result, identities


def study(model: Path, output: Path) -> bool:
    parent_sidecars = {output.with_suffix(".wall.json"), output.with_suffix(".log")}
    if output.exists() or any(
        path not in parent_sidecars for path in output.parent.glob(output.stem + ".*")
    ):
        raise FileExistsError("Preserve existing primitive receipts")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "completed": False,
        "quality_passed": False,
        "primitive_eligible": {"kv": False, "qkv": False},
        "accepted": False,
        "phase": "admission",
        "operands": [],
        "protocol": {
            "profiles": [[name, list(indices)] for name, indices in PROFILES],
            "orders": [list(order) for order in ORDERS],
            "arms": list(ARMS),
            "cold_calls": 1,
            "warm_calls": 1,
            "measured_calls_per_arm": 6,
            "total_invocations": 96,
            "threads": 8,
            "dtype": "float32",
            "device": "cpu",
            "projection_gate": {
                "atol": 1e-3,
                "rtol": 1e-3,
                "rule": "symmetric max, not additive",
            },
            "eligibility": {
                "all_four_operands": True,
                "median_minimum": 1.12,
                "p05_exclusive": 1.0,
            },
            "timing": (
                "One project() call including F.linear, split and tuple validation; "
                "no observer/I/O/quality/setup inside clock"
            ),
            "scope": (
                "Layer0 projection only; no attention/head/rank/probability/"
                "model speed or promotion"
            ),
        },
    }

    def write() -> None:
        output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")

    with output.open("x") as handle:
        handle.write(json.dumps(payload, indent=2) + "\n")
    previous = torch.get_num_threads()
    restored = False

    def restore() -> None:
        nonlocal restored
        restored = True
        torch.set_num_threads(previous)
        if torch.get_num_threads() != previous:
            raise ValueError("Original thread count not restored")
        payload["threads_restored"] = True

    try:
        rows = input_vectors()
        assets = {model / name: sha for name, sha in original.MODEL_FILES.items()}
        assets |= {
            original.DATA: original.DATA_SHA,
            original.REFERENCE: original.REFERENCE_SHA,
        }
        original.verify_files(assets)
        sources = freeze_sources(source_paths(), output)
        policy = original.process_policy()
        if policy["cpu_autocast"] is not False:
            raise ValueError("CPU autocast must be disabled")
        payload.update(
            asset_sha256={str(p): sha for p, sha in assets.items()},
            source_sha256=sources.hashes(),
            source_archive=sources.archive(),
            process_policy=policy,
            versions={
                name: version(name) for name in ("torch", "transformers", "safetensors")
            },
            torch_build=torch.__config__.show(),
            prior_threads=previous,
            phase="setup",
        )
        write()
        torch.set_num_threads(8)
        with torch.inference_mode():
            packed, inputs, identities = operands(model, rows, padding_id(model))
            payload.update(
                weight_identities=identities,
                retained_projection_bytes=packed.weight.numel() * 4,
                setup_projection_logical_peak_bytes=packed.weight.numel() * 8,
                hypothetical_all28_projection_bytes=packed.weight.numel() * 4 * 28,
                ownership=(
                    "One packed storage, separate Q/K/V and packedKV views alias it; "
                    "setup originals released; no RSS claim"
                ),
            )
            observations: list[dict[str, object]] = []
            payload["operands"] = observations
            eligibility: dict[str, list[bool]] = {"kv": [], "qkv": []}
            for label, value, metadata in inputs:
                entry: dict[str, object] = {
                    "name": label,
                    "input": metadata,
                    "calls": [],
                    "completed": False,
                }
                observations.append(entry)
                calls = cast(list[dict[str, object]], entry["calls"])
                retained: dict[str, torch.Tensor] = {
                    "input": value,
                    "packed_weight": packed.weight,
                }
                raw_path = output.with_name(
                    output.stem + "." + label + ".tensors.safetensors"
                )
                reference: Outputs | None = None
                measured: dict[Arm, list[float]] = {arm: [] for arm in ARMS}
                schedule = [
                    (stage, -1, arm) for stage in ("cold", "warm") for arm in ARMS
                ]
                schedule += [
                    ("measured", repeat, arm)
                    for repeat, order in enumerate(ORDERS)
                    for arm in order
                ]
                entry["pending_calls"] = len(schedule)
                for stage, repeat, arm in schedule:
                    original.verify_sources(sources)
                    if (
                        original.process_policy() != policy
                        or torch.get_num_threads() != 8
                    ):
                        raise ValueError("Owned process policy changed")
                    started = time.perf_counter_ns()
                    actual = packed.project(value, arm)
                    elapsed = (time.perf_counter_ns() - started) / 1e6
                    hashes = [tensor_sha(t) for t in actual]
                    changed = False
                    for name, tensor, checksum in zip(
                        ("q", "k", "v"), actual, hashes, strict=True
                    ):
                        key = name + "-" + checksum
                        if key not in retained:
                            retained[key] = tensor.detach().contiguous().clone()
                            changed = True
                    if changed:
                        entry.update(
                            raw_tensors=str(raw_path),
                            raw_sha256=retain_tensors(retained, raw_path),
                        )
                    calls.append(
                        {
                            "stage": stage,
                            "repeat": repeat,
                            "arm": arm,
                            "latency_ms": elapsed if math.isfinite(elapsed) else None,
                            "invalid_latency_repr": None
                            if math.isfinite(elapsed)
                            else repr(elapsed),
                            "output_sha256": hashes,
                            "output_shapes": [list(t.shape) for t in actual],
                            "output_strides": [list(t.stride()) for t in actual],
                            "quality": None,
                        }
                    )
                    entry["pending_calls"] = len(schedule) - len(calls)
                    write()
                    gate = quality(actual if reference is None else reference, actual)
                    calls[-1]["quality"] = gate
                    write()
                    if not gate["passed"] or not math.isfinite(elapsed) or elapsed <= 0:
                        raise ValueError(
                            "Projection numerical/finite/timing gate failed"
                        )
                    if reference is None:
                        reference = actual
                    if stage == "measured":
                        measured[arm].append(elapsed)
                ratios = {
                    arm: [
                        a / b
                        for a, b in zip(
                            measured["separate"], measured[arm], strict=True
                        )
                    ]
                    for arm in cast(tuple[Arm, ...], ("kv", "qkv"))
                }
                stats = {
                    arm: {
                        "ratios": values,
                        "median": percentile(values, 0.5),
                        "p05": percentile(values, 0.05),
                    }
                    for arm, values in ratios.items()
                }
                for arm in cast(tuple[Arm, ...], ("kv", "qkv")):
                    eligibility[arm].append(
                        cast(float, stats[arm]["median"]) >= 1.12
                        and cast(float, stats[arm]["p05"]) > 1
                    )
                entry.update(completed=True, measured_ms=measured, statistics=stats)
                write()
        original.verify_sources(sources)
        original.verify_files(assets)
        if original.process_policy() != policy or torch.get_num_threads() != 8:
            raise ValueError("Owned process policy changed")
        restore()
        payload.update(
            completed=True,
            quality_passed=True,
            primitive_eligible={
                arm: all(values) and len(values) == 4
                for arm, values in eligibility.items()
            },
            source_verified_after=True,
            asset_verified_after=True,
            phase="complete",
        )
        write()
        return True
    except BaseException as error:
        payload.update(
            completed=False,
            quality_passed=False,
            primitive_eligible={"kv": False, "qkv": False},
            error=f"{type(error).__name__}: {error}",
            phase="failed",
        )
        write()
        raise
    finally:
        if not restored:
            try:
                restore()
                write()
            except BaseException as error:
                payload.update(
                    completed=False,
                    quality_passed=False,
                    primitive_eligible={"kv": False, "qkv": False},
                    thread_restoration_error=f"{type(error).__name__}: {error}",
                )
                write()
                raise
