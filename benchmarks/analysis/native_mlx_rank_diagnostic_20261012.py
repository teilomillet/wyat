"""Capture one frozen case, then isolate CPU-head and backbone numerical drift.

CPU and head commands require the pinned release PYTHONPATH and an outside-repo
working directory. Run captures sequentially; head loads no backbone model.
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any

import native_mlx_benchmark_20261012 as bench  # type: ignore[import-not-found, unused-ignore]


def pinned_runtime(args: argparse.Namespace) -> None:
    from wyat.loading import native

    root = args.runtime.resolve()
    if not Path(native.__file__).resolve().is_relative_to(root):
        raise RuntimeError("CPU imported outside pinned release runtime")
    for relative, expected in bench.read(root / "SOURCE.json")["files_sha256"].items():
        if bench.digest(root / relative) != expected:
            raise RuntimeError("pinned source changed: " + relative)


def frozen_rows(model: Any, frozen: dict[str, Any], case_id: str) -> tuple[Any, ...]:
    case = next(case for case in frozen["cases"] if case["id"] == case_id)
    records = {record["id"]: record for record in frozen["records"]}
    rows = []
    for record_id in case["records"]:
        record = records[record_id]
        encoded = model.encode_record(record["public_record"])
        if list(map(bench.encoded_value, encoded)) != record["encoded"]:
            raise RuntimeError("frozen encoding differs")
        rows.extend(encoded)
    if len(rows) != 1:
        raise ValueError("diagnostic captures one independent question row only")
    return tuple(rows)


def capture(args: argparse.Namespace, result: dict[str, Any]) -> None:
    import numpy as np

    frozen = bench.read(args.frozen)
    bench.check_frozen(frozen)
    args.arm = "cpu" if args.command == "cpu" else args.mlx_arm
    model = bench.load_model(args, metal=args.arm == "metal")
    rows = frozen_rows(model, frozen, args.case)
    row = rows[0]
    result.update(
        arm=args.arm,
        model_id=model.model_id,
        artifact_manifest_sha256=bench.digest(args.artifact / "native_manifest.json"),
        metadata=bench.metadata(),
        encoded=bench.encoded_value(row),
        metal_enabled=getattr(model, "metal_enabled", False),
        head_sha256=bench.digest(args.artifact / "decision_head.safetensors"),
    )
    bench.checkpoint(args.output, result)
    if args.command == "cpu":
        import torch

        with torch.inference_mode():
            ids = torch.tensor([row.ids], dtype=torch.long, device="cpu")
            hidden = model.backbone(input_ids=ids, use_cache=False).last_hidden_state
            candidates = hidden[:, row.candidate_positions, :].float().contiguous()
            query = hidden[:, row.query_position, :].float().contiguous()
            logits = model.head(candidates, query).float().cpu().tolist()
            candidate_values = candidates.cpu().numpy().copy()
            query_values = query.cpu().numpy().copy()
    else:
        import mlx.core as mx

        with model._inference_lock, mx.stream(model.device):
            mlx_ids = mx.array([row.ids], dtype=mx.int32)
            hidden = model._hidden(mlx_ids)
            candidates = hidden[:, mx.array(row.candidate_positions), :]
            query = hidden[:, row.query_position, :]
            outputs = model.head(candidates, query)
            mx.eval(candidates, query, outputs)
            logits = outputs.tolist()
            candidate_values = np.asarray(candidates, dtype=np.float32).copy()
            query_values = np.asarray(query, dtype=np.float32).copy()
        mx.synchronize()
    if not np.isfinite(candidate_values).all() or not np.isfinite(query_values).all():
        raise RuntimeError("captured hidden states are nonfinite")
    archive = args.output.with_suffix(".npz")
    with archive.open("xb") as stream:
        np.savez(stream, candidates=candidate_values, query=query_values)
    result.update(
        archive=str(archive.resolve()),
        archive_sha256=bench.digest(archive),
        hidden_bytes=candidate_values.nbytes + query_values.nbytes,
        outputs=bench.snapshot(rows, logits),
        memory=bench.memory(args.arm),
        error=None,
    )


def head_only(args: argparse.Namespace, result: dict[str, Any]) -> None:
    import hashlib

    import numpy as np
    import torch
    from safetensors.torch import load_file

    from wyat.backends.wyat.torch import CandidateHead

    pinned_runtime(args)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    manifest = bench.read(args.artifact / "native_manifest.json")
    config = bench.read(args.artifact / "backbone/config.json")
    head_path = args.artifact / "decision_head.safetensors"
    head_hash = bench.digest(head_path)
    if head_hash != manifest["files_sha256"]["decision_head.safetensors"]:
        raise RuntimeError("head artifact hash differs")
    predictor = hashlib.sha256(inspect.getsource(CandidateHead).encode()).hexdigest()
    if predictor != manifest["contract"]["predictor_sha256"]:
        raise RuntimeError("head source identity differs")
    head = CandidateHead(config["hidden_size"], manifest["head_dim"]).eval()
    state = load_file(str(head_path), device="cpu")
    if any(value.dtype != torch.float32 for value in state.values()):
        raise RuntimeError("head is not entirely FP32")
    head.load_state_dict(state, strict=True)
    frozen = bench.read(args.frozen)
    bench.check_frozen(frozen)
    captures = {}
    recomputed = {}
    arrays = {}
    from wyat import tasks

    for name, path in (("cpu", args.cpu_capture), ("mlx", args.mlx_capture)):
        item = bench.read(path)
        if item.get("error") is not None:
            raise RuntimeError("failed capture: " + name)
        if (
            item["frozen_sha256"] != bench.digest(args.frozen)
            or item["case"] != args.case
        ):
            raise RuntimeError("capture cohort/case differs")
        if item["head_sha256"] != head_hash:
            raise RuntimeError("capture head identity differs")
        archive = Path(item["archive"])
        if bench.digest(archive) != item["archive_sha256"]:
            raise RuntimeError("capture archive identity differs")
        with np.load(archive, allow_pickle=False) as loaded:
            candidates = loaded["candidates"].copy()
            query = loaded["query"].copy()
        width = len(item["encoded"]["candidate_positions"])
        if candidates.dtype != np.float32 or query.dtype != np.float32:
            raise RuntimeError("capture storage dtype differs")
        if candidates.shape != (1, width, config["hidden_size"]) or query.shape != (
            1,
            config["hidden_size"],
        ):
            raise RuntimeError("captured hidden shape differs")
        with torch.inference_mode():
            logits = head(torch.from_numpy(candidates), torch.from_numpy(query))[
                0
            ].tolist()
        original = item["outputs"][0]
        question = item["encoded"]["question"]
        parsed = tasks.Question(
            id=question["id"],
            type=question["type"],
            instructions=question["instructions"],
            options=tuple(tuple(option) for option in question["options"]),
        )
        answer = tasks.interpret(parsed, logits)
        observed = dict(
            original,
            logits=list(answer.logits),
            probabilities=list(answer.probabilities),
            ranking=sorted(range(len(logits)), key=logits.__getitem__, reverse=True),
            answer=answer.to_dict(),
        )
        captures[name] = item
        recomputed[name] = [observed]
        arrays[name] = (candidates, query)
    if captures["cpu"]["encoded"] != captures["mlx"]["encoded"]:
        raise RuntimeError("CPU and MLX captured rows differ")
    gates = frozen["quality_gates"]
    result.update(
        metadata=bench.metadata(),
        head_sha256=head_hash,
        predictor_sha256=predictor,
        capture_sha256={
            "cpu": bench.digest(args.cpu_capture),
            "mlx": bench.digest(args.mlx_capture),
        },
        cpu_head_on_cpu_hidden=recomputed["cpu"],
        cpu_head_on_mlx_hidden=recomputed["mlx"],
        cpu_capture_vs_cpu_recomputed=bench.compare_rows(
            captures["cpu"]["outputs"], recomputed["cpu"], gates
        ),
        backbone_drift_same_cpu_head=bench.compare_rows(
            recomputed["cpu"], recomputed["mlx"], gates
        ),
        head_drift_same_mlx_hidden=bench.compare_rows(
            recomputed["mlx"], captures["mlx"]["outputs"], gates
        ),
        hidden_max_absolute={
            name: float(np.max(np.abs(arrays["cpu"][index] - arrays["mlx"][index])))
            for index, name in enumerate(("candidates", "query"))
        },
        error=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("cpu", "mlx", "head"))
    for name in ("artifact", "frozen", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--case", default="single-34")
    parser.add_argument("--mlx-arm", choices=("ordinary", "metal"), default="metal")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-limit-mib", type=int, default=6144)
    parser.add_argument("--cache-limit-mib", type=int, default=256)
    parser.add_argument("--cpu-capture", type=Path)
    parser.add_argument("--mlx-capture", type=Path)
    args = parser.parse_args()
    if args.command in {"cpu", "head"} and args.runtime is None:
        parser.error("pinned --runtime required")
    if args.command == "head" and (
        args.cpu_capture is None or args.mlx_capture is None
    ):
        parser.error("head requires --cpu-capture and --mlx-capture")
    if not 1 <= args.threads <= 4 or not 1 <= args.memory_limit_mib <= 6144:
        parser.error("use 1..4 CPU threads and 1..6144 MiB MLX memory guideline")
    if not 0 <= args.cache_limit_mib <= 256:
        parser.error("use 0..256 MiB MLX cache")
    for path in (
        args.output,
        args.output.with_suffix(".npz"),
        args.output.with_name(args.output.name + ".partial"),
    ):
        if path.exists():
            parser.error("output exists; choose fresh output")
    result = {
        "command": args.command,
        "case": args.case,
        "frozen_sha256": bench.digest(args.frozen),
        "diagnostic_sha256": bench.digest(Path(__file__)),
        "error": "not completed",
    }
    bench.checkpoint(args.output, result)
    try:
        (head_only if args.command == "head" else capture)(args, result)
    except Exception as error:
        result.update(error=repr(error), metadata=bench.metadata())
        bench.finish(args.output, result)
        raise
    bench.finish(args.output, result)


if __name__ == "__main__":
    main()
