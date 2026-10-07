from __future__ import annotations

import argparse
import csv
import importlib
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
VQPY_ROOT = Path("D:/pycharm/vqpy-main")
EVA_ROOT = Path("D:/pycharm/pythonProject1")
VOCAL_UDF_ROOT = Path("D:/pycharm/VOCAL-UDF")
VIDEO_COLBERT_ROOT = Path("D:/pycharm/Video-ColBERT")
SEIDEN_ROOT = Path("D:/pycharm/seiden_style_baseline")
LAVA_ROOT = Path("D:/pycharm/LAVA")
DEFAULT_THETA_N_PARTS = 10
DEFAULT_THETA_D_PARTS = 8


def _import_from_root(root: Path, module_name: str):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return importlib.import_module(module_name)


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_query_specs(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _spatial_constraint_scope(
    pair_constraints: dict[tuple[int, int], tuple[float, float]],
    theta_n_parts: int,
    theta_d_parts: int,
) -> str:
    if not pair_constraints:
        return "none"
    covers_all_pairs = all(
        theta_lt >= theta_n_parts and d_lt >= theta_d_parts
        for theta_lt, d_lt in pair_constraints.values()
    )
    return "full_relation" if covers_all_pairs else "restricted_relation"


def _result_to_row(
    baseline: str,
    spec: dict[str, str],
    result: Any,
    pair_constraints: dict[tuple[int, int], tuple[float, float]],
    theta_n_parts: int,
    theta_d_parts: int,
) -> dict[str, Any]:
    top1 = result.matches[0] if result.matches else None
    timing = result.timing
    load_s = getattr(timing, "load_s", 0.0)
    candidate_s = getattr(timing, "candidate_s", 0.0)
    scan_s = getattr(timing, "scan_s", None)
    if scan_s is None:
        scan_s = getattr(timing, "relational_scan_s", 0.0)
    total_s = float(timing.total_s)
    query_s_excluding_load = max(0.0, total_s - float(load_s))

    return {
        "baseline": baseline,
        "query_id": spec["query_id"],
        "dataset": spec["dataset"],
        "types": spec["types"],
        "pair_constraints": spec["pair_constraints"],
        "frame_threshold": int(spec["frame_threshold"]),
        "topk": int(spec["topk"]),
        "temporal_mode": spec["temporal_mode"],
        "match_count": len(result.matches),
        "top1_binding": "" if top1 is None else ",".join(map(str, top1.binding)),
        "top1_frame_count": 0 if top1 is None else int(top1.frame_count),
        "top1_frames": "" if top1 is None else ",".join(map(str, sorted(top1.frames))),
        "load_s": float(load_s),
        "candidate_s": float(candidate_s),
        "scan_s": float(scan_s),
        "aggregate_s": float(timing.aggregate_s),
        "total_s": total_s,
        "query_s_excluding_load": query_s_excluding_load,
        "spatial_constraint_scope": _spatial_constraint_scope(
            pair_constraints,
            theta_n_parts=theta_n_parts,
            theta_d_parts=theta_d_parts,
        ),
        "note": spec.get("note", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser("Run batch benchmarks for structured-query baselines")
    parser.add_argument(
        "--query_specs",
        default=str(REPO_ROOT / "supplement" / "baseline_formal_queries.csv"),
        help="CSV file with query specs",
    )
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "supplement" / "out" / "baseline_batch_benchmark.csv"),
        help="output CSV path",
    )
    parser.add_argument(
        "--baselines",
        default="vqpy,eva",
        help="comma-separated baseline names from {vqpy,eva,vocal_udf,video_colbert,seiden,lava}",
    )
    parser.add_argument(
        "--query_ids",
        default="",
        help="optional comma-separated query_id filter",
    )
    parser.add_argument("--theta_n_parts", type=int, default=DEFAULT_THETA_N_PARTS)
    parser.add_argument("--theta_d_parts", type=int, default=DEFAULT_THETA_D_PARTS)
    args = parser.parse_args()

    query_specs = _read_query_specs(Path(args.query_specs))
    query_id_filter = {token.strip() for token in args.query_ids.split(",") if token.strip()}
    if query_id_filter:
        query_specs = [spec for spec in query_specs if spec["query_id"] in query_id_filter]
    baseline_names = [token.strip() for token in args.baselines.split(",") if token.strip()]

    module_specs = {
        "vqpy": (VQPY_ROOT, "offline_structured_query"),
        "eva": (EVA_ROOT, "evadb_structured_query_baseline"),
        "vocal_udf": (VOCAL_UDF_ROOT, "vocal_udf_structured_adapter"),
        "video_colbert": (VIDEO_COLBERT_ROOT, "video_colbert_structured_adapter"),
        "seiden": (SEIDEN_ROOT, "seiden_structured_query"),
        "lava": (LAVA_ROOT, "lava_structured_adapter"),
    }
    modules = {
        baseline: _import_from_root(*module_specs[baseline])
        for baseline in baseline_names
        if baseline in module_specs
    }

    rows: list[dict[str, Any]] = []
    total_runs = len(query_specs) * len(baseline_names)
    current = 0
    for spec in query_specs:
        query_types = [int(token.strip()) for token in spec["types"].split(",") if token.strip()]
        for baseline in baseline_names:
            current += 1
            print(
                f"[{current}/{total_runs}] baseline={baseline} query_id={spec['query_id']} dataset={spec['dataset']}",
                flush=True,
            )
            if baseline not in modules:
                raise ValueError(f"unsupported baseline '{baseline}'")
            module = modules[baseline]

            pair_constraints = module.parse_pair_constraints(spec["pair_constraints"], role_count=len(query_types))
            spatial_scope = _spatial_constraint_scope(
                pair_constraints,
                theta_n_parts=args.theta_n_parts,
                theta_d_parts=args.theta_d_parts,
            )
            if spatial_scope == "full_relation":
                print(
                    "  warning=spatial constraints cover the full configured relation range; "
                    "query runtime is dominated by object binding enumeration",
                    flush=True,
                )
            data_path, frame_width, frame_height = module._resolve_dataset_path(  # type: ignore[attr-defined]
                dataset=spec["dataset"],
                data_path=None,
                storage_root=module.DEFAULT_STORAGE_ROOT,
            )
            result = module.run_query_from_path(
                data_path=data_path,
                query_types=query_types,
                pair_constraints=pair_constraints,
                frame_threshold=int(spec["frame_threshold"]),
                frame_width=frame_width,
                frame_height=frame_height,
                temporal_mode=spec["temporal_mode"],
                topk=int(spec["topk"]),
                theta_n_parts=args.theta_n_parts,
                theta_d_parts=args.theta_d_parts,
            )
            row = _result_to_row(
                baseline,
                spec,
                result,
                pair_constraints=pair_constraints,
                theta_n_parts=args.theta_n_parts,
                theta_d_parts=args.theta_d_parts,
            )
            rows.append(row)
            print(
                f"  total={row['total_s']:.4f}s match_count={row['match_count']} "
                f"top1_frame_count={row['top1_frame_count']}",
                flush=True,
            )

    out_path = Path(args.out)
    _ensure_parent_dir(out_path)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved_csv={out_path}", flush=True)


if __name__ == "__main__":
    main()
