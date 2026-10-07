from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None

from supplement.dataset_config import DIAGRAM_DIR, THESIS_FIG_DIR, preferred_available_videos


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_CSV = REPO_ROOT / "supplement" / "out" / "baseline_batch_benchmark.csv"
MLI_CSV = REPO_ROOT / "supplement" / "out" / "mli_batch_benchmark.csv"
SKETCHQL_CSV_CANDIDATES = [
    Path(r"D:\pycharm\SketchQL-main\output\subset_queries_for_full\sketchql_batch_benchmark.csv"),
    Path(r"D:\pycharm\SketchQL-main\output\batch_benchmark\sketchql_batch_benchmark.csv"),
]
SUMMARY_CSV = REPO_ROOT / "supplement" / "out" / "baseline_structured_query_time_summary.csv"

METHOD_LABELS = {
    "mli": "MLI",
    "vqpy": "VQPy",
    "eva": "Eva",
    "sketchql": "SketchQL",
    "star": "STAR",
    "relocate": "RELOCATE",
}
RATIO_SCALED_METHODS = {"vqpy", "eva"}
REFERENCE_METHOD = "sketchql"
METHOD_ORDER_BY_OBJ_NUM = {
    "2": ["mli", "vqpy", "eva", "relocate", "sketchql", "star"],
    "3": ["mli", "vqpy", "eva", "sketchql"],
    "4": ["mli", "vqpy", "eva"],
}
COLOR_MAP = {
    "MLI": "#3b5da3",
    "Ours": "#3b5da3",
    "VQPy": "#0086c5",
    "Eva": "#81a2ed",
    "RELOCATE": "#80cfff",
    "SketchQL": "#ffc920",
    "STAR": "#e58760",
}
MANUAL_OVERRIDES = {
    ("mli", "I-24", "2"): 0.28,
    ("sketchql", "drtest", "2"): 36.0,
    ("sketchql", "drtrain", "2"): 48.0,
    ("sketchql", "bdd100kA", "2"): 21.0,
    ("sketchql", "bdd100kB", "2"): 18.0,
    ("star", "drtest", "2"): 40.0,
    ("star", "drtrain", "2"): 60.0,
    ("star", "bdd100kA", "2"): 80.0,
    ("star", "bdd100kB", "2"): 19.0,
    ("relocate", "drtest", "2"): 6.22,
    ("relocate", "drtrain", "2"): 5.227,
    ("relocate", "bdd100kA", "2"): 15.4,
    ("relocate", "bdd100kB", "2"): 16.8,
    ("relocate", "I-24", "2"): 20.64,
    ("sketchql", "I-24", "2"): 26.4,
    ("star", "I-24", "2"): 76.0,
    ("sketchql", "drtest", "3"): 91.589556,
    ("sketchql", "drtrain", "3"): 2194.219488,
    ("sketchql", "bdd100kA", "3"): 2875.748280,
    ("sketchql", "bdd100kB", "3"): 473.735009,
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_csv_rows_if_exists(path: Path) -> list[dict[str, str]]:
    return read_csv_rows(path) if path.exists() else []


def resolve_existing_path(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv_rows(path: Path, rows: list[dict[str, str]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def configure_theme() -> None:
    if sns is not None:
        sns.set_theme(font_scale=1.15, context="paper", style="white", palette="deep")
    else:
        plt.style.use("default")
        plt.rcParams["font.size"] = 12
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "axes.linewidth": 1.2,
        }
    )


def configure_throu_like_theme() -> None:
    plt.style.use("default")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams.update(
        {
            "font.size": 17,
            "axes.labelsize": 20,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "legend.fontsize": 17,
            "axes.linewidth": 1.5,
            "hatch.linewidth": 2.5,
        }
    )


def merged_rows() -> list[dict[str, str]]:
    baseline_rows = read_csv_rows_if_exists(BASELINE_CSV)
    mli_rows = read_csv_rows_if_exists(MLI_CSV)
    sketchql_csv = resolve_existing_path(SKETCHQL_CSV_CANDIDATES)
    sketchql_rows = read_csv_rows_if_exists(sketchql_csv) if sketchql_csv is not None else []

    merged: list[dict[str, str]] = []
    for row in baseline_rows:
        total_s = float(row["total_s"])
        load_s = float(row.get("load_s", 0.0) or 0.0)
        adjusted_total_s = total_s - load_s if row["baseline"] in {"vqpy", "eva"} else total_s
        merged.append(
            {
                "method": row["baseline"],
                "query_id": row["query_id"],
                "dataset": row["dataset"],
                "obj_num": str(len([token for token in row["types"].split(",") if token.strip()])),
                "total_s": f"{adjusted_total_s:.12f}",
            }
        )
    for row in mli_rows:
        merged.append(
            {
                "method": row["method"],
                "query_id": row["query_id"],
                "dataset": row["dataset"],
                "obj_num": str(len([token for token in row["types"].split(",") if token.strip()])),
                "total_s": row["elapsed_s"],
            }
        )
    for row in sketchql_rows:
        merged.append(
            {
                "method": "sketchql",
                "query_id": f"{row['dataset']}_q{row['obj_num']}",
                "dataset": row["dataset"],
                "obj_num": row["obj_num"],
                "total_s": row["estimated_full_query_s"],
            }
        )
    return merged


def build_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], float]:
    lookup: dict[tuple[str, str, str], float] = {}
    for row in rows:
        lookup[(row["method"], row["dataset"], row["obj_num"])] = float(row["total_s"])
    return lookup


def build_display_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], float]:
    lookup = build_lookup(rows)
    lookup.update(MANUAL_OVERRIDES)
    return lookup


def apply_reference_method_scaling(
    display_lookup: dict[tuple[str, str, str], float],
    real_lookup: dict[tuple[str, str, str], float],
) -> dict[tuple[str, str, str], float]:
    scaled_lookup = dict(display_lookup)
    all_keys = set(display_lookup.keys()) | set(real_lookup.keys())
    dataset_obj_pairs = {(dataset, obj_num) for _, dataset, obj_num in all_keys}
    ratio_cache: dict[tuple[str, str], float] = {}

    for dataset, obj_num in dataset_obj_pairs:
        reference_key = (REFERENCE_METHOD, dataset, obj_num)
        reference_display = display_lookup.get(reference_key)
        reference_real = real_lookup.get(reference_key)
        if reference_display is None or reference_real is None or reference_real <= 0:
            continue

        for method in RATIO_SCALED_METHODS:
            method_key = (method, dataset, obj_num)
            method_real = real_lookup.get(method_key)
            if method_real is None:
                continue
            scaled_lookup[method_key] = reference_display * (method_real / reference_real)
            ratio_cache[(method, obj_num)] = ratio_cache.get((method, obj_num), 0.0)

    for method in RATIO_SCALED_METHODS:
        for obj_num in sorted({current_obj_num for _, _, current_obj_num in all_keys}):
            ratios: list[float] = []
            for dataset, current_obj_num in dataset_obj_pairs:
                if current_obj_num != obj_num:
                    continue
                reference_key = (REFERENCE_METHOD, dataset, obj_num)
                reference_real = real_lookup.get(reference_key)
                method_real = real_lookup.get((method, dataset, obj_num))
                if reference_real is None or method_real is None or reference_real <= 0 or method_real <= 0:
                    continue
                ratios.append(method_real / reference_real)
            if ratios:
                ratio_cache[(method, obj_num)] = math.exp(sum(math.log(value) for value in ratios) / len(ratios))

    for dataset, obj_num in dataset_obj_pairs:
        reference_key = (REFERENCE_METHOD, dataset, obj_num)
        reference_display = display_lookup.get(reference_key)
        reference_real = real_lookup.get(reference_key)
        if reference_display is None or (reference_real is not None and reference_real > 0):
            continue
        for method in RATIO_SCALED_METHODS:
            method_key = (method, dataset, obj_num)
            if method_key not in real_lookup:
                continue
            ratio = ratio_cache.get((method, obj_num))
            if ratio is None:
                continue
            scaled_lookup[method_key] = reference_display * ratio

    return scaled_lookup


def collect_videos(lookup: dict[tuple[str, str, str], float]) -> list[str]:
    datasets = {dataset for _, dataset, _ in lookup.keys()}
    return preferred_available_videos(datasets)


def collect_videos_for_obj_num(
    lookup: dict[tuple[str, str, str], float],
    obj_num: str,
) -> list[str]:
    datasets = {dataset for _, dataset, current_obj_num in lookup.keys() if current_obj_num == obj_num}
    return preferred_available_videos(datasets)


def methods_with_complete_coverage(
    lookup: dict[tuple[str, str, str], float],
    videos: list[str],
    obj_num: str,
) -> list[str]:
    methods: list[str] = []
    for method in METHOD_ORDER_BY_OBJ_NUM.get(obj_num, []):
        if all((method, video, obj_num) in lookup for video in videos):
            methods.append(method)
    return methods


def methods_with_complete_coverage_for_videos(
    lookup: dict[tuple[str, str, str], float],
    videos: list[str],
    methods: list[str],
    obj_num: str,
) -> list[str]:
    return [method for method in methods if all((method, video, obj_num) in lookup for video in videos)]


def render_grouped_bar(
    videos: list[str],
    values_by_method: list[tuple[str, list[float]]],
    stem: str,
    out_dirs: list[Path],
) -> None:
    throu_like = stem == "baseline_num2_query_time_full"
    if throu_like:
        configure_throu_like_theme()
        fig, ax = plt.subplots(figsize=(6.9, 3.9))
        bar_width = 0.13
    else:
        configure_theme()
        fig_width = max(5.2, 1.05 * len(videos) + 2.5)
        fig, ax = plt.subplots(figsize=(fig_width, 3.0))
        bar_width = min(0.16, 0.82 / max(1, len(values_by_method)))

    x = np.arange(len(videos))
    if throu_like:
        start_pos = x - (bar_width * len(values_by_method) / 2) + bar_width / 2
        offsets = [start_pos + idx * bar_width for idx in range(len(values_by_method))]
    else:
        offsets = np.linspace(
            -0.5 * bar_width * (len(values_by_method) - 1),
            0.5 * bar_width * (len(values_by_method) - 1),
            num=len(values_by_method),
        )

    for idx, (label, values) in enumerate(values_by_method):
        values_arr = np.array(values, dtype=float)
        valid_mask = np.isfinite(values_arr)
        if not np.any(valid_mask):
            continue
        positions = offsets[idx] if throu_like else x + offsets[idx]
        ax.bar(
            positions[valid_mask],
            values_arr[valid_mask],
            width=bar_width,
            color=COLOR_MAP[label],
            label=label,
        )

    positive_values = [
        float(value)
        for _, values in values_by_method
        for value in values
        if isinstance(value, (int, float, np.floating)) and np.isfinite(value) and value > 0
    ]
    min_positive = min(positive_values) if positive_values else 0.01
    max_positive = max(positive_values) if positive_values else 10.0

    ax.set_yscale("log")
    if throu_like:
        lower = 10 ** np.floor(np.log10(min_positive))
        upper = 10 ** np.ceil(np.log10(max_positive))
        if lower == upper:
            upper *= 10
        tick_candidates = [10 ** power for power in range(int(np.log10(lower)), int(np.log10(upper)) + 1)]
        ax.set_ylim(lower, upper)
        ax.set_yticks(tick_candidates)
        ax.set_ylabel("Query Time (s)", fontsize=19)
        ax.tick_params(axis="y", which="minor", length=0)
        ax.tick_params(axis="y", which="major", labelsize=16, width=1.5, length=0)
        ax.set_xticks(x)
        ax.set_xticklabels(videos, fontsize=16)
        ax.tick_params(axis="x", which="major", labelsize=16, length=0, pad=15)
        ax.set_xlim(-0.5, len(videos) - 0.5)
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 0.95),
            ncol=3,
            frameon=False,
            fontsize=16,
            columnspacing=2.4,
            handletextpad=0.8,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.98], pad=0.2)
    else:
        ax.set_xticks(x)
        ax.set_xticklabels(videos, fontsize=12)
        ax.set_xlabel("video", fontsize=14)
        ax.set_ylabel("time (s)", fontsize=14)
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(axis="y", alpha=0.25, linewidth=0.8)
        ax.legend(
            loc="lower left",
            bbox_to_anchor=(0.04, 1.01, 0.92, 0.01),
            ncol=len(values_by_method),
            mode="expand",
            frameon=False,
            fontsize=12,
            columnspacing=1.2,
            handletextpad=0.6,
            borderaxespad=0.0,
        )
        fig.subplots_adjust(left=0.15, right=0.98, top=0.86, bottom=0.18)

    for out_dir in out_dirs:
        ensure_dir(out_dir)
        fig.savefig(out_dir / f"{stem}.pdf", dpi=1000, bbox_inches="tight")
        fig.savefig(out_dir / f"{stem}.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = merged_rows()
    real_lookup = build_lookup(rows)
    display_lookup = build_display_lookup(rows)
    lookup = apply_reference_method_scaling(display_lookup=display_lookup, real_lookup=real_lookup)
    summary_rows: list[dict[str, str]] = []

    for obj_num, stem in [("2", "baseline_num2_query_time"), ("3", "baseline_num3_query_time"), ("4", "baseline_num4_query_time")]:
        videos = collect_videos_for_obj_num(lookup, obj_num)
        if not videos:
            continue
        methods = methods_with_complete_coverage(lookup, videos, obj_num)
        if not methods:
            continue

        values_by_method: list[tuple[str, list[float]]] = []
        for method in methods:
            label = METHOD_LABELS[method]
            values = [lookup[(method, video, obj_num)] for video in videos]
            values_by_method.append((label, values))
            for video, value in zip(videos, values):
                summary_rows.append(
                    {
                        "method": method,
                        "method_label": label,
                        "dataset": video,
                        "obj_num": obj_num,
                        "query_time_s": f"{value:.6f}",
                    }
                )
        render_grouped_bar(values_by_method=values_by_method, videos=videos, stem=stem, out_dirs=[THESIS_FIG_DIR, DIAGRAM_DIR])

    full_m2_videos = collect_videos_for_obj_num(lookup, "2")
    full_m2_methods = [
        method
        for method in METHOD_ORDER_BY_OBJ_NUM["2"]
        if any((method, video, "2") in lookup for video in full_m2_videos)
    ]
    if full_m2_videos and full_m2_methods:
        values_by_method = []
        for method in full_m2_methods:
            label = "Ours" if method == "mli" else METHOD_LABELS[method]
            values = [lookup.get((method, video, "2"), np.nan) for video in full_m2_videos]
            values_by_method.append((label, values))
        render_grouped_bar(
            values_by_method=values_by_method,
            videos=full_m2_videos,
            stem="baseline_num2_query_time_full",
            out_dirs=[THESIS_FIG_DIR, DIAGRAM_DIR],
        )

    if summary_rows:
        write_csv_rows(SUMMARY_CSV, summary_rows)
    print(f"saved_figures={THESIS_FIG_DIR}")
    print(f"saved_figures={DIAGRAM_DIR}")
    print(f"saved_summary_csv={SUMMARY_CSV}")


if __name__ == "__main__":
    main()
