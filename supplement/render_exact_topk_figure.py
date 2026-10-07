from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from supplement.dataset_config import DIAGRAM_DIR, REPO_ROOT, THESIS_FIG_DIR


SUMMARY_PATH = (
    REPO_ROOT
    / "supplement"
    / "out"
    / "exact_topk_benchmark"
    / "exact_topk_summary.csv"
)
PAPER_TOPK_SUMMARY_PATHS = tuple(
    REPO_ROOT
    / "supplement"
    / "out"
    / directory
    / "paper_query_time_summary.csv"
    for directory in (
        "paper_benchmark_figure7_topk_10_50_drtest",
        "paper_benchmark_figure7_topk_10_50_drtrain",
        "paper_benchmark_figure7_topk_10_50_bdd100kA",
        "paper_benchmark_figure7_topk_10_50_bdd100kB",
        "paper_benchmark_figure7_topk_10_50_i24",
    )
)
OUTPUT_STEM = "topk_query_time_paper_consistent"
OUTPUT_DIRS = (DIAGRAM_DIR, THESIS_FIG_DIR)
VIDEO_ORDER = ("drtest", "drtrain", "bdd100kA", "bdd100kB", "I-24")
COLORS = {
    "drtest": "#3b5da3",
    "drtrain": "#0086c5",
    "bdd100kA": "#81a2ed",
    "bdd100kB": "#ffc920",
    "I-24": "#e58760",
}
MARKERS = {
    "drtest": "o",
    "drtrain": "s",
    "bdd100kA": "^",
    "bdd100kB": "D",
    "I-24": "P",
}


def load_summary(path: str | Path = SUMMARY_PATH) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open(newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            rows.append(
                {
                    "video": raw["video"],
                    "k": int(raw["k"]),
                    "query_count": int(raw["query_count"]),
                    "exhaustive_s": float(raw["exhaustive_s"]),
                    "optimized_s": float(raw["optimized_s"]),
                    "processed_window_ratio": float(raw["processed_window_ratio"]),
                    "early_termination_rate": float(raw["early_termination_rate"]),
                    "all_exact": raw["all_exact"].strip().lower() == "true",
                }
            )
    return rows


def load_paper_query_times(
    paths: tuple[Path, ...] = PAPER_TOPK_SUMMARY_PATHS,
) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as f:
            for raw in csv.DictReader(f):
                if raw["experiment"] != "vary_topk":
                    continue
                rows.append(
                    {
                        "video": raw["video"],
                        "k": int(raw["vary_value"]),
                        "query_count": int(raw["query_count"]),
                        "query_time_s": float(raw["robust_t_total_s"]),
                    }
                )
    return rows


def query_time_series(rows: list[dict]) -> dict[str, list[tuple[int, float]]]:
    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        series[row["video"]].append((int(row["k"]), float(row["query_time_s"])))
    for values in series.values():
        values.sort(key=lambda item: item[0])
    return dict(series)


def scale_video_query_times(rows: list[dict], video: str, scale: float) -> list[dict]:
    scaled_rows: list[dict] = []
    for row in rows:
        scaled = dict(row)
        if scaled["video"] == video:
            scaled["query_time_s"] = float(scaled["query_time_s"]) * scale
        scaled_rows.append(scaled)
    return scaled_rows


def configure_theme() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.linewidth": 1.4,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "axes.grid": False,
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", which="major", width=1.2, length=3.5)
    ax.tick_params(axis="both", which="minor", length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.4)


def render(
    output_dirs: tuple[Path, ...] = OUTPUT_DIRS,
    rows: list[dict] | None = None,
    output_stem: str = OUTPUT_STEM,
    illustrative_video: str | None = None,
) -> list[Path]:
    rows = load_paper_query_times() if rows is None else rows
    if len(rows) != len(VIDEO_ORDER) * 5:
        raise ValueError("the figure requires all five paper Top-k measurements")

    series = query_time_series(rows)

    configure_theme()
    fig, ax = plt.subplots(figsize=(6.9, 3.9))

    for video in VIDEO_ORDER:
        values = series[video]
        ks = [item[0] for item in values]
        query_times = [item[1] for item in values]
        common = {
            "color": COLORS[video],
            "marker": MARKERS[video],
            "linewidth": 2.0,
            "markersize": 5.8,
            "markerfacecolor": "white",
            "markeredgewidth": 1.4,
        }
        label = f"{video}*" if video == illustrative_video else video
        ax.plot(ks, query_times, label=label, **common)

    ax.set_ylabel("Query Time (s)")
    ax.set_xlabel("Top-k")
    ax.set_xticks([10, 20, 30, 40, 50])
    if illustrative_video is None:
        ax.set_ylim(0.0, 6.2)
        ax.set_yticks([0, 2, 4, 6])
    else:
        ax.set_ylim(0.0, 1.2)
        ax.set_yticks([0.0, 0.3, 0.6, 0.9, 1.2])
    style_axis(ax)

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=False,
        columnspacing=1.4,
        handletextpad=0.5,
    )
    bottom = 0.23 if illustrative_video is not None else 0.18
    fig.subplots_adjust(left=0.13, right=0.985, top=0.79, bottom=bottom)
    if illustrative_video is not None:
        fig.text(
            0.985,
            0.03,
            "* I-24 uses illustrative values scaled to 0.15x.",
            ha="right",
            fontsize=9,
        )

    outputs: list[Path] = []
    for output_dir in output_dirs:
        output_dir.mkdir(parents=True, exist_ok=True)
        for suffix, dpi in ((".pdf", 1000), (".png", 400)):
            output = output_dir / f"{output_stem}{suffix}"
            fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
            outputs.append(output)
    plt.close(fig)
    return outputs


def render_i24_illustrative(
    output_dirs: tuple[Path, ...] = OUTPUT_DIRS,
) -> list[Path]:
    rows = scale_video_query_times(load_paper_query_times(), "I-24", 0.15)
    return render(
        output_dirs=output_dirs,
        rows=rows,
        output_stem="topk_query_time_i24_illustrative",
        illustrative_video="I-24",
    )


if __name__ == "__main__":
    for output_path in render():
        print(output_path)
