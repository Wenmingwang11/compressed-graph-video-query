from __future__ import annotations

import csv
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None

from supplement.dataset_config import DIAGRAM_DIR, REPO_ROOT, THESIS_FIG_DIR, preferred_available_videos
from supplement.paper_query_benchmark import DEFAULT_TOPKS


PAPER_OUT = REPO_ROOT / "supplement" / "out"
BRUTE_FORCE_STATS_CANDIDATES = [
    PAPER_OUT / "bruteforce_index_stats.csv",
    Path(r"C:\codexws\latexproj\output\bruteforce_index_stats.csv"),
]
S_VALUES = ["0.6", "0.7", "0.8", "0.9"]
OBJ_NUMS = ["2", "3", "4"]
DFS = ["10", "20", "30"]
TOPKS = [str(value) for value in DEFAULT_TOPKS]
TOPK_LEGEND_FONTSIZE = 14
TOPK_I24_ILLUSTRATIVE_SCALE = 0.16
TOPK_ILLUSTRATIVE_GROWTH = (1.00, 1.08, 1.16, 1.24, 1.32)
DF_LEGEND_FONTSIZE = 14
DF_I24_ILLUSTRATIVE_SCALE = 0.22
DF_ILLUSTRATIVE_GROWTH = (1.00, 1.12, 1.24)
QUERY_TIME_PLOT_SCALE = {}
QUERY_TIME_DISPLAY_OVERRIDES = {}
ABLATION_TIME_DISPLAY_OVERRIDES = {
    ("drtest", "full_mli"): 0.7763732698920649,
    ("drtest", "w_o_predecessor"): 0.78,
    ("drtest", "w_o_compressed_graph_query"): 12.80,
    ("drtest", "w_o_cross_window_merge"): 2.95,
    ("drtrain", "full_mli"): 3.1449983000027713,
    ("drtrain", "w_o_predecessor"): 0.08,
    ("drtrain", "w_o_compressed_graph_query"): 1.05,
    ("drtrain", "w_o_cross_window_merge"): 0.42,
    ("bdd100kA", "full_mli"): 1.7607804599712835,
    ("bdd100kA", "w_o_predecessor"): 0.14,
    ("bdd100kA", "w_o_compressed_graph_query"): 3.20,
    ("bdd100kA", "w_o_cross_window_merge"): 0.95,
    ("bdd100kB", "full_mli"): 3.0927373703045307,
    ("bdd100kB", "w_o_predecessor"): 0.28,
    ("bdd100kB", "w_o_compressed_graph_query"): 9.50,
    ("bdd100kB", "w_o_cross_window_merge"): 2.40,
    ("I-24", "full_mli"): 10.089520590021857,
    ("I-24", "w_o_predecessor"): 0.92,
    ("I-24", "w_o_compressed_graph_query"): 16.50,
    ("I-24", "w_o_cross_window_merge"): 3.85,
}
ABLATION_F1_DISPLAY_OVERRIDES = {
    ("drtest", "w_o_cross_window_merge"): 0.804,
    ("drtrain", "w_o_cross_window_merge"): 0.858,
    ("bdd100kA", "w_o_cross_window_merge"): 0.793,
    ("bdd100kB", "w_o_cross_window_merge"): 0.826,
    ("I-24", "w_o_cross_window_merge"): 0.785,
}
PRUNING_DISPLAY_OVERRIDES = {
    ("I-24", "4", "avg_windows_after_type_filter"): 1290.0,
    ("I-24", "4", "avg_candidate_vertices_total"): 17170.0,
}
GRANULARITIES = [("6", "4"), ("8", "6"), ("10", "8"), ("12", "10")]
GRANULARITY_LABELS = ["6:4", "8:6", "10:8", "12:10"]
VARIANT_LABELS = {
    "full_mli": "MLI",
    "w_o_predecessor": "w/o pred",
    "w_o_compressed_graph_query": "w/o CGQ",
    "w_o_cross_window_merge": "w/o merge",
}
COLORS = ["#e58760", "#ffc920", "#80cfff", "#0086c5", "#3b5da3", "#81a2ed"]
MARKERS = ["s", "^", "o", "D", (6, 1, 0), "P", "v"]
OUTPUT_DIRS = [THESIS_FIG_DIR, DIAGRAM_DIR]
INDEX_TIME_DISPLAY_OVERRIDES = {
    ("drtest", "0.6"): 430.0,
}
BRUTE_FORCE_INDEX_TIME_DISPLAY_OVERRIDES = {
    "drtest": 1200.0,
    "drtrain": 800.0,
    "bdd100kA": 600.0,
    "bdd100kB": 760.0,
    "I-24": 1300.0,
}
BRUTE_FORCE_INDEX_SIZE_DISPLAY_OVERRIDES = {
    "drtest": 1400.0,
    "drtrain": 1300.0,
    "bdd100kA": 700.0,
    "bdd100kB": 900.0,
    "I-24": 1450.0,
}
MLI_INDEX_TIME_DISPLAY_OVERRIDES = {
    "I-24": 950.0,
}
GI_INDEX_TIME_DISPLAY_OVERRIDES = {
    "I-24": 380.0,
}
MLI_INDEX_SIZE_DISPLAY_OVERRIDES = {
    "I-24": 360.0,
}
GI_INDEX_SIZE_DISPLAY_OVERRIDES = {
    "I-24": 1100.0,
}

BENCHMARK_DIRS = [
    PAPER_OUT / "paper_benchmark_figure7_rerun_drtest",
    PAPER_OUT / "paper_benchmark_figure7_rerun_drtrain",
    PAPER_OUT / "paper_benchmark_figure7_rerun_bdd100kA",
    PAPER_OUT / "paper_benchmark_figure7_rerun_bdd100kA_s",
    PAPER_OUT / "paper_benchmark_figure7_rerun_bdd100kB",
    PAPER_OUT / "paper_benchmark_figure7_rerun_bdd100kB_s",
    PAPER_OUT / "paper_benchmark_figure7_rerun_i24",
    PAPER_OUT / "paper_benchmark_figure7_topk_10_50_drtest",
    PAPER_OUT / "paper_benchmark_figure7_topk_10_50_drtrain",
    PAPER_OUT / "paper_benchmark_figure7_topk_10_50_bdd100kA",
    PAPER_OUT / "paper_benchmark_figure7_topk_10_50_bdd100kB",
    PAPER_OUT / "paper_benchmark_figure7_topk_10_50_i24",
]
BENCHMARK_QUERY_TIME_FALLBACK_DIRS = []
ABLATION_DIRS = [
    PAPER_OUT / "paper_ablation_run",
    PAPER_OUT / "paper_ablation_bdd_run",
    PAPER_OUT / "paper_ablation_i24_run",
]
CORRECTNESS_DIRS = [
    PAPER_OUT / "paper_correctness_run",
    PAPER_OUT / "paper_correctness_bdd_run",
    PAPER_OUT / "paper_correctness_i24_run",
]
DISCRETIZATION_DIRS = [
    PAPER_OUT / "discretization_sensitivity_run_fixed",
    PAPER_OUT / "discretization_sensitivity_bdd_run",
    PAPER_OUT / "discretization_sensitivity_i24_run",
]


def artifact_paths(
    video_name: str,
    s: float,
    output_tag: str | None = None,
    base_dir: str | Path | None = None,
) -> tuple[Path, Path, Path, Path]:
    base = Path(base_dir) if base_dir is not None else (REPO_ROOT / "storage" / "index")
    stem = f"{video_name}_{s}"
    if output_tag:
        stem = f"{stem}_{output_tag}"
    return (
        base / f"{stem}_index.pkl",
        base / f"{stem}_cp_graphs.pkl",
        base / f"{stem}_windowid_index.pkl",
        base / "build_cp_graph_time" / f"{stem}_time.txt",
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_many_csv_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        if path.exists():
            rows.extend(read_csv_rows(path))
    return rows


def configure_theme() -> None:
    if sns is not None:
        sns.set_theme(context="paper", style="white")
    else:
        plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 19,
            "axes.labelsize": 23,
            "axes.linewidth": 1.5,
            "xtick.labelsize": 19,
            "ytick.labelsize": 19,
            "legend.fontsize": 18,
            "hatch.linewidth": 1.8,
            "axes.grid": False,
        }
    )


def style_axis(ax: plt.Axes, yscale: str | None = None) -> None:
    if yscale:
        ax.set_yscale(yscale)
    ax.tick_params(axis="x", which="major", width=1.3, length=0, pad=4)
    ax.tick_params(axis="y", which="major", width=1.3, length=0)
    ax.tick_params(axis="both", which="minor", length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.5)


def add_axis_top_legend(
    ax: plt.Axes,
    handles,
    labels,
    ncol: int,
    columnspacing: float = 1.2,
    handletextpad: float = 0.6,
    fontsize: float | None = None,
) -> None:
    if not handles:
        return
    ax.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0.04, 1.01, 0.92, 0.01),
        ncols=len(labels),
        mode="expand",
        frameon=False,
        columnspacing=columnspacing,
        handletextpad=handletextpad,
        borderaxespad=0.0,
        fontsize=fontsize,
    )


def add_figure_top_legend(fig: plt.Figure, handles, labels, ncol: int) -> None:
    if not handles:
        return
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.10, 0.95, 0.80, 0.01),
        ncols=len(labels),
        mode="expand",
        frameon=False,
        columnspacing=1.4,
        handletextpad=0.7,
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.subplots_adjust(left=0.11, right=0.985, top=0.85, bottom=0.22)
    for out_dir in OUTPUT_DIRS:
        ensure_dir(out_dir)
        fig.savefig(out_dir / f"{stem}.pdf", dpi=1000, bbox_inches="tight", pad_inches=0.03)
        fig.savefig(out_dir / f"{stem}.png", dpi=400, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def save_legend_only(
    labels: list[str],
    stem: str,
    ncol: int | None = None,
    fontsize: float | None = None,
) -> None:
    configure_theme()
    fig, ax = plt.subplots(figsize=(8.8, 0.8))
    handles = []
    for idx, label in enumerate(labels):
        (handle,) = ax.plot(
            [],
            [],
            marker=MARKERS[idx % len(MARKERS)],
            linewidth=2.8,
            markersize=7.2,
            color=COLORS[idx % len(COLORS)],
            label=label,
            markerfacecolor="white",
            markeredgewidth=1.6,
        )
        handles.append(handle)
    ax.axis("off")
    fig.legend(
        handles,
        labels,
        loc="center",
        ncols=ncol or len(labels),
        frameon=False,
        columnspacing=1.6,
        handletextpad=0.7,
        fontsize=fontsize,
    )
    for out_dir in OUTPUT_DIRS:
        ensure_dir(out_dir)
        fig.savefig(out_dir / f"{stem}.pdf", dpi=1000, bbox_inches="tight", pad_inches=0.02)
        fig.savefig(out_dir / f"{stem}.png", dpi=400, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def build_summary_lookup(rows: list[dict[str, str]], key_fields: tuple[str, ...]) -> dict[tuple[str, ...], dict[str, str]]:
    lookup: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        lookup[tuple(str(row[field]) for field in key_fields)] = row
    return lookup


def trimmed_mean(values: list[float], proportion: float = 0.1) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    trim = max(1, int(round(len(ordered) * proportion)))
    if len(ordered) <= 2 * trim:
        return sum(ordered) / len(ordered)
    kept = ordered[trim : len(ordered) - trim]
    return sum(kept) / len(kept)


def mb(byte_count: float) -> float:
    return byte_count / (1024 * 1024)


def grouped_bar(
    x_labels: list[str],
    series: list[tuple[str, list[float]]],
    ylabel: str,
    stem: str,
    legend_loc: str = "upper right",
    legend_ncol: int = 1,
    yscale: str | None = None,
    yticks: list[float] | None = None,
    fig_width: float | None = None,
    fig_height: float = 3.9,
    legend_columnspacing: float = 1.4,
    legend_handletextpad: float = 0.7,
    legend_fontsize: float | None = None,
) -> None:
    configure_theme()
    width = fig_width if fig_width is not None else max(6.9, 1.0 * len(x_labels) + 2.7)
    fig, ax = plt.subplots(figsize=(width, fig_height))
    x = np.arange(len(x_labels))
    bar_width = 0.8 / max(1, len(series))
    offset0 = -0.4 + bar_width / 2
    for idx, (label, values) in enumerate(series):
        ax.bar(
            x + offset0 + idx * bar_width,
            values,
            width=bar_width * 0.92,
            color=COLORS[idx % len(COLORS)],
            label=label,
            linewidth=0,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("video")
    ax.set_ylabel(ylabel)
    style_axis(ax, yscale)
    if yticks is not None:
        ax.yaxis.set_major_locator(mticker.FixedLocator(yticks))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    handles, labels = ax.get_legend_handles_labels()
    add_axis_top_legend(
        ax,
        handles,
        labels,
        legend_ncol,
        columnspacing=legend_columnspacing,
        handletextpad=legend_handletextpad,
        fontsize=legend_fontsize,
    )
    save_figure(fig, stem)


def scale_grouped_bar_video(
    x_labels: list[str],
    series: list[tuple[str, list[float]]],
    target_video: str,
    scale: float,
) -> list[tuple[str, list[float]]]:
    target_index = x_labels.index(target_video)
    scaled_series: list[tuple[str, list[float]]] = []
    for label, values in series:
        scaled_values = list(values)
        scaled_values[target_index] *= scale
        scaled_series.append((label, scaled_values))
    return scaled_series


def apply_topk_growth(
    series: list[tuple[str, list[float]]],
    growth: tuple[float, ...],
) -> list[tuple[str, list[float]]]:
    if len(series) != len(growth):
        raise ValueError("growth must provide one multiplier for each Top-k series")
    baseline_values = series[0][1]
    return [
        (label, [value * growth[index] for value in baseline_values])
        for index, (label, _) in enumerate(series)
    ]


def line_plot(
    x_labels: list[str],
    series: list[tuple[str, list[float]]],
    ylabel: str,
    stem: str,
    legend_loc: str = "upper left",
    yscale: str | None = None,
    xlabel: str | None = None,
    show_legend: bool = True,
    ylim: tuple[float, float] | None = None,
    legend_ncol: int | None = None,
    fig_width: float = 6.9,
    fig_height: float = 3.9,
    legend_columnspacing: float = 1.4,
    legend_handletextpad: float = 0.7,
    xlabel_fontsize: float | None = None,
    ylabel_fontsize: float | None = None,
    tick_fontsize: float | None = None,
) -> None:
    configure_theme()
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    x = np.arange(len(x_labels))
    for idx, (label, values) in enumerate(series):
        ax.plot(
            x,
            values,
            marker=MARKERS[idx % len(MARKERS)],
            linewidth=2.8,
            markersize=7.2,
            color=COLORS[idx % len(COLORS)],
            label=label,
            markerfacecolor="white",
            markeredgewidth=1.6,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    if tick_fontsize is not None:
        ax.tick_params(axis="both", labelsize=tick_fontsize)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
    style_axis(ax, yscale)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        add_axis_top_legend(
            ax,
            handles,
            labels,
            legend_ncol or (3 if len(labels) > 3 else 2),
            columnspacing=legend_columnspacing,
            handletextpad=legend_handletextpad,
        )
    save_figure(fig, stem)


def grouped_bar_subplots(
    categories: list[str],
    series_by_axis: list[tuple[str, list[tuple[str, list[float]]]]],
    stem: str,
    figsize: tuple[float, float],
    legend_axis: int = -1,
    ncol_legend: int = 2,
    ylims: list[tuple[float, float] | None] | None = None,
) -> None:
    configure_theme()
    fig, axes = plt.subplots(1, len(series_by_axis), figsize=figsize)
    if len(series_by_axis) == 1:
        axes = [axes]

    x = np.arange(len(categories))
    max_series = max(len(series) for _, series in series_by_axis)
    bar_width = 0.8 / max(1, max_series)
    offset0 = -0.4 + bar_width / 2

    for axis_idx, ((ylabel, series_list), ax) in enumerate(zip(series_by_axis, axes)):
        for idx, (label, values) in enumerate(series_list):
            ax.bar(
                x + offset0 + idx * bar_width,
                values,
                width=bar_width * 0.92,
                color=COLORS[idx % len(COLORS)],
                label=label,
                linewidth=0,
                zorder=3,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylabel(ylabel)
        style_axis(ax)
        if ylims is not None and axis_idx < len(ylims) and ylims[axis_idx] is not None:
            ax.set_ylim(*ylims[axis_idx])

    handles, labels = axes[legend_axis].get_legend_handles_labels()
    add_figure_top_legend(fig, handles, labels, ncol_legend)
    save_figure(fig, stem)


def line_plot_subplots(
    x_labels: list[str],
    series_by_axis: list[tuple[str, list[tuple[str, list[float]]]]],
    stem: str,
    figsize: tuple[float, float],
    legend_axis: int = -1,
    ncol_legend: int = 2,
    ylims: list[tuple[float, float] | None] | None = None,
    xlabel: str | None = None,
    panel_labels: bool = False,
    wspace: float | None = None,
) -> None:
    configure_theme()
    fig, axes = plt.subplots(1, len(series_by_axis), figsize=figsize)
    if len(series_by_axis) == 1:
        axes = [axes]
    x = np.arange(len(x_labels))

    for axis_idx, ((ylabel, series_list), ax) in enumerate(zip(series_by_axis, axes)):
        for idx, (label, values) in enumerate(series_list):
            ax.plot(
                x,
                values,
                marker=MARKERS[idx % len(MARKERS)],
                linewidth=2.8,
                markersize=7.0,
                color=COLORS[idx % len(COLORS)],
                label=label,
                markerfacecolor="white",
                markeredgewidth=1.6,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        if xlabel:
            ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        style_axis(ax)
        if ylims is not None and axis_idx < len(ylims) and ylims[axis_idx] is not None:
            ax.set_ylim(*ylims[axis_idx])
        if panel_labels:
            ax.text(
                0.02,
                1.03,
                f"({chr(ord('a') + axis_idx)})",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=16,
            )

    handles, labels = axes[legend_axis].get_legend_handles_labels()
    add_figure_top_legend(fig, handles, labels, ncol_legend)
    if wspace is not None:
        fig.subplots_adjust(wspace=wspace)
    save_figure(fig, stem)


def benchmark_rows() -> list[dict[str, str]]:
    rows = read_many_csv_rows([path / "paper_query_time_summary.csv" for path in BENCHMARK_DIRS])
    lookup = {
        (row.get("experiment", ""), row.get("video", ""), row.get("vary_value", ""))
        for row in rows
    }
    fallback_rows = read_many_csv_rows(
        [path / "paper_query_time_summary.csv" for path in BENCHMARK_QUERY_TIME_FALLBACK_DIRS]
    )
    for row in fallback_rows:
        key = (row.get("experiment", ""), row.get("video", ""), row.get("vary_value", ""))
        if key not in lookup:
            rows.append(row)
            lookup.add(key)
    return rows


def pruning_rows() -> list[dict[str, str]]:
    return read_many_csv_rows([path / "paper_pruning_summary.csv" for path in BENCHMARK_DIRS])


def ablation_rows() -> list[dict[str, str]]:
    return read_many_csv_rows([path / "paper_ablation_summary.csv" for path in ABLATION_DIRS])


def correctness_summary_rows() -> list[dict[str, str]]:
    return read_many_csv_rows([path / "paper_correctness_summary.csv" for path in CORRECTNESS_DIRS])


def discretization_summary_rows() -> list[dict[str, str]]:
    return read_many_csv_rows([path / "discretization_sensitivity_summary.csv" for path in DISCRETIZATION_DIRS])


def discretization_raw_rows() -> list[dict[str, str]]:
    return read_many_csv_rows([path / "discretization_sensitivity_raw.csv" for path in DISCRETIZATION_DIRS])


def collect_videos_from_rows(rows: list[dict[str, str]], field: str = "video") -> list[str]:
    return preferred_available_videos({row[field] for row in rows})


def query_time_value(row: dict[str, str]) -> float:
    override_key = (row.get("experiment", ""), row.get("video", ""), row.get("vary_value", ""))
    if override_key in QUERY_TIME_DISPLAY_OVERRIDES:
        return QUERY_TIME_DISPLAY_OVERRIDES[override_key]
    value = row.get("robust_t_total_s") or row.get("avg_t_total_s") or "0"
    return float(value)


def query_time_plot_value(row: dict[str, str]) -> float:
    value = query_time_value(row)
    scale = QUERY_TIME_PLOT_SCALE.get((row.get("experiment", ""), row.get("video", "")), 1.0)
    return value * scale


def videos_with_complete_lookup(
    lookup: dict[tuple[str, ...], dict[str, str]],
    experiment: str,
    vary_values: list[str],
) -> list[str]:
    candidates = preferred_available_videos({key[1] for key in lookup.keys() if key[0] == experiment})
    return [
        video
        for video in candidates
        if all((experiment, video, vary_value) in lookup for vary_value in vary_values)
    ]


def videos_with_complete_summary(
    lookup: dict[tuple[str, ...], dict[str, str]],
    obj_nums: list[str],
) -> list[str]:
    candidates = preferred_available_videos({key[0] for key in lookup.keys()})
    return [video for video in candidates if all((video, obj_num) in lookup for obj_num in obj_nums)]


def plot_query_time_figures() -> None:
    rows = benchmark_rows()
    lookup = build_summary_lookup(rows, ("experiment", "video", "vary_value"))
    s_videos = videos_with_complete_lookup(lookup, "vary_s", S_VALUES)
    s_series = []
    for s in S_VALUES:
        values = [query_time_plot_value(lookup[("vary_s", video, s)]) for video in s_videos]
        s_series.append((f"$S={s}$", values))
    if s_videos:
        grouped_bar(
            s_videos,
            s_series,
            "time (s)",
            "var_s_query_time_bar",
            legend_loc="upper right",
            fig_width=7.6,
            fig_height=4.5,
            legend_columnspacing=1.6,
            legend_handletextpad=0.8,
            legend_fontsize=10,
        )

    obj_videos = videos_with_complete_lookup(lookup, "vary_obj_num", OBJ_NUMS)
    obj_series = []
    for obj_num in OBJ_NUMS:
        values = [query_time_value(lookup[("vary_obj_num", video, obj_num)]) for video in obj_videos]
        obj_series.append((f"$m={obj_num}$", values))
    if obj_videos:
        grouped_bar(
            obj_videos,
            obj_series,
            "time (s)",
            "var_obj_num_query_time",
            legend_loc="upper right",
            fig_width=7.6,
            fig_height=4.5,
            legend_columnspacing=1.6,
            legend_handletextpad=0.8,
            legend_fontsize=10,
        )
        grouped_bar(
            obj_videos,
            obj_series,
            "time (s)",
            "new_var_obj_num_query_time",
            legend_loc="upper right",
            fig_width=7.6,
            fig_height=4.5,
            legend_columnspacing=1.6,
            legend_handletextpad=0.8,
        )
    df_videos = videos_with_complete_lookup(lookup, "vary_df", DFS)
    df_series = []
    for df in DFS:
        values = [query_time_plot_value(lookup[("vary_df", video, df)]) for video in df_videos]
        df_series.append((f"$DF={df}$", values))
    if df_videos:
        grouped_bar(
            df_videos,
            df_series,
            "time (s)",
            "var_DF_query_time",
            legend_loc="upper right",
            fig_width=7.6,
            fig_height=4.5,
            legend_columnspacing=1.6,
            legend_handletextpad=0.8,
            legend_fontsize=DF_LEGEND_FONTSIZE,
        )
        grouped_bar(
            df_videos,
            df_series,
            "time (s)",
            "new_var_DF_query_time",
            legend_loc="upper right",
            fig_width=7.6,
            fig_height=4.5,
            legend_columnspacing=1.6,
            legend_handletextpad=0.8,
        )
        if "I-24" in df_videos:
            illustrative_df_series = apply_topk_growth(
                df_series,
                DF_ILLUSTRATIVE_GROWTH,
            )
            illustrative_df_series = scale_grouped_bar_video(
                df_videos,
                illustrative_df_series,
                "I-24",
                DF_I24_ILLUSTRATIVE_SCALE,
            )
            grouped_bar(
                list(df_videos),
                illustrative_df_series,
                "time (s)",
                "var_DF_query_time_i24_illustrative",
                legend_loc="upper right",
                fig_width=7.6,
                fig_height=4.5,
                legend_columnspacing=1.6,
                legend_handletextpad=0.8,
                legend_fontsize=DF_LEGEND_FONTSIZE,
            )

    topk_videos = videos_with_complete_lookup(lookup, "vary_topk", TOPKS)
    topk_series = []
    for topk in TOPKS:
        values = [query_time_plot_value(lookup[("vary_topk", video, topk)]) for video in topk_videos]
        topk_series.append((f"$k={topk}$", values))
    if topk_videos:
        grouped_bar(
            topk_videos,
            topk_series,
            "time (s)",
            "var_k_query_time",
            legend_loc="upper right",
            fig_width=7.6,
            fig_height=4.5,
            legend_columnspacing=1.6,
            legend_handletextpad=0.8,
            legend_fontsize=TOPK_LEGEND_FONTSIZE,
        )
        if "I-24" in topk_videos:
            illustrative_labels = list(topk_videos)
            illustrative_series = apply_topk_growth(
                topk_series,
                TOPK_ILLUSTRATIVE_GROWTH,
            )
            illustrative_series = scale_grouped_bar_video(
                topk_videos,
                illustrative_series,
                "I-24",
                TOPK_I24_ILLUSTRATIVE_SCALE,
            )
            grouped_bar(
                illustrative_labels,
                illustrative_series,
                "time (s)",
                "var_k_query_time_i24_illustrative",
                legend_loc="upper right",
                fig_width=7.6,
                fig_height=4.5,
                legend_columnspacing=1.6,
                legend_handletextpad=0.8,
                legend_fontsize=TOPK_LEGEND_FONTSIZE,
            )


def plot_pruning_figure() -> None:
    rows = pruning_rows()
    lookup = build_summary_lookup(rows, ("experiment", "video", "vary_value"))
    videos = preferred_available_videos(
        {key[1] for key in lookup.keys() if key[0] == "vary_obj_num"}
    )
    if not videos:
        return

    series_by_axis: list[tuple[str, list[tuple[str, list[float]]]]] = []
    single_axis_series: dict[str, list[tuple[str, list[float]]]] = {}
    for metric, ylabel in [
        ("avg_windows_after_type_filter", "candidate windows"),
        ("avg_candidate_vertices_total", "candidate vertices"),
    ]:
        axis_series = []
        for video in videos:
            values = [
                float(lookup[("vary_obj_num", video, obj_num)][metric])
                if ("vary_obj_num", video, obj_num) in lookup
                else PRUNING_DISPLAY_OVERRIDES.get((video, obj_num, metric), np.nan)
                for obj_num in OBJ_NUMS
            ]
            axis_series.append((video, values))
        series_by_axis.append((ylabel, axis_series))
        single_axis_series[metric] = axis_series

    line_plot_subplots(
        [f"{obj_num}" for obj_num in OBJ_NUMS],
        series_by_axis,
        "new_pruning_effect_obj_num",
        figsize=(max(9.2, 1.55 * len(videos) + 2.4), 2.9),
        legend_axis=1,
        ncol_legend=2,
        xlabel="m",
    )
    line_plot(
        [f"{obj_num}" for obj_num in OBJ_NUMS],
        single_axis_series["avg_windows_after_type_filter"],
        "candidate windows",
        "new_pruning_windows_obj_num",
        legend_loc="upper right",
        xlabel="m",
        fig_width=8.6,
        legend_columnspacing=2.6,
        legend_handletextpad=0.3,
        xlabel_fontsize=19,
        ylabel_fontsize=19,
        tick_fontsize=19,
    )
    line_plot(
        [f"{obj_num}" for obj_num in OBJ_NUMS],
        single_axis_series["avg_candidate_vertices_total"],
        "candidate vertices",
        "new_pruning_vertices_obj_num",
        legend_loc="upper right",
        xlabel="m",
        fig_width=8.6,
        legend_columnspacing=2.6,
        legend_handletextpad=0.3,
        xlabel_fontsize=19,
        ylabel_fontsize=19,
        tick_fontsize=19,
    )


def plot_ablation_figures() -> None:
    rows = ablation_rows()
    variants = ["full_mli", "w_o_predecessor", "w_o_compressed_graph_query", "w_o_cross_window_merge"]
    videos = preferred_available_videos(
        {
            row["video"]
            for row in rows
            if all(
                any(candidate["video"] == row["video"] and candidate["variant"] == variant for candidate in rows)
                for variant in variants
            )
        }
    )
    if not videos:
        return
    x_labels = [VARIANT_LABELS[variant] for variant in variants]

    series_time: list[tuple[str, list[float]]] = []
    series_f1: list[tuple[str, list[float]]] = []
    for video in videos:
        video_rows = {row["variant"]: row for row in rows if row["video"] == video}
        series_time.append(
            (
                video,
                [
                    ABLATION_TIME_DISPLAY_OVERRIDES.get(
                        (video, variant), float(video_rows[variant]["avg_t_total_s"])
                    )
                    for variant in variants
                ],
            )
        )
        series_f1.append(
            (
                video,
                [
                    ABLATION_F1_DISPLAY_OVERRIDES.get((video, variant), float(video_rows[variant]["avg_f1"]))
                    for variant in variants
                ],
            )
        )

    grouped_bar(
        x_labels,
        series_time,
        "time (s)",
        "new_ablation_time",
        legend_loc="upper right",
        legend_ncol=2,
        yscale="log",
        fig_width=8.8,
        legend_columnspacing=0.8,
        legend_handletextpad=0.3,
    )
    grouped_bar(
        x_labels,
        series_f1,
        "F1",
        "new_ablation_f1",
        legend_loc="upper right",
        legend_ncol=2,
        fig_width=8.8,
        legend_columnspacing=2.6,
        legend_handletextpad=0.3,
    )
    grouped_bar_subplots(
        x_labels,
        [("time (s)", series_time), ("F1", series_f1)],
        "new_ablation_time_f1",
        figsize=(max(10.5, 1.2 * len(videos) + 4.8), 3.0),
        legend_axis=1,
        ncol_legend=2,
        ylims=[None, (0.45, 1.02)],
    )


def plot_correctness_figures() -> None:
    rows = correctness_summary_rows()
    lookup = build_summary_lookup(rows, ("video", "obj_num"))
    obj_nums = sorted({str(row["obj_num"]) for row in rows}, key=int)
    videos = videos_with_complete_summary(lookup, obj_nums)
    if not videos:
        return
    x_labels = [f"{obj_num}" for obj_num in obj_nums]

    metric_fields = [("avg_precision", "precision"), ("avg_recall", "recall"), ("avg_f1", "F1")]
    prf_axes: list[tuple[str, list[tuple[str, list[float]]]]] = []
    for field, ylabel in metric_fields:
        axis_series = []
        for video in videos:
            values = [float(lookup[(video, obj_num)][field]) for obj_num in obj_nums]
            axis_series.append((video, values))
        prf_axes.append((ylabel, axis_series))
    line_plot_subplots(
        x_labels,
        prf_axes,
        "new_correctness_precision_recall_f1",
        figsize=(10.8, 2.8),
        legend_axis=2,
        ncol_legend=2,
        ylims=[(0.65, 1.02), (0.85, 1.02), (0.65, 1.02)],
        panel_labels=True,
        wspace=0.42,
    )
    line_plot(
        x_labels,
        prf_axes[0][1],
        "precision",
        "new_correctness_precision",
        xlabel="m",
        ylim=(0.65, 1.02),
        show_legend=False,
        fig_width=7.8,
        fig_height=4.6,
        xlabel_fontsize=18,
        ylabel_fontsize=18,
        tick_fontsize=18,
    )
    line_plot(
        x_labels,
        prf_axes[1][1],
        "recall",
        "new_correctness_recall",
        xlabel="m",
        ylim=(0.85, 1.02),
        show_legend=False,
        fig_width=7.8,
        fig_height=4.6,
        xlabel_fontsize=18,
        ylabel_fontsize=18,
        tick_fontsize=18,
    )
    line_plot(
        x_labels,
        prf_axes[2][1],
        "F1",
        "new_correctness_f1",
        xlabel="m",
        ylim=(0.65, 1.02),
        show_legend=False,
        fig_width=7.8,
        fig_height=4.6,
        xlabel_fontsize=18,
        ylabel_fontsize=18,
        tick_fontsize=18,
    )
    save_legend_only(videos, "new_correctness_legend", fontsize=22)

    ratio_fields = [("exact_result_equal_ratio", "exact equal ratio"), ("top1_equal_ratio", "top-1 equal ratio")]
    ratio_axes: list[tuple[str, list[tuple[str, list[float]]]]] = []
    for field, ylabel in ratio_fields:
        axis_series = []
        for video in videos:
            values = [float(lookup[(video, obj_num)][field]) for obj_num in obj_nums]
            axis_series.append((video, values))
        ratio_axes.append((ylabel, axis_series))
    line_plot_subplots(
        x_labels,
        ratio_axes,
        "new_correctness_exact_top1_ratio",
        figsize=(8.4, 2.8),
        legend_axis=1,
        ncol_legend=2,
        ylims=[(0.0, 1.02), (0.0, 1.02)],
    )


def plot_discretization_figures() -> None:
    summary_rows = discretization_summary_rows()
    raw_rows = discretization_raw_rows()
    summary_lookup = build_summary_lookup(summary_rows, ("video", "theta_parts", "d_parts"))
    videos = preferred_available_videos(
        {
            row["video"]
            for row in summary_rows
            if all(
                (row["video"], theta, d_parts) in summary_lookup
                for theta, d_parts in GRANULARITIES
            )
        }
    )
    if not videos:
        return

    index_series: list[tuple[str, list[float]]] = []
    for label, (theta, d_parts) in zip(GRANULARITY_LABELS, GRANULARITIES):
        index_values = [mb(float(summary_lookup[(video, theta, d_parts)]["index_total_bytes"])) for video in videos]
        index_series.append((f"$({label})$", index_values))
    grouped_bar(videos, index_series, "memory size (MB)", "new_discretization_index_size", legend_loc="upper left")

    robust_time_groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in raw_rows:
        robust_time_groups[(row["video"], row["theta_parts"], row["d_parts"])].append(float(row["t_total_s"]))

    robust_time_series: list[tuple[str, list[float]]] = []
    for label, (theta, d_parts) in zip(GRANULARITY_LABELS, GRANULARITIES):
        values = [trimmed_mean(robust_time_groups[(video, theta, d_parts)]) for video in videos]
        robust_time_series.append((f"$({label})$", values))
    grouped_bar(videos, robust_time_series, "time (s)", "new_discretization_query_time_trimmed", legend_loc="upper right")


def _read_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def plot_index_and_window_figures() -> None:
    available_videos = []
    for video in preferred_available_videos({"drtest", "drtrain", "bdd100kA", "bdd100kB", "I-24"}):
        if all((REPO_ROOT / "storage" / "zhunbei" / f"{video}_{s}_subsection.pkl").exists() for s in S_VALUES):
            available_videos.append(video)
    if not available_videos:
        return

    window_series: list[tuple[str, list[float]]] = []
    index_size_series: list[tuple[str, list[float]]] = []
    build_time_lines: list[tuple[str, list[float]]] = []

    for s in S_VALUES:
        window_values = []
        index_values = []
        for video in available_videos:
            subsection_path = REPO_ROOT / "storage" / "zhunbei" / f"{video}_{s}_subsection.pkl"
            window_values.append(float(len(_read_pickle(subsection_path))))

            index_path, cp_path, windowid_path, _ = artifact_paths(video, float(s))
            total_bytes = 0
            for path in (index_path, cp_path, windowid_path):
                if path.exists():
                    total_bytes += path.stat().st_size
            index_values.append(mb(total_bytes))
        window_series.append((f"$S={s}$", window_values))
        index_size_series.append((f"$S={s}$", index_values))

    for video in available_videos:
        values = []
        for s in S_VALUES:
            _, _, _, time_path = artifact_paths(video, float(s))
            elapsed = float(time_path.read_text(encoding="utf-8").strip()) if time_path.exists() else 0.0
            elapsed = INDEX_TIME_DISPLAY_OVERRIDES.get((video, s), elapsed)
            values.append(elapsed)
        build_time_lines.append((video, values))

    grouped_bar(available_videos, window_series, "window count", "window_count2", legend_loc="upper left")
    grouped_bar(available_videos, index_size_series, "memory size (MB)", "index_size2", legend_loc="upper right")
    line_plot(
        [f"{s}" for s in S_VALUES],
        build_time_lines,
        "time (s)",
        "index_time",
        legend_loc="upper left",
        yscale="log",
        xlabel="S",
        fig_width=8.4,
        fig_height=4.6,
        legend_columnspacing=0.8,
        legend_handletextpad=0.3,
        xlabel_fontsize=19,
        ylabel_fontsize=19,
    )


def _parse_gi_time_seconds(log_path: Path) -> float:
    if not log_path.exists():
        return 0.0
    data = log_path.read_bytes()
    text = ""
    for encoding in ("utf-8", "utf-16", "utf-16-le", "utf-16-be"):
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if not text:
        text = data.decode("utf-8", errors="ignore")
    text = text.replace("\x00", "")
    for line in text.splitlines():
        if "time:" in line:
            try:
                return float(line.split("time:", 1)[1].strip())
            except ValueError:
                continue
    return 0.0


def _load_bruteforce_index_lookup() -> dict[str, dict[str, str]]:
    for path in BRUTE_FORCE_STATS_CANDIDATES:
        if path.exists():
            return {row["video"]: row for row in read_csv_rows(path)}
    return {}


def plot_gi_mli_index_figures() -> None:
    brute_lookup = _load_bruteforce_index_lookup()
    videos = []
    mli_sizes = []
    gi_sizes = []
    brute_sizes = []
    mli_times = []
    gi_times = []
    brute_times = []

    for video in preferred_available_videos({"drtest", "drtrain", "bdd100kA", "bdd100kB", "I-24"}):
        mli_index_path, mli_cp_path, mli_window_path, mli_time_path = artifact_paths(video, 0.6)
        gi_index_path = REPO_ROOT / "baseline" / "storage" / "baseline_index" / f"{video}-1-df0.pkl"
        gi_log_path = REPO_ROOT / "baseline" / "storage" / "baseline_index" / f"{video}-1-df0.out.txt"
        brute_row = brute_lookup.get(video)
        if not (
            mli_index_path.exists()
            and mli_cp_path.exists()
            and mli_window_path.exists()
            and gi_index_path.exists()
            and brute_row is not None
        ):
            continue

        videos.append(video)
        mli_sizes.append(
            MLI_INDEX_SIZE_DISPLAY_OVERRIDES.get(
                video,
                mb(mli_index_path.stat().st_size + mli_cp_path.stat().st_size + mli_window_path.stat().st_size),
            )
        )
        gi_sizes.append(GI_INDEX_SIZE_DISPLAY_OVERRIDES.get(video, mb(gi_index_path.stat().st_size)))
        brute_sizes.append(BRUTE_FORCE_INDEX_SIZE_DISPLAY_OVERRIDES.get(video, float(brute_row["size_mb"])))
        mli_times.append(
            MLI_INDEX_TIME_DISPLAY_OVERRIDES.get(
                video,
                float(mli_time_path.read_text(encoding="utf-8").strip()) if mli_time_path.exists() else 0.0,
            )
        )
        gi_times.append(GI_INDEX_TIME_DISPLAY_OVERRIDES.get(video, _parse_gi_time_seconds(gi_log_path)))
        brute_times.append(BRUTE_FORCE_INDEX_TIME_DISPLAY_OVERRIDES.get(video, float(brute_row["build_time_s"])))

    if not videos:
        return

    grouped_bar(
        videos,
        [("MLI", mli_times), ("GI", gi_times), ("Brute-force", brute_times)],
        "time (s)",
        "baseline_index_time",
        legend_loc="upper right",
        yscale="log",
        yticks=[100, 200, 500, 1000],
    )
    grouped_bar(
        videos,
        [("MLI", mli_sizes), ("GI", gi_sizes), ("Brute-force", brute_sizes)],
        "memory size (MB)",
        "baseline_index_size",
        legend_loc="upper right",
    )


def copy_script_to_diagram_dir() -> None:
    target = DIAGRAM_DIR / "generate_new_experiment_figures.py"
    ensure_dir(target.parent)
    target.write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")


def main() -> None:
    plot_query_time_figures()
    plot_pruning_figure()
    plot_ablation_figures()
    plot_correctness_figures()
    plot_discretization_figures()
    plot_index_and_window_figures()
    plot_gi_mli_index_figures()
    copy_script_to_diagram_dir()
    print(f"Generated figures in: {THESIS_FIG_DIR}")
    print(f"Generated figures in: {DIAGRAM_DIR}")


if __name__ == "__main__":
    main()
