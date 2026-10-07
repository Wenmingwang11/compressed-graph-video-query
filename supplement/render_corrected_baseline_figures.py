from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from supplement.dataset_config import PREFERRED_VIDEO_ORDER


METHODS = (
    "Ours",
    "VQPy-style",
    "EVA-style",
    "VOCAL-UDF-style",
    "Video-ColBERT-style",
    "Seiden-style",
    "LAVA-style",
    "STAR-style",
)

COLORS = (
    "#2F6B4F",
    "#D97706",
    "#3973AC",
    "#B5495B",
    "#6B5CA5",
    "#008B8B",
    "#7A6A53",
    "#555555",
)


def _read_validated_rows(summary_path: str | Path) -> list[dict[str, str]]:
    path = Path(summary_path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("benchmark summary is empty")
    for row in rows:
        if row.get("source", "").strip().lower() != "measured":
            raise ValueError("all plotted rows must come from measured runs")
        if row.get("signature_verified", "").strip().lower() not in {"1", "true", "yes"}:
            raise ValueError("all plotted rows must have a verified result signature")
        if float(row["median_s"]) <= 0:
            raise ValueError("log-scale query times must be positive")
    _validate_provenance(path, rows)
    return rows


def _validate_provenance(summary_path: Path, summary_rows: list[dict[str, str]]) -> None:
    raw_path = summary_path.with_name("raw_measurements.csv")
    manifest_path = summary_path.with_name("run_manifest.json")
    if not raw_path.exists() or not manifest_path.exists():
        raise ValueError("summary must be accompanied by raw_measurements.csv and run_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True:
        raise ValueError("benchmark manifest is not marked complete")
    repetitions = int(manifest["repetitions"])
    oracle_signatures = manifest["oracle_signatures"]

    with raw_path.open("r", encoding="utf-8", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    raw_groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in raw_rows:
        if row.get("signature_verified", "").strip().lower() not in {"1", "true", "yes"}:
            raise ValueError("raw measurement contains an unverified signature")
        key = (row["dataset"], row["query_id"], row["method"])
        raw_groups.setdefault(key, []).append(row)

    summary_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in summary_rows:
        key = (row["dataset"], row["query_id"], row["method"])
        if key in summary_by_key:
            raise ValueError(f"duplicate summary row for {key}")
        summary_by_key[key] = row
    if set(summary_by_key) != set(raw_groups):
        raise ValueError("summary and raw measurement groups do not match")

    for key, row in summary_by_key.items():
        dataset, query_id, _ = key
        group = raw_groups[key]
        if len(group) != repetitions:
            raise ValueError(f"raw repetition count does not match manifest for {key}")
        signatures = {item["signature"] for item in group}
        expected_signature = str(oracle_signatures[dataset][query_id])
        if signatures != {expected_signature} or row["signature"] != expected_signature:
            raise ValueError(f"signature provenance mismatch for {key}")
        values = np.asarray([float(item["elapsed_s"]) for item in group], dtype=float)
        expected_values = {
            "median_s": float(np.median(values)),
            "q1_s": float(np.quantile(values, 0.25)),
            "q3_s": float(np.quantile(values, 0.75)),
        }
        for column, expected in expected_values.items():
            if not math.isclose(float(row[column]), expected, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(f"summary statistic does not match raw measurements for {key}: {column}")


def _validate_coverage(rows: list[dict[str, str]], query_id: str) -> dict[tuple[str, str], dict[str, str]]:
    selected: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        if row["query_id"].upper() != query_id.upper():
            continue
        key = (row["dataset"], row["method"])
        if key in selected:
            raise ValueError(f"duplicate {query_id} row for {key}")
        selected[key] = row
    expected = {(dataset, method) for dataset in PREFERRED_VIDEO_ORDER for method in METHODS}
    missing = sorted(expected.difference(selected))
    extra = sorted(set(selected).difference(expected))
    if missing or extra:
        raise ValueError(f"invalid {query_id} method coverage: missing={missing}, extra={extra}")
    return selected


def _render_one(rows: list[dict[str, str]], query_id: str, output_dir: Path) -> list[Path]:
    selected = _validate_coverage(rows, query_id)
    x = np.arange(len(PREFERRED_VIDEO_ORDER), dtype=float)
    group_width = 0.84
    bar_width = group_width / len(METHODS)

    fig, ax = plt.subplots(figsize=(10.4, 4.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.16, top=0.78)
    for method_index, (method, color) in enumerate(zip(METHODS, COLORS)):
        medians = np.array(
            [float(selected[(dataset, method)]["median_s"]) for dataset in PREFERRED_VIDEO_ORDER]
        )
        q1 = np.array([float(selected[(dataset, method)]["q1_s"]) for dataset in PREFERRED_VIDEO_ORDER])
        q3 = np.array([float(selected[(dataset, method)]["q3_s"]) for dataset in PREFERRED_VIDEO_ORDER])
        offset = -group_width / 2.0 + (method_index + 0.5) * bar_width
        ax.bar(
            x + offset,
            medians,
            width=bar_width * 0.92,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            label=method,
            yerr=np.vstack((np.maximum(0.0, medians - q1), np.maximum(0.0, q3 - medians))),
            error_kw={"elinewidth": 0.65, "capsize": 1.5, "capthick": 0.65},
        )

    ax.set_yscale("log")
    ax.set_ylabel("Query time (s)")
    ax.set_xlabel("Dataset")
    ax.set_xticks(x, PREFERRED_VIDEO_ORDER)
    ax.grid(axis="y", which="both", color="#D5D5D5", linewidth=0.55, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=4,
        frameon=False,
        fontsize=8.2,
        columnspacing=1.3,
        handlelength=1.4,
    )

    stem = f"baseline_query_time_{query_id.lower()}_corrected"
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, facecolor="white", transparent=False)
    fig.savefig(png_path, dpi=240, facecolor="white", transparent=False)
    plt.close(fig)
    return [pdf_path, png_path]


def render_figures(summary_path: str | Path, output_dir: str | Path) -> list[Path]:
    rows = _read_validated_rows(summary_path)
    destination = Path(output_dir)
    return _render_one(rows, "Q1", destination) + _render_one(rows, "Q2", destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render validated corrected-baseline figures.")
    parser.add_argument("summary", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    for output in render_figures(args.summary, args.output_dir):
        print(output)


if __name__ == "__main__":
    main()
