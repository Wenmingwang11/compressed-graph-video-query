from __future__ import annotations

import sys
import argparse
import csv
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

# Ensure repo root is on sys.path when running as a script: `python supplement/index_stats.py`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vsimsearch.index_files import resolve_index_paths

from supplement.metrics_utils import deep_getsizeof


@dataclass
class IndexStatRow:
    video: str
    s: str
    index_file_bytes: int
    windowid_index_file_bytes: int
    cp_graphs_file_bytes: int
    windows: int
    avg_types_per_window: float
    avg_nodes_per_window: float
    windowid_windows: int
    avg_pairs_per_window: float
    avg_theta_d_bins_per_window: float
    avg_frames_per_theta_d_bin: float
    deep_size_index_bytes: int = 0
    deep_size_windowid_bytes: int = 0


def _safe_stat_bytes(p: Optional[Path]) -> int:
    if p is None or not p.exists():
        return 0
    return p.stat().st_size


def main():
    parser = argparse.ArgumentParser('Collect index statistics (paper supplement)')
    parser.add_argument('--videos', type=str, default='drtest,drtrain', help='comma-separated video names')
    parser.add_argument('--s', type=str, default='0.6', help='comma-separated S values; use "" for legacy naming only')
    parser.add_argument('--out', type=str, default='supplement/out/index_stats.csv')
    parser.add_argument('--load', action='store_true', help='load pickles and estimate deep size (slower)')
    args = parser.parse_args()

    videos = [v.strip() for v in args.videos.split(',') if v.strip()]
    s_values: list[Optional[float]] = []
    for token in args.s.split(','):
        token = token.strip()
        if token == '':
            s_values.append(None)
        else:
            s_values.append(float(token))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[IndexStatRow] = []
    for video in videos:
        for s in s_values:
            index_path, windowid_index_path, cp_graphs_path = resolve_index_paths(video, s)

            index_bytes = _safe_stat_bytes(index_path)
            windowid_bytes = _safe_stat_bytes(windowid_index_path)
            cp_bytes = _safe_stat_bytes(cp_graphs_path)

            windows = 0
            avg_types = 0.0
            avg_nodes = 0.0

            win_windows = 0
            avg_pairs = 0.0
            avg_bins = 0.0
            avg_frames_per_bin = 0.0

            deep_index = 0
            deep_windowid = 0

            if args.load:
                with open(index_path, 'rb') as f:
                    index = pickle.load(f)
                windows = len(index)
                total_types = sum(len(window_data) for window_data in index.values()) if windows else 0
                total_nodes = (
                    sum(len(node_ids) for window_data in index.values() for node_ids in window_data.values())
                    if windows
                    else 0
                )
                avg_types = (total_types / windows) if windows else 0.0
                avg_nodes = (total_nodes / windows) if windows else 0.0
                deep_index = deep_getsizeof(index)

                with open(windowid_index_path, 'rb') as f:
                    windowid_index = pickle.load(f)
                win_windows = len(windowid_index)
                total_pairs = 0
                total_bins = 0
                total_frames = 0
                for window_edge_index in windowid_index.values():
                    total_pairs += len(window_edge_index)
                    for theta_d_map in window_edge_index.values():
                        total_bins += len(theta_d_map)
                        for frame_set in theta_d_map.values():
                            total_frames += len(frame_set)
                avg_pairs = (total_pairs / win_windows) if win_windows else 0.0
                avg_bins = (total_bins / win_windows) if win_windows else 0.0
                avg_frames_per_bin = (total_frames / total_bins) if total_bins else 0.0
                deep_windowid = deep_getsizeof(windowid_index)

            row = IndexStatRow(
                video=video,
                s='' if s is None else str(s),
                index_file_bytes=index_bytes,
                windowid_index_file_bytes=windowid_bytes,
                cp_graphs_file_bytes=cp_bytes,
                windows=windows,
                avg_types_per_window=avg_types,
                avg_nodes_per_window=avg_nodes,
                windowid_windows=win_windows,
                avg_pairs_per_window=avg_pairs,
                avg_theta_d_bins_per_window=avg_bins,
                avg_frames_per_theta_d_bin=avg_frames_per_bin,
                deep_size_index_bytes=deep_index,
                deep_size_windowid_bytes=deep_windowid,
            )
            rows.append(row)

    with out_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()) if rows else [])
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    print(f'Wrote {len(rows)} rows to {out_path}')
    if not args.load:
        print('Tip: add --load to compute deep size and structural stats (slower).')


if __name__ == '__main__':
    main()
