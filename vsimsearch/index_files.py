from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple


def resolve_index_paths(
    video_name: str,
    s: Optional[float],
    base_dir: str | Path = './storage/index',
) -> Tuple[Path, Path, Optional[Path]]:
    """
    Resolve index file paths with backward-compatible naming.

    Preferred naming (new):
      {video}_{s}_index.pkl
      {video}_{s}_windowid_index.pkl
      {video}_{s}_cp_graphs.pkl

    Legacy naming (old):
      {video}_index.pkl
      {video}_windowid_index.pkl
      {video}_cp_graphs.pkl
    """
    base = Path(base_dir)
    s_part = None if s is None else str(s)

    def pick(kind: str) -> Path:
        if s_part is not None:
            p = base / f'{video_name}_{s_part}_{kind}.pkl'
            if p.exists():
                return p
        return base / f'{video_name}_{kind}.pkl'

    index_path = pick('index')
    windowid_index_path = pick('windowid_index')
    cp_graphs_path = pick('cp_graphs')
    return index_path, windowid_index_path, (cp_graphs_path if cp_graphs_path.exists() else None)
