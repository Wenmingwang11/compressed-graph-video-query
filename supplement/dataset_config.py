from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
THESIS_FIG_DIR = Path(r"<LOCAL_USER_HOME>\Desktop\论文\latex-project\figures")
DIAGRAM_DIR = REPO_ROOT / "supplement" / "out" / "rendered_figures"

PREFERRED_VIDEO_ORDER = ["drtest", "drtrain", "bdd100kA", "bdd100kB", "I-24"]

DATASET_META: dict[str, dict[str, object]] = {
    "drtest": {
        "storage_file": "drtest.txt",
        "frame_width": 960,
        "frame_height": 540,
        "label": "drtest",
    },
    "drtrain": {
        "storage_file": "drtrain.txt",
        "frame_width": 960,
        "frame_height": 540,
        "label": "drtrain",
    },
    "bdd100kA": {
        "storage_file": "bdd100kA.txt",
        "frame_width": 1280,
        "frame_height": 720,
        "label": "bdd100kA",
    },
    "bdd100kB": {
        "storage_file": "bdd100kB.txt",
        "frame_width": 1280,
        "frame_height": 720,
        "label": "bdd100kB",
    },
    "I-24": {
        "storage_file": "I-24.txt",
        "frame_width": 1920,
        "frame_height": 1080,
        "label": "I-24",
        "video_path": r"D:\data\I-24.MP4",
        "source_raw_path": r"D:\pycharm\star-master\storage\results_new\vsim-all\raw\I-24.txt",
    },
}


def dataset_storage_path(video: str) -> Path:
    return REPO_ROOT / "storage" / str(DATASET_META[video]["storage_file"])


def dataset_frame_size(video: str) -> tuple[int, int]:
    meta = DATASET_META[video]
    return int(meta["frame_width"]), int(meta["frame_height"])


def preferred_available_videos(candidates: list[str] | set[str]) -> list[str]:
    candidate_set = set(candidates)
    ordered = [video for video in PREFERRED_VIDEO_ORDER if video in candidate_set]
    for video in sorted(candidate_set):
        if video not in ordered:
            ordered.append(video)
    return ordered
