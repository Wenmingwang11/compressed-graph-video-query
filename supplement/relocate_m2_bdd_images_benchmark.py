from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from supplement.relocate_m2_benchmark import DetectionRow, PairConstraint, QueryInstance, find_query_instance

DEFAULT_IMAGE_ROOT = Path(r"D:\data\bdd100k_images_100k\100k\train")
DEFAULT_LABEL_ZIP = Path(r"D:\data\bdd100k_labels.zip")
DEFAULT_OUT_DIR = REPO_ROOT / "supplement" / "out" / "relocate_m2_bdd_images"
DEFAULT_FRAME_SIZE = (1280, 720)

BDD_CATEGORY_TO_CLASS = {
    "person": 0,
    "bike": 1,
    "car": 2,
    "motor": 3,
    "bus": 5,
    "train": 6,
    "truck": 7,
    "traffic light": 81,
    "traffic sign": 82,
    "rider": 83,
}


@dataclass(frozen=True)
class ImageFrame:
    frame_id: int
    stem: str
    image_path: Path


@dataclass(frozen=True)
class CandidatePair:
    frame_id: int
    stem: str
    left: DetectionRow
    right: DetectionRow
    left_score: float
    right_score: float
    total_score: float


def parse_pair_constraint(text: str) -> PairConstraint:
    roles_text, values_text = text.strip().split(":")
    left_role, right_role = [int(token) for token in roles_text.split("-")]
    theta_text, d_text = [token.strip() for token in values_text.split(",")]
    return PairConstraint(
        left_role=left_role,
        right_role=right_role,
        theta_lt=float(theta_text),
        d_ratio_lt=float(d_text),
    )


def pair_satisfies_constraint(
    left: DetectionRow,
    right: DetectionRow,
    theta_lt: float,
    d_ratio_lt: float,
    frame_width: int,
    frame_height: int,
) -> bool:
    x1, y1 = left.center
    x2, y2 = right.center
    theta = math.atan2(y2 - y1, x2 - x1)
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    base_distance = float(frame_width + frame_height)
    d_ratio = distance / base_distance if base_distance else 0.0
    return theta < theta_lt and d_ratio < d_ratio_lt


def iter_local_bdd_images(image_root: Path) -> Iterable[ImageFrame]:
    for frame_id, image_path in enumerate(sorted(image_root.glob("*.jpg")), start=1):
        yield ImageFrame(frame_id=frame_id, stem=image_path.stem, image_path=image_path)


def read_bdd_frame_detections(
    image_root: Path,
    label_zip_path: Path,
    allowed_classes: set[int],
) -> tuple[list[ImageFrame], list[DetectionRow]]:
    frames: list[ImageFrame] = []
    detections: list[DetectionRow] = []
    with zipfile.ZipFile(label_zip_path) as zf:
        for frame in iter_local_bdd_images(image_root):
            label_name = f"100k/train/{frame.stem}.json"
            try:
                payload = json.loads(zf.read(label_name))
            except KeyError:
                continue
            frame_entries = payload.get("frames", [])
            if not frame_entries:
                continue
            objects = frame_entries[0].get("objects", [])
            frame_rows: list[DetectionRow] = []
            for local_idx, item in enumerate(objects):
                category = item.get("category")
                box = item.get("box2d")
                if category not in BDD_CATEGORY_TO_CLASS or not box:
                    continue
                cls = BDD_CATEGORY_TO_CLASS[category]
                if cls not in allowed_classes:
                    continue
                x1 = float(box["x1"])
                y1 = float(box["y1"])
                x2 = float(box["x2"])
                y2 = float(box["y2"])
                width = max(1.0, x2 - x1)
                height = max(1.0, y2 - y1)
                frame_rows.append(
                    DetectionRow(
                        frame=frame.frame_id,
                        object_id=local_idx + 1,
                        left=x1,
                        top=y1,
                        width=width,
                        height=height,
                        cls=cls,
                    )
                )
            if frame_rows:
                frames.append(frame)
                detections.extend(frame_rows)
    return frames, detections


@lru_cache(maxsize=256)
def load_image(path_str: str) -> Image.Image:
    return Image.open(path_str).convert("RGB")


def crop_detection(image: Image.Image, det: DetectionRow) -> Image.Image:
    width, height = image.size
    x1 = max(0, min(width - 1, int(det.left)))
    y1 = max(0, min(height - 1, int(det.top)))
    x2 = max(x1 + 1, min(width, int(math.ceil(det.left + det.width))))
    y2 = max(y1 + 1, min(height, int(math.ceil(det.top + det.height))))
    return image.crop((x1, y1, x2, y2))


def _corrcoef_safe(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype("float32").reshape(-1)
    b = b.astype("float32").reshape(-1)
    if np.array_equal(a, b):
        return 1.0
    a_std = float(a.std())
    b_std = float(b.std())
    if a_std == 0.0 or b_std == 0.0:
        return 0.0
    a = a - float(a.mean())
    b = b - float(b.mean())
    return float((a * b).mean() / (a_std * b_std))


def crop_similarity(query_crop: Image.Image, candidate_crop: Image.Image) -> float:
    query_resized = query_crop.resize((64, 64), Image.Resampling.BILINEAR)
    candidate_resized = candidate_crop.resize((64, 64), Image.Resampling.BILINEAR)

    q_hsv = np.asarray(query_resized.convert("HSV"), dtype=np.float32)
    c_hsv = np.asarray(candidate_resized.convert("HSV"), dtype=np.float32)

    hist_q, _, _ = np.histogram2d(
        q_hsv[..., 0].reshape(-1),
        q_hsv[..., 1].reshape(-1),
        bins=(16, 16),
        range=((0, 255), (0, 255)),
    )
    hist_c, _, _ = np.histogram2d(
        c_hsv[..., 0].reshape(-1),
        c_hsv[..., 1].reshape(-1),
        bins=(16, 16),
        range=((0, 255), (0, 255)),
    )
    hist_score = _corrcoef_safe(hist_q, hist_c)

    q_gray = np.asarray(query_resized.convert("L"), dtype=np.float32)
    c_gray = np.asarray(candidate_resized.convert("L"), dtype=np.float32)
    ncc = _corrcoef_safe(q_gray, c_gray)
    return 0.5 * hist_score + 0.5 * ncc


def build_frame_lookup(frames: Sequence[ImageFrame]) -> dict[int, ImageFrame]:
    return {frame.frame_id: frame for frame in frames}


def iter_valid_candidate_pairs(
    detections: Iterable[DetectionRow],
    query_types: Sequence[int],
    pair_constraint: PairConstraint,
    frame_width: int,
    frame_height: int,
) -> Iterable[tuple[int, DetectionRow, DetectionRow]]:
    by_frame: dict[int, list[DetectionRow]] = {}
    query_type_set = set(query_types)
    for det in detections:
        if det.cls in query_type_set:
            by_frame.setdefault(det.frame, []).append(det)

    left_type = query_types[pair_constraint.left_role]
    right_type = query_types[pair_constraint.right_role]

    for frame_id in sorted(by_frame):
        frame_dets = by_frame[frame_id]
        lefts = [det for det in frame_dets if det.cls == left_type]
        rights = [det for det in frame_dets if det.cls == right_type]
        if not lefts or not rights:
            continue
        for left in lefts:
            for right in rights:
                if left.object_id == right.object_id:
                    continue
                if pair_satisfies_constraint(
                    left,
                    right,
                    pair_constraint.theta_lt,
                    pair_constraint.d_ratio_lt,
                    frame_width,
                    frame_height,
                ):
                    yield frame_id, left, right


def write_query_crops(
    out_dir: Path,
    frame_lookup: dict[int, ImageFrame],
    query: QueryInstance,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    image = load_image(str(frame_lookup[query.frame].image_path))
    for role_idx, det in enumerate(query.roles):
        crop = crop_detection(image, det)
        crop.save(out_dir / f"bdd_images_query_role{role_idx}_cls{det.cls}_{frame_lookup[query.frame].stem}.jpg")


def append_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def status_row(
    *,
    status: str,
    image_count: int | str = "",
    detection_count: int | str = "",
    left_category: str = "",
    right_category: str = "",
    query_frame: int | str = "",
    query_image: str = "",
    elapsed_s: float | str = "",
    valid_pair_count: int | str = "",
    note: str = "",
) -> dict[str, object]:
    return {
        "dataset": "bdd100kTrainCombined",
        "status": status,
        "image_count": image_count,
        "detection_count": detection_count,
        "left_category": left_category,
        "right_category": right_category,
        "query_frame": query_frame,
        "query_image": query_image,
        "elapsed_s": elapsed_s,
        "valid_pair_count": valid_pair_count,
        "note": note,
    }


def benchmark(
    frames: Sequence[ImageFrame],
    detections: Sequence[DetectionRow],
    query_types: Sequence[int],
    pair_constraint: PairConstraint,
    topk: int,
    frame_width: int,
    frame_height: int,
    left_category: str,
    right_category: str,
    progress_every: int = 5000,
) -> tuple[QueryInstance, list[CandidatePair], dict[str, object]]:
    frame_lookup = build_frame_lookup(frames)
    query = find_query_instance(
        detections,
        query_types=query_types,
        pair_constraint=pair_constraint,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    query_image = load_image(str(frame_lookup[query.frame].image_path))
    query_left_crop = crop_detection(query_image, query.roles[pair_constraint.left_role])
    query_right_crop = crop_detection(query_image, query.roles[pair_constraint.right_role])

    start = time.perf_counter()
    candidates: list[CandidatePair] = []
    valid_pair_count = 0

    for idx, (frame_id, left, right) in enumerate(
        iter_valid_candidate_pairs(
            detections,
            query_types=query_types,
            pair_constraint=pair_constraint,
            frame_width=frame_width,
            frame_height=frame_height,
        ),
        start=1,
    ):
        if frame_id == query.frame and left.object_id == query.roles[pair_constraint.left_role].object_id and right.object_id == query.roles[pair_constraint.right_role].object_id:
            continue
        valid_pair_count += 1
        image = load_image(str(frame_lookup[frame_id].image_path))
        left_crop = crop_detection(image, left)
        right_crop = crop_detection(image, right)
        left_score = crop_similarity(query_left_crop, left_crop)
        right_score = crop_similarity(query_right_crop, right_crop)
        candidates.append(
            CandidatePair(
                frame_id=frame_id,
                stem=frame_lookup[frame_id].stem,
                left=left,
                right=right,
                left_score=left_score,
                right_score=right_score,
                total_score=left_score + right_score,
            )
        )
        if idx % progress_every == 0:
            print(f"  scanned_pairs={idx} kept_candidates={len(candidates)}", flush=True)

    elapsed = time.perf_counter() - start
    candidates.sort(key=lambda item: item.total_score, reverse=True)
    top_candidates = candidates[:topk]
    summary = {
        "query_id": "bdd_images_q1",
        "dataset": "bdd100kTrainCombined",
        "temporal_length": 1,
        "temporal_mode": "single_frame_non_contiguous",
        "query_frame": query.frame,
        "query_image": frame_lookup[query.frame].stem,
        "query_types": ",".join(map(str, query_types)),
        "pair_constraint": f"{pair_constraint.left_role}-{pair_constraint.right_role}:{pair_constraint.theta_lt},{pair_constraint.d_ratio_lt}",
        "image_count": len(frames),
        "valid_pair_count": valid_pair_count,
        "topk_returned": len(top_candidates),
        "elapsed_s": elapsed,
        "top1_frame": "" if not top_candidates else top_candidates[0].frame_id,
        "top1_image": "" if not top_candidates else top_candidates[0].stem,
        "top1_score": "" if not top_candidates else top_candidates[0].total_score,
        "left_category": left_category,
        "right_category": right_category,
    }
    return query, top_candidates, summary


def candidate_rows(candidates: Sequence[CandidatePair]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rank, candidate in enumerate(candidates, start=1):
        rows.append(
            {
                "query_id": "bdd_images_q1",
                "dataset": "bdd100kTrainCombined",
                "rank": rank,
                "frame_id": candidate.frame_id,
                "image_name": candidate.stem,
                "left_object_id": candidate.left.object_id,
                "left_class": candidate.left.cls,
                "right_object_id": candidate.right.object_id,
                "right_class": candidate.right.cls,
                "left_score": candidate.left_score,
                "right_score": candidate.right_score,
                "total_score": candidate.total_score,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser("Run a RELOCATE-style m=2 single-frame benchmark on local BDD images")
    parser.add_argument("--image_root", default=str(DEFAULT_IMAGE_ROOT))
    parser.add_argument("--label_zip", default=str(DEFAULT_LABEL_ZIP))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--left_category", default="bus")
    parser.add_argument("--right_category", default="traffic sign")
    parser.add_argument("--pair_constraint", default="0-1:10,8")
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = out_dir / "relocate_m2_bdd_images_raw.csv"
    summary_csv = out_dir / "relocate_m2_bdd_images_summary.csv"
    status_csv = out_dir / "relocate_m2_bdd_images_status.csv"
    crop_dir = out_dir / "query_crops"
    for path in (raw_csv, summary_csv, status_csv):
        if path.exists():
            path.unlink()
    if crop_dir.exists():
        for item in crop_dir.iterdir():
            if item.is_file():
                item.unlink()

    left_cls = BDD_CATEGORY_TO_CLASS[args.left_category]
    right_cls = BDD_CATEGORY_TO_CLASS[args.right_category]
    allowed_classes = {left_cls, right_cls}
    pair_constraint = parse_pair_constraint(args.pair_constraint)

    print(
        f"loading BDD image dataset from {args.image_root} with categories=({args.left_category}, {args.right_category})",
        flush=True,
    )
    frames, detections = read_bdd_frame_detections(
        image_root=Path(args.image_root),
        label_zip_path=Path(args.label_zip),
        allowed_classes=allowed_classes,
    )
    append_rows(
        status_csv,
        [
            status_row(
                status="loaded",
                image_count=len(frames),
                detection_count=len(detections),
                left_category=args.left_category,
                right_category=args.right_category,
            )
        ],
    )
    print(f"loaded_images={len(frames)} detections={len(detections)}", flush=True)

    query, top_candidates, summary = benchmark(
        frames=frames,
        detections=detections,
        query_types=[left_cls, right_cls],
        pair_constraint=pair_constraint,
        topk=args.topk,
        frame_width=DEFAULT_FRAME_SIZE[0],
        frame_height=DEFAULT_FRAME_SIZE[1],
        left_category=args.left_category,
        right_category=args.right_category,
    )

    write_query_crops(crop_dir, build_frame_lookup(frames), query)
    append_rows(raw_csv, candidate_rows(top_candidates))
    append_rows(summary_csv, [summary])
    append_rows(
        status_csv,
        [
            status_row(
                status="success",
                image_count=len(frames),
                detection_count=len(detections),
                left_category=args.left_category,
                right_category=args.right_category,
                query_frame=query.frame,
                query_image=build_frame_lookup(frames)[query.frame].stem,
                elapsed_s=summary["elapsed_s"],
                valid_pair_count=summary["valid_pair_count"],
            )
        ],
    )
    print(
        f"query_image={summary['query_image']} valid_pairs={summary['valid_pair_count']} elapsed={float(summary['elapsed_s']):.4f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
