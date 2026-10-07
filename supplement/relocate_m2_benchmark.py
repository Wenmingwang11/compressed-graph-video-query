from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Iterable, Iterator, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUERY_SPECS = REPO_ROOT / "supplement" / "baseline_formal_queries.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "supplement" / "out" / "relocate_m2"
DEFAULT_DETRAC_ROOTS = {
    "drtest": Path(r"D:\data\DETRAC-test-data\Insight-MVT_Annotation_Test"),
    "drtrain": Path(r"D:\data\DETRAC-train-data\Insight-MVT_Annotation_Train"),
}
DEFAULT_FRAME_SIZE = {
    "drtest": (960, 540),
    "drtrain": (960, 540),
}


@dataclass(frozen=True)
class DetectionRow:
    frame: int
    object_id: int
    left: float
    top: float
    width: float
    height: float
    cls: int

    @property
    def center(self) -> tuple[float, float]:
        return self.left + self.width / 2.0, self.top + self.height / 2.0


@dataclass(frozen=True)
class PairConstraint:
    left_role: int
    right_role: int
    theta_lt: float
    d_ratio_lt: float


@dataclass(frozen=True)
class FrameSpan:
    video_name: str
    image_count: int
    usable_frames: int
    global_start: int
    global_end: int


@dataclass(frozen=True)
class FrameRef:
    video_name: str
    global_frame: int
    local_frame: int
    image_path: Path


@dataclass(frozen=True)
class QueryInstance:
    frame: int
    roles: tuple[DetectionRow, ...]


@dataclass(frozen=True)
class PairSequence:
    start_frame: int
    end_frame: int
    frames: tuple[int, ...]
    left_dets: tuple[DetectionRow, ...]
    right_dets: tuple[DetectionRow, ...]
    left_object_id: int
    right_object_id: int


@dataclass(frozen=True)
class QuerySequenceInstance:
    start_frame: int
    end_frame: int
    frames: tuple[int, ...]
    left_dets: tuple[DetectionRow, ...]
    right_dets: tuple[DetectionRow, ...]
    left_object_id: int
    right_object_id: int


@dataclass(frozen=True)
class QuerySpec:
    query_id: str
    dataset: str
    query_types: tuple[int, ...]
    pair_constraint: PairConstraint
    frame_threshold: int
    topk: int
    note: str


@dataclass(frozen=True)
class CandidateSequence:
    start_frame: int
    end_frame: int
    frames: tuple[int, ...]
    left_dets: tuple[DetectionRow, ...]
    right_dets: tuple[DetectionRow, ...]
    left_score: float
    right_score: float
    total_score: float


def build_frame_spans(sequence_counts: Sequence[tuple[str, int]]) -> list[FrameSpan]:
    spans: list[FrameSpan] = []
    next_global = 1
    for video_name, image_count in sequence_counts:
        usable = max(0, image_count - 1)
        if usable == 0:
            continue
        spans.append(
            FrameSpan(
                video_name=video_name,
                image_count=image_count,
                usable_frames=usable,
                global_start=next_global,
                global_end=next_global + usable - 1,
            )
        )
        next_global += usable
    return spans


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


def find_query_instance(
    detections: Iterable[DetectionRow],
    query_types: Sequence[int],
    pair_constraint: PairConstraint,
    frame_width: int,
    frame_height: int,
) -> QueryInstance:
    by_frame: dict[int, list[DetectionRow]] = {}
    for det in detections:
        by_frame.setdefault(det.frame, []).append(det)

    for frame in sorted(by_frame):
        frame_dets = by_frame[frame]
        per_role = [[det for det in frame_dets if det.cls == query_type] for query_type in query_types]
        if any(len(role_dets) == 0 for role_dets in per_role):
            continue

        for roles in product(*per_role):
            object_ids = {det.object_id for det in roles}
            if len(object_ids) != len(roles):
                continue
            left = roles[pair_constraint.left_role]
            right = roles[pair_constraint.right_role]
            if pair_satisfies_constraint(
                left,
                right,
                pair_constraint.theta_lt,
                pair_constraint.d_ratio_lt,
                frame_width,
                frame_height,
            ):
                return QueryInstance(frame=frame, roles=tuple(roles))

    raise ValueError("could not find a valid m=2 query instance")


def _valid_pairs_by_frame(
    detections: Iterable[DetectionRow],
    query_types: Sequence[int],
    pair_constraint: PairConstraint,
    frame_width: int,
    frame_height: int,
) -> dict[int, dict[tuple[int, int], tuple[DetectionRow, DetectionRow]]]:
    by_frame: dict[int, list[DetectionRow]] = {}
    query_type_set = set(query_types)
    for det in detections:
        if det.cls in query_type_set:
            by_frame.setdefault(det.frame, []).append(det)

    left_type = query_types[pair_constraint.left_role]
    right_type = query_types[pair_constraint.right_role]
    valid_pairs: dict[int, dict[tuple[int, int], tuple[DetectionRow, DetectionRow]]] = {}
    for frame, frame_dets in by_frame.items():
        lefts = [det for det in frame_dets if det.cls == left_type]
        rights = [det for det in frame_dets if det.cls == right_type]
        if not lefts or not rights:
            continue
        frame_pairs: dict[tuple[int, int], tuple[DetectionRow, DetectionRow]] = {}
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
                    frame_pairs[(left.object_id, right.object_id)] = (left, right)
        if frame_pairs:
            valid_pairs[frame] = frame_pairs
    return valid_pairs


def find_pair_sequences(
    detections: Iterable[DetectionRow],
    query_types: Sequence[int],
    pair_constraint: PairConstraint,
    frame_width: int,
    frame_height: int,
    temporal_length: int,
) -> list[PairSequence]:
    valid_pairs = _valid_pairs_by_frame(
        detections,
        query_types=query_types,
        pair_constraint=pair_constraint,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    pair_timelines: dict[tuple[int, int], list[tuple[int, DetectionRow, DetectionRow]]] = {}
    for frame in sorted(valid_pairs):
        for pair_key, (left, right) in valid_pairs[frame].items():
            pair_timelines.setdefault(pair_key, []).append((frame, left, right))

    sequences: list[PairSequence] = []
    for (left_object_id, right_object_id), timeline in pair_timelines.items():
        run_start = 0
        while run_start < len(timeline):
            run_end = run_start + 1
            while run_end < len(timeline) and timeline[run_end][0] == timeline[run_end - 1][0] + 1:
                run_end += 1
            run = timeline[run_start:run_end]
            if len(run) >= temporal_length:
                for window_start in range(0, len(run) - temporal_length + 1):
                    window = run[window_start : window_start + temporal_length]
                    sequences.append(
                        PairSequence(
                            start_frame=window[0][0],
                            end_frame=window[-1][0],
                            frames=tuple(item[0] for item in window),
                            left_dets=tuple(item[1] for item in window),
                            right_dets=tuple(item[2] for item in window),
                            left_object_id=left_object_id,
                            right_object_id=right_object_id,
                        )
                    )
            run_start = run_end

    sequences.sort(key=lambda item: (item.start_frame, item.left_object_id, item.right_object_id))
    return sequences


def find_query_sequence_instance(
    detections: Iterable[DetectionRow],
    query_types: Sequence[int],
    pair_constraint: PairConstraint,
    frame_width: int,
    frame_height: int,
    temporal_length: int,
) -> QuerySequenceInstance:
    sequences = find_pair_sequences(
        detections,
        query_types=query_types,
        pair_constraint=pair_constraint,
        frame_width=frame_width,
        frame_height=frame_height,
        temporal_length=temporal_length,
    )
    if not sequences:
        raise ValueError(f"could not find a valid m=2 query sequence of length {temporal_length}")
    best = sequences[0]
    return QuerySequenceInstance(
        start_frame=best.start_frame,
        end_frame=best.end_frame,
        frames=best.frames,
        left_dets=best.left_dets,
        right_dets=best.right_dets,
        left_object_id=best.left_object_id,
        right_object_id=best.right_object_id,
    )


def max_pair_sequence_length(
    detections: Iterable[DetectionRow],
    query_types: Sequence[int],
    pair_constraint: PairConstraint,
    frame_width: int,
    frame_height: int,
) -> int:
    valid_pairs = _valid_pairs_by_frame(
        detections,
        query_types=query_types,
        pair_constraint=pair_constraint,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    pair_timelines: dict[tuple[int, int], list[int]] = {}
    for frame in sorted(valid_pairs):
        for pair_key in valid_pairs[frame]:
            pair_timelines.setdefault(pair_key, []).append(frame)

    max_run = 0
    for frames in pair_timelines.values():
        run_start = 0
        while run_start < len(frames):
            run_end = run_start + 1
            while run_end < len(frames) and frames[run_end] == frames[run_end - 1] + 1:
                run_end += 1
            max_run = max(max_run, run_end - run_start)
            run_start = run_end
    return max_run


def read_detection_rows(path: Path) -> list[DetectionRow]:
    rows: list[DetectionRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            rows.append(
                DetectionRow(
                    frame=int(row[0]),
                    object_id=int(row[1]),
                    left=float(row[2]),
                    top=float(row[3]),
                    width=float(row[4]),
                    height=float(row[5]),
                    cls=int(row[7]),
                )
            )
    return rows


def _read_query_spec_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_query_specs(path: Path) -> list[QuerySpec]:
    specs = []
    for row in _read_query_spec_rows(path):
        query_types = tuple(int(token.strip()) for token in row["types"].split(",") if token.strip())
        specs.append(
            QuerySpec(
                query_id=row["query_id"],
                dataset=row["dataset"],
                query_types=query_types,
                pair_constraint=parse_pair_constraint(row["pair_constraints"]),
                frame_threshold=int(row["frame_threshold"]),
                topk=int(row["topk"]),
                note=row.get("note", ""),
            )
        )
    return specs


def sequence_counts_from_root(root: Path) -> list[tuple[str, int]]:
    counts: list[tuple[str, int]] = []
    for seq_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        image_count = len(list(seq_dir.glob("*.jpg")))
        counts.append((seq_dir.name, image_count))
    return counts


def locate_frame(global_frame: int, spans: Sequence[FrameSpan], root: Path) -> FrameRef:
    for span in spans:
        if span.global_start <= global_frame <= span.global_end:
            local_frame = global_frame - span.global_start + 1
            image_name = f"img{local_frame:05d}.jpg"
            image_path = root / span.video_name / image_name
            return FrameRef(
                video_name=span.video_name,
                global_frame=global_frame,
                local_frame=local_frame,
                image_path=image_path,
            )
    raise ValueError(f"global frame {global_frame} is out of range")


def _import_cv2():
    import cv2

    return cv2


@lru_cache(maxsize=2048)
def load_image(path_str: str):
    cv2 = _import_cv2()
    image = cv2.imread(path_str, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path_str)
    return image


def crop_detection(image, det: DetectionRow):
    h, w = image.shape[:2]
    x1 = max(0, min(w - 1, int(det.left)))
    y1 = max(0, min(h - 1, int(det.top)))
    x2 = max(x1 + 1, min(w, int(math.ceil(det.left + det.width))))
    y2 = max(y1 + 1, min(h, int(math.ceil(det.top + det.height))))
    return image[y1:y2, x1:x2]


def crop_similarity(query_crop, candidate_crop) -> float:
    cv2 = _import_cv2()
    if query_crop.size == 0 or candidate_crop.size == 0:
        return -1.0

    query_resized = cv2.resize(query_crop, (64, 64), interpolation=cv2.INTER_LINEAR)
    candidate_resized = cv2.resize(candidate_crop, (64, 64), interpolation=cv2.INTER_LINEAR)

    query_hsv = cv2.cvtColor(query_resized, cv2.COLOR_BGR2HSV)
    candidate_hsv = cv2.cvtColor(candidate_resized, cv2.COLOR_BGR2HSV)

    hist_q = cv2.calcHist([query_hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    hist_c = cv2.calcHist([candidate_hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist_q, hist_q)
    cv2.normalize(hist_c, hist_c)
    hist_score = float(cv2.compareHist(hist_q, hist_c, cv2.HISTCMP_CORREL))

    gray_q = cv2.cvtColor(query_resized, cv2.COLOR_BGR2GRAY).astype("float32")
    gray_c = cv2.cvtColor(candidate_resized, cv2.COLOR_BGR2GRAY).astype("float32")
    gray_q -= gray_q.mean()
    gray_c -= gray_c.mean()
    denom = float((gray_q.std() * gray_c.std()) or 1.0)
    ncc = float((gray_q * gray_c).mean() / denom)

    return 0.5 * hist_score + 0.5 * ncc


def iter_valid_candidate_pairs(
    detections: Iterable[DetectionRow],
    query_types: Sequence[int],
    pair_constraint: PairConstraint,
    frame_width: int,
    frame_height: int,
) -> Iterator[tuple[int, DetectionRow, DetectionRow]]:
    by_frame: dict[int, list[DetectionRow]] = {}
    for det in detections:
        if det.cls in set(query_types):
            by_frame.setdefault(det.frame, []).append(det)

    left_type = query_types[pair_constraint.left_role]
    right_type = query_types[pair_constraint.right_role]

    for frame in sorted(by_frame):
        frame_dets = by_frame[frame]
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
                    yield frame, left, right


def _load_sequence_crops(
    frames: Sequence[int],
    dets: Sequence[DetectionRow],
    spans: Sequence[FrameSpan],
    detrac_root: Path,
) -> list:
    crops = []
    for frame, det in zip(frames, dets):
        frame_ref = locate_frame(frame, spans, detrac_root)
        image = load_image(str(frame_ref.image_path))
        crops.append(crop_detection(image, det))
    return crops


def _mean_similarity(query_crops: Sequence, candidate_crops: Sequence) -> float:
    if not query_crops or not candidate_crops or len(query_crops) != len(candidate_crops):
        return -1.0
    total = 0.0
    for query_crop, candidate_crop in zip(query_crops, candidate_crops):
        total += crop_similarity(query_crop, candidate_crop)
    return total / len(query_crops)


def benchmark_query(
    spec: QuerySpec,
    detections: Sequence[DetectionRow],
    spans: Sequence[FrameSpan],
    detrac_root: Path,
    frame_width: int,
    frame_height: int,
    temporal_length: int,
) -> tuple[QuerySequenceInstance, FrameRef, list[CandidateSequence], dict[str, float | int | str]]:
    query_instance = find_query_sequence_instance(
        detections,
        query_types=spec.query_types,
        pair_constraint=spec.pair_constraint,
        frame_width=frame_width,
        frame_height=frame_height,
        temporal_length=temporal_length,
    )
    query_frame_ref = locate_frame(query_instance.start_frame, spans, detrac_root)
    query_left_crops = _load_sequence_crops(query_instance.frames, query_instance.left_dets, spans, detrac_root)
    query_right_crops = _load_sequence_crops(query_instance.frames, query_instance.right_dets, spans, detrac_root)

    start = time.perf_counter()
    candidates: list[CandidateSequence] = []
    pair_sequences = find_pair_sequences(
        detections,
        query_types=spec.query_types,
        pair_constraint=spec.pair_constraint,
        frame_width=frame_width,
        frame_height=frame_height,
        temporal_length=temporal_length,
    )
    scanned_frame_count = 0
    seen_frames: set[int] = set()

    for sequence in pair_sequences:
        if (
            sequence.left_object_id == query_instance.left_object_id
            and sequence.right_object_id == query_instance.right_object_id
            and not (
                sequence.end_frame < query_instance.start_frame
                or sequence.start_frame > query_instance.end_frame
            )
        ):
            continue
        for frame in sequence.frames:
            if frame not in seen_frames:
                seen_frames.add(frame)
                scanned_frame_count += 1
        left_crops = _load_sequence_crops(sequence.frames, sequence.left_dets, spans, detrac_root)
        right_crops = _load_sequence_crops(sequence.frames, sequence.right_dets, spans, detrac_root)
        left_score = _mean_similarity(query_left_crops, left_crops)
        right_score = _mean_similarity(query_right_crops, right_crops)
        total_score = left_score + right_score
        candidates.append(
            CandidateSequence(
                start_frame=sequence.start_frame,
                end_frame=sequence.end_frame,
                frames=sequence.frames,
                left_dets=sequence.left_dets,
                right_dets=sequence.right_dets,
                left_score=left_score,
                right_score=right_score,
                total_score=total_score,
            )
        )

    elapsed = time.perf_counter() - start
    candidates.sort(key=lambda item: item.total_score, reverse=True)
    top_candidates = candidates[: spec.topk]
    summary = {
        "query_id": spec.query_id,
        "dataset": spec.dataset,
        "temporal_length": temporal_length,
        "query_frame": query_instance.start_frame,
        "query_end_frame": query_instance.end_frame,
        "query_video": query_frame_ref.video_name,
        "query_local_frame": query_frame_ref.local_frame,
        "query_types": ",".join(map(str, spec.query_types)),
        "pair_constraint": f"{spec.pair_constraint.left_role}-{spec.pair_constraint.right_role}:{spec.pair_constraint.theta_lt},{spec.pair_constraint.d_ratio_lt}",
        "candidate_frames_scanned": scanned_frame_count,
        "valid_pair_count": max(0, len(pair_sequences) - 1),
        "topk_returned": len(top_candidates),
        "elapsed_s": elapsed,
        "top1_frame": "" if not top_candidates else top_candidates[0].start_frame,
        "top1_end_frame": "" if not top_candidates else top_candidates[0].end_frame,
        "top1_score": "" if not top_candidates else top_candidates[0].total_score,
        "note": spec.note,
    }
    return query_instance, query_frame_ref, top_candidates, summary


def write_query_crops(
    out_dir: Path,
    spec: QuerySpec,
    spans: Sequence[FrameSpan],
    detrac_root: Path,
    query_instance: QuerySequenceInstance,
) -> None:
    cv2 = _import_cv2()
    out_dir.mkdir(parents=True, exist_ok=True)
    for offset, (frame, left_det, right_det) in enumerate(
        zip(query_instance.frames, query_instance.left_dets, query_instance.right_dets),
        start=1,
    ):
        frame_ref = locate_frame(frame, spans, detrac_root)
        image = load_image(str(frame_ref.image_path))
        for role_idx, det in enumerate((left_det, right_det)):
            crop = crop_detection(image, det)
            out_path = (
                out_dir
                / f"{spec.query_id}_t{offset:02d}_role{role_idx}_cls{det.cls}_{frame_ref.video_name}_{frame_ref.local_frame:05d}.jpg"
            )
            cv2.imwrite(str(out_path), crop)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def append_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    ensure_parent_dir(path)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def candidate_rows(
    spec: QuerySpec,
    spans: Sequence[FrameSpan],
    detrac_root: Path,
    candidates: Sequence[CandidateSequence],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rank, candidate in enumerate(candidates, start=1):
        frame_ref = locate_frame(candidate.start_frame, spans, detrac_root)
        end_frame_ref = locate_frame(candidate.end_frame, spans, detrac_root)
        rows.append(
            {
                "query_id": spec.query_id,
                "dataset": spec.dataset,
                "rank": rank,
                "start_frame": candidate.start_frame,
                "end_frame": candidate.end_frame,
                "video_name": frame_ref.video_name,
                "local_start_frame": frame_ref.local_frame,
                "local_end_frame": end_frame_ref.local_frame,
                "left_object_id": candidate.left_dets[0].object_id,
                "left_class": candidate.left_dets[0].cls,
                "right_object_id": candidate.right_dets[0].object_id,
                "right_class": candidate.right_dets[0].cls,
                "left_score": candidate.left_score,
                "right_score": candidate.right_score,
                "total_score": candidate.total_score,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser("Run a RELOCATE-style m=2 benchmark on drtest/drtrain")
    parser.add_argument("--query_specs", default=str(DEFAULT_QUERY_SPECS))
    parser.add_argument("--query_ids", default="drtest_q1,drtrain_q1")
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--temporal_length", type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = out_dir / "relocate_m2_raw.csv"
    summary_csv = out_dir / "relocate_m2_summary.csv"
    crop_dir = out_dir / "query_crops"
    if raw_csv.exists():
        raw_csv.unlink()
    if summary_csv.exists():
        summary_csv.unlink()
    if crop_dir.exists():
        for item in crop_dir.iterdir():
            if item.is_file():
                item.unlink()

    query_id_filter = {token.strip() for token in args.query_ids.split(",") if token.strip()}
    raw_spec_rows = _read_query_spec_rows(Path(args.query_specs))
    filtered_rows = [row for row in raw_spec_rows if row["query_id"] in query_id_filter]
    specs = []
    for row in filtered_rows:
        query_types = tuple(int(token.strip()) for token in row["types"].split(",") if token.strip())
        specs.append(
            QuerySpec(
                query_id=row["query_id"],
                dataset=row["dataset"],
                query_types=query_types,
                pair_constraint=parse_pair_constraint(row["pair_constraints"]),
                frame_threshold=int(row["frame_threshold"]),
                topk=int(row["topk"]),
                note=row.get("note", ""),
            )
        )

    for idx, spec in enumerate(specs, start=1):
        detrac_root = DEFAULT_DETRAC_ROOTS[spec.dataset]
        frame_width, frame_height = DEFAULT_FRAME_SIZE[spec.dataset]
        sequence_counts = sequence_counts_from_root(detrac_root)
        spans = build_frame_spans(sequence_counts)
        detections = read_detection_rows(REPO_ROOT / "storage" / f"{spec.dataset}.txt")

        print(
            f"[{idx}/{len(specs)}] query_id={spec.query_id} dataset={spec.dataset} "
            f"types={spec.query_types} pair={spec.pair_constraint}",
            flush=True,
        )

        try:
            query_instance, query_frame_ref, top_candidates, summary = benchmark_query(
                spec=spec,
                detections=detections,
                spans=spans,
                detrac_root=detrac_root,
                frame_width=frame_width,
                frame_height=frame_height,
                temporal_length=args.temporal_length,
            )
            write_query_crops(crop_dir, spec, spans, detrac_root, query_instance)
            append_rows(raw_csv, candidate_rows(spec, spans, detrac_root, top_candidates))
            append_rows(summary_csv, [summary])

            print(
                f"  query_frames={summary['query_frame']}-{summary['query_end_frame']} "
                f"query_video={summary['query_video']} "
                f"valid_sequences={summary['valid_pair_count']} "
                f"elapsed={float(summary['elapsed_s']):.4f}s",
                flush=True,
            )
        except ValueError as exc:
            max_run = max_pair_sequence_length(
                detections,
                query_types=spec.query_types,
                pair_constraint=spec.pair_constraint,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            failure_summary = {
                "query_id": spec.query_id,
                "dataset": spec.dataset,
                "temporal_length": args.temporal_length,
                "query_frame": "",
                "query_end_frame": "",
                "query_video": "",
                "query_local_frame": "",
                "query_types": ",".join(map(str, spec.query_types)),
                "pair_constraint": f"{spec.pair_constraint.left_role}-{spec.pair_constraint.right_role}:{spec.pair_constraint.theta_lt},{spec.pair_constraint.d_ratio_lt}",
                "candidate_frames_scanned": 0,
                "valid_pair_count": 0,
                "topk_returned": 0,
                "elapsed_s": "",
                "top1_frame": "",
                "top1_end_frame": "",
                "top1_score": "",
                "note": f"{spec.note} | skipped: {exc} | max_available_temporal_length={max_run}",
            }
            append_rows(summary_csv, [failure_summary])
            print(
                f"  skipped: {exc}; max_available_temporal_length={max_run}",
                flush=True,
            )


if __name__ == "__main__":
    main()
