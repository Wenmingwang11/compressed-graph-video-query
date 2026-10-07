"""Time-R1 native temporal localization bridge, separate from binding-aware QST.

Run in a CUDA environment with the official Time-R1 dependencies installed.
Each input media file must be a fixed, answer-independent sampled-video clip.
No track IDs or reference answer intervals are supplied to the model.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace

from supplement.corrected_query_model import QuerySpec
from supplement.generate_corrected_baseline_workload import query_from_dict


DEFAULT_SOURCE = Path('D:/pycharm/_baseline_prepare_20260919/time-r1-main')


def query_text(query: QuerySpec, class_names: dict[int, str], width: int, height: int,
               sampled_fps: float) -> str:
    if width <= 0 or height <= 0 or not math.isfinite(sampled_fps) or sampled_fps <= 0:
        raise ValueError('Positive dimensions and finite sampled-video FPS are required')
    if any(class_id not in class_names for class_id in query.role_types):
        raise ValueError('Provide an explicit class name for every query role')
    roles = ', '.join(f'R{i} is a {class_names[class_id]}' for i, class_id in enumerate(query.role_types))
    conditions = []
    for (source, target), constraint in sorted(query.relations.items()):
        low = constraint.d_min * (width + height) / query.distance_parts
        high = (constraint.d_max + 1) * (width + height) / query.distance_parts
        angle_ranges = ' or '.join(f'[{angle * 180 / query.theta_parts:g}, '
                                  f'{(angle + 1) * 180 / query.theta_parts:g}) degrees'
                                  for angle in sorted(constraint.theta_bins))
        conditions.append(f'from the center of R{source} to the center of R{target}, '
                          f'the distance is in [{low:g}, {high:g}) pixels and the angle is '
                          f'in {angle_ranges}')
    geometry = '; '.join(conditions) if conditions else 'no additional spatial condition'
    return (f'Find a time interval with {roles}. All roles must be distinct objects and each '
            f'role must keep the same individual throughout the interval. Simultaneously, {geometry}. '
            f'Coordinates refer to a {width} by {height} image with x to the right and y downward; '
            f'angles are atan2(dy, dx), measured from the positive x axis. The conditions must '
            f'hold for at least {query.min_consecutive_frames} consecutive sampled frames '
            f'({query.min_consecutive_frames / sampled_fps:g} seconds at {sampled_fps:g} fps).')


def parse_prediction(text: str, *, first_frame: int, frame_count: int, sampled_fps: float) -> dict:
    if frame_count <= 0 or not math.isfinite(sampled_fps) or sampled_fps <= 0:
        raise ValueError('Positive frame count and finite FPS required')
    answers = re.findall(r'<answer>\s*(.*?)\s*</answer>', text, flags=re.DOTALL | re.IGNORECASE)
    match = re.fullmatch(r'\s*(\d+(?:\.\d+)?)\s+(?:to|and)\s+(\d+(?:\.\d+)?)\s*',
                         answers[-1] if answers else '')
    if match is None:
        raise ValueError('No valid timestamp pair inside <answer> tags')
    start_s, end_s = map(float, match.groups())
    duration = frame_count / sampled_fps
    if not (0 <= start_s < end_s <= duration + 1e-8):
        raise ValueError('Prediction must lie within the actual clip duration')
    # Report only frames whose full sampled-frame time cells lie inside [s,e).
    start_frame = first_frame + math.ceil(start_s * sampled_fps - 1e-8)
    end_frame = first_frame + math.floor(end_s * sampled_fps + 1e-8) - 1
    if end_frame < start_frame:
        raise ValueError('Prediction contains no complete sampled frame')
    return dict(start_seconds=start_s, end_seconds=end_s,
                start_frame=start_frame, end_frame=end_frame,
                output_unit='temporal_interval_without_object_binding')


def build_requests(query, clips, class_names, width, height):
    requests = []
    for clip in clips:
        # Explicit fields prevent accidental propagation of reference answers.
        first, count, fps = int(clip['first_frame']), int(clip['frame_count']), float(clip['sampled_fps'])
        if count <= 0 or not math.isfinite(fps) or fps <= 0:
            raise ValueError('Invalid sampled clip metadata')
        video = str(Path(clip['video']).resolve())
        requests.append(dict(video=video, first_frame=first, frame_count=count, sampled_fps=fps,
                             duration=count / fps, sentence=query_text(query, class_names, width, height, fps)))
    if not requests:
        raise ValueError('Clip manifest must be nonempty')
    return requests


def infer(requests, *, source_root: Path, model_path: str, max_new_tokens=256, total_pixels=3584*28*28):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Native Time-R1 requires a configured CUDA inference environment')
    from transformers import AutoProcessor
    for request in requests:
        if not Path(request['video']).is_file():
            raise FileNotFoundError(request['video'])
    sys.path.insert(0, str(source_root.resolve()))
    spec = importlib.util.spec_from_file_location('_official_time_r1_demo', source_root / 'demo.py')
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    args = SimpleNamespace(model_base=model_path, pipeline_parallel_size=1,
                           total_pixels=total_pixels, max_new_tokens=max_new_tokens)
    setup_start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
    processor.tokenizer.padding_side = 'left'
    model = native.vllmWrapper(args)
    torch.cuda.synchronize()
    setup_s = time.perf_counter() - setup_start
    outputs = []
    for request in requests:
        torch.cuda.synchronize()
        started = time.perf_counter()
        # Native build_dataset requires a metadata field named timestamp; it is
        # not an input to the model. Nulls explicitly mean no reference answer.
        item = dict(video=request['video'], duration=request['duration'],
                    timestamp=[None, None], sentence=request['sentence'])
        batch = native.build_dataset(item, processor, num_workers=1, total_pixels=total_pixels)
        torch.cuda.synchronize()
        prepared = time.perf_counter()
        texts = model.generate(batch['inputs'], max_new_tokens=max_new_tokens, seed=0)
        torch.cuda.synchronize()
        generated = time.perf_counter()
        try:
            prediction = parse_prediction(texts[0], first_frame=request['first_frame'],
                                          frame_count=request['frame_count'], sampled_fps=request['sampled_fps'])
            status = 'predicted'
        except ValueError as error:
            prediction, status = None, f'invalid_prediction: {error}'
        finished = time.perf_counter()
        outputs.append(dict(video=request['video'], status=status, prediction=prediction,
                            decode_preprocess_s=prepared-started, inference_s=generated-prepared,
                            online_end_to_end_s=finished-started, output_text=texts[0]))
    return dict(status='inference_completed', model_setup_s=setup_s, outputs=outputs,
                evaluation_unit='one temporal localization per fixed clip; no binding-aware Top-k claim')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workload', type=Path, required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--query-id', required=True)
    parser.add_argument('--clips-json', type=Path, required=True)
    parser.add_argument('--class-names-json', type=Path, required=True)
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--height', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--model-path')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    records = json.loads(args.workload.read_text(encoding='utf-8'))
    selected = [row for row in records if row['dataset'] == args.dataset and row['query_id'] == args.query_id]
    if len(selected) != 1:
        raise ValueError('Expected one workload query')
    query = query_from_dict(selected[0])
    names = {int(key): value for key, value in json.loads(args.class_names_json.read_text(encoding='utf-8')).items()}
    requests = build_requests(query, json.loads(args.clips_json.read_text(encoding='utf-8')), names, args.width, args.height)
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.prepare_only:
        result = dict(status='prepared_not_measured', requests=requests)
    else:
        if not args.model_path:
            parser.error('--model-path is required for inference')
        result = infer(requests, source_root=args.source_root, model_path=args.model_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
