"""Measured QST adapter comparison; no raw-video E2E or native-model claims."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import statistics
import sys
import time

import numpy as np
import pandas as pd

from supplement.corrected_baseline_strategies import build_strategy
from supplement.corrected_query_model import exact_query, ordered_result_signature
from supplement.craw_element_adapter import CrawElementStrategy, DEFAULT_CRAW_ROOT, NATIVE_FILES
from supplement.dataset_config import dataset_frame_size, dataset_storage_path
from supplement.generate_corrected_baseline_workload import load_tracking_table, query_from_dict
from supplement.run_corrected_baseline_benchmark import _OursStrategy
from paper_query_corrected import corrected_mli_tag
from vsimsearch.mul_build_index import artifact_paths


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKLOAD = ROOT / 'supplement/out/corrected_baseline_20260712/workload.json'
METHODS = ('Exact Frame Scan', 'VQPy-style', 'VOCAL-UDF-style', 'Video-ColBERT-style',
           'LAVA-style', 'STAR-style', 'Craw-Element+Verifier', 'Ours')
PROVENANCE = {
    'Exact Frame Scan': 'Complete enumeration over supplied detections; no candidate index.',
    'VQPy-style': 'Custom procedural category filter; does not execute native VQPy runtime.',
    'VOCAL-UDF-style': 'Custom category posting intersection; no native generated UDF/SQL executor.',
    'Video-ColBERT-style': 'Custom category token segment index; no learned embeddings or MaxSim.',
    'LAVA-style': 'Custom hierarchical category summary; no native LAVA language/model pipeline.',
    'STAR-style': 'Custom quantized directed edge index; not native STAR query engine; '
                  'offline postings restricted to workload class pairs (declared workload-aware).',
    'Craw-Element+Verifier': 'Official Craw MSU splitter and F-class inverted index; '
                           'lookup acceleration; all observed classes retained; no T/I/F similarity.',
    'Ours': 'Existing corrected MLI candidate generator plus shared raw-detection verifier; '
            'includes window-edge fallback; existing indexes loaded, not rebuilt.',
}
SOURCE_FILES = ('supplement/corrected_query_model.py', 'supplement/corrected_baseline_strategies.py',
                'supplement/craw_element_adapter.py', 'supplement/run_baseline_audit.py',
                'supplement/run_corrected_baseline_benchmark.py',
                'vsimsearch/corrected_mli_querying.py')


class ExactFrameScan:
    def __init__(self, data, width, height):
        self.data, self.width, self.height = data, width, height

    def query(self, query, *, stats=None, return_all=False):
        if stats is not None:
            stats['candidate_s'] = 0.0
        return exact_query(self.data, query, self.width, self.height, stats=stats, return_all=return_all)

    def release(self):
        self.data = None


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(output, *, dataset='drtest', query_id='Q1', workload_path=DEFAULT_WORKLOAD,
        methods=METHODS, repetitions=3, warmups=1):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if repetitions < 1 or warmups < 0 or not methods or len(set(methods)) != len(methods):
        raise ValueError('Need unique nonempty methods, repetitions >= 1 and warmups >= 0')
    if set(methods).difference(METHODS):
        raise ValueError('Unknown method')
    entries = json.loads(Path(workload_path).read_text(encoding='utf-8'))
    matches = [entry for entry in entries if entry['dataset'] == dataset and entry['query_id'] == query_id]
    if len(matches) != 1:
        raise ValueError('Expected exactly one workload query')
    # Keep known answer IDs/times out of every strategy and output query specification.
    query_record = {key: value for key, value in matches[0].items() if key != 'source_event'}
    query = query_from_dict(query_record)
    width, height = dataset_frame_size(dataset)
    input_path = dataset_storage_path(dataset)
    input_hash = sha256(input_path)
    load_start = time.perf_counter()
    data = load_tracking_table(input_path)
    load_s = time.perf_counter() - load_start
    source_hashes = {relative: sha256(ROOT / relative) for relative in SOURCE_FILES}
    native_hashes = ({name: sha256(DEFAULT_CRAW_ROOT / name) for name in NATIVE_FILES}
                     if 'Craw-Element+Verifier' in methods else {})
    ours_artifacts = []
    if 'Ours' in methods:
        for path in artifact_paths(dataset, .7, output_tag=corrected_mli_tag())[:3]:
            ours_artifacts.append(dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=sha256(path)))
    manifest = dict(complete=False, all_correct=False, started_utc=datetime.now(timezone.utc).isoformat(),
                    dataset=dataset, query=query_record, frame_dimensions=[width, height],
                    sampled_frame_assumption=True, data_rows=len(data),
                    unique_observed_frames=int(data['frame'].nunique()),
                    frame_min=int(data['frame'].min()), frame_max=int(data['frame'].max()),
                    data_path=str(input_path), data_sha256=input_hash, data_load_s=load_s,
                    workload_path=str(workload_path), workload_sha256=sha256(workload_path),
                    warmups=warmups, repetitions=repetitions, methods=list(methods),
                    timing_scope='preloaded online candidate generation + exact verification + temporal '
                                 'aggregation + Top-k; counters disabled; input load/setup excluded',
                    audit_scope='one separate instrumented query, all qualifying intervals before Top-k',
                    method_provenance={name: PROVENANCE[name] for name in methods},
                    skipped={'Time-R1': 'No CUDA GPU or model weights on this host; no inference measured',
                             'SketchQL': 'Native trajectory-sketch query requires a separate task evaluation'},
                    source_sha256=source_hashes, python=sys.version, platform=platform.platform(),
                    craw_source_sha256=native_hashes, ours_artifacts=ours_artifacts,
                    ours_index_parameters=dict(similarity=.7, tag=corrected_mli_tag()),
                    processor=platform.processor(), logical_cpus=os.cpu_count(),
                    numpy=np.__version__, pandas=pd.__version__)
    write_json(output / 'manifest.json', manifest)
    print(f'Loaded {dataset}: {len(data):,} detections, {manifest["unique_observed_frames"]:,} frames. Oracle...', flush=True)
    oracle_stats = {}
    oracle = exact_query(data, query, width, height, stats=oracle_stats, return_all=True)
    oracle_key = {(r.binding, r.start_frame, r.end_frame) for r in oracle}
    expected_topk = ordered_result_signature(oracle[:query.topk])
    expected_all = ordered_result_signature(oracle)
    write_json(output / 'oracle.json', {'all_results': [asdict(r) for r in oracle],
                                      'stats': oracle_stats, 'ordered_topk_sha256': expected_topk,
                                      'ordered_all_sha256': expected_all})
    rows, raw, predictions, audits = [], [], {}, {}
    for method in methods:
        print(f'{method}: setup...', flush=True)
        setup_kind, payload_bytes = 'artifact_build', 0
        setup_start = time.perf_counter()
        if method == 'Exact Frame Scan':
            strategy = ExactFrameScan(data, width, height)
            setup_kind = 'no_index'
        elif method == 'Ours':
            paths = artifact_paths(dataset, .7, output_tag=corrected_mli_tag())[:3]
            if not all(path.is_file() for path in paths):
                raise FileNotFoundError('Required prebuilt Ours index missing; refusing implicit rebuild')
            strategy = _OursStrategy(dataset, data, width, height)
            setup_kind = 'prebuilt_index_load_NOT_build'
            payload_bytes = sum(path.stat().st_size for path in paths)
        elif method == 'Craw-Element+Verifier':
            strategy = CrawElementStrategy(data, width, height, artifact_dir=output / 'craw_artifact')
        else:
            pairs = {(query.role_types[s], query.role_types[t]) for s, t in query.relations}
            strategy = build_strategy(method, data, width, height,
                                      allowed_type_pairs=pairs if method == 'STAR-style' else None)
        setup_s = time.perf_counter() - setup_start
        print(f'{method}: setup {setup_s:.3f}s; warming and timing...', flush=True)
        correct = True
        for _ in range(warmups):
            correct &= ordered_result_signature(strategy.query(query)) == expected_topk
        durations = []
        for repeat in range(1, repetitions + 1):
            started = time.perf_counter()
            results = strategy.query(query)
            elapsed = time.perf_counter() - started
            verified = ordered_result_signature(results) == expected_topk
            correct &= verified
            durations.append(elapsed)
            raw.append(dict(method=method, repetition=repeat, elapsed_s=elapsed,
                            ordered_topk_verified=verified))
            write_csv(output / 'raw_measurements.csv', raw)
            print(f'{method}: repeat {repeat}/{repetitions} {elapsed:.6f}s, Top-k verified={verified}', flush=True)
        audit = {}
        full = strategy.query(query, stats=audit, return_all=True)
        audits[method] = audit
        full_correct = ordered_result_signature(full) == expected_all
        actual_key = {(r.binding, r.start_frame, r.end_frame) for r in full}
        true_positives = len(actual_key & oracle_key)
        predictions[method] = [asdict(r) for r in full]
        # Comparable serialized query artifacts, not resident RAM or compressed file size.
        if method not in ('Ours', 'Exact Frame Scan'):
            payload_path = output / (method.replace('+', '_') + '.artifact.pkl')
            payload = strategy.artifact
            if method == 'Craw-Element+Verifier':
                payload = {'postings': dict(payload.index.index), 'msu_frames': payload.msu_frames,
                           'ranges': payload.ranges}
            with payload_path.open('wb') as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            payload_bytes = payload_path.stat().st_size
            del payload
        row = dict(method=method, median_s=statistics.median(durations),
                   min_s=min(durations), max_s=max(durations),
                   setup_s=setup_s, setup_kind=setup_kind, artifact_payload_bytes=payload_bytes,
                   ordered_topk_verified=bool(correct), full_intervals_verified=full_correct,
                   qualifying_interval_recall=true_positives / len(oracle_key) if oracle_key else None,
                   qualifying_interval_precision=true_positives / len(actual_key) if actual_key else None,
                   candidate_frames=audit['candidate_frames'],
                   frame_pruning=1-audit['candidate_frames']/manifest['unique_observed_frames'],
                   binding_frame_enumerated=audit['binding_frame_enumerated'],
                   binding_frame_admitted=audit['binding_frame_admitted'],
                   binding_frame_pruning=(1-audit['binding_frame_admitted']/oracle_stats['binding_frame_enumerated']
                                          if oracle_stats['binding_frame_enumerated'] else None),
                   spatial_predicate_evaluations=audit['spatial_predicate_evaluations'],
                   predicate_reduction=(1-audit['spatial_predicate_evaluations']/oracle_stats['spatial_predicate_evaluations']
                                        if oracle_stats['spatial_predicate_evaluations'] else None),
                   spatial_match_binding_frames=audit['spatial_match_binding_frames'],
                   qualifying_intervals=audit['qualifying_intervals'],
                   audit_candidate_s=audit['candidate_s'], audit_verification_s=audit['verification_s'],
                   audit_temporal_merge_s=audit['temporal_merge_s'], audit_topk_s=audit['topk_s'])
        rows.append(row)
        write_csv(output / 'summary.csv', rows)
        write_json(output / 'results.json', rows)
        write_json(output / 'predictions.json', predictions)
        write_json(output / 'audit_details.json', audits)
        strategy.release()
        del strategy
        gc.collect()
        print(f'{method}: full intervals verified={full_correct}; candidate frames={audit["candidate_frames"]}', flush=True)
    manifest['data_unchanged'] = sha256(input_path) == input_hash
    manifest['source_unchanged'] = all(sha256(ROOT / relative) == value for relative, value in source_hashes.items())
    manifest['native_source_unchanged'] = all(sha256(DEFAULT_CRAW_ROOT / name) == value
                                             for name, value in native_hashes.items())
    manifest['ours_artifacts_unchanged'] = all(sha256(item['path']) == item['sha256'] for item in ours_artifacts)
    manifest['complete'] = True
    manifest['all_correct'] = all(row['full_intervals_verified'] and row['ordered_topk_verified'] for row in rows)
    manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(output / 'manifest.json', manifest)
    if not all(manifest[key] for key in ('all_correct', 'data_unchanged', 'source_unchanged',
                                       'native_source_unchanged', 'ours_artifacts_unchanged')):
        raise RuntimeError('Measured run completed with correctness/input/source integrity failure; inspect outputs')
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dataset', default='drtest')
    parser.add_argument('--query-id', default='Q1')
    parser.add_argument('--workload', type=Path, default=DEFAULT_WORKLOAD)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--warmups', type=int, default=1)
    args = parser.parse_args()
    run(args.output_dir, dataset=args.dataset, query_id=args.query_id, workload_path=args.workload,
        repetitions=args.repetitions, warmups=args.warmups)
