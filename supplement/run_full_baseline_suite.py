"""Resumable full-dataset QST measurements. Native QBE is a separate protocol."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import pickle
import platform
import statistics
import subprocess
import sys
import time
import traceback
import uuid

import numpy as np
import pandas as pd

from supplement.corrected_baseline_strategies import build_strategy
from supplement.corrected_query_model import QueryResult, exact_query, ordered_result_signature
from supplement.craw_element_adapter import CrawElementStrategy, DEFAULT_CRAW_ROOT, NATIVE_FILES
from supplement.dataset_config import PREFERRED_VIDEO_ORDER, dataset_frame_size, dataset_storage_path
from supplement.generate_corrected_baseline_workload import load_tracking_table, query_from_dict
from supplement.run_baseline_audit import ExactFrameScan, PROVENANCE, SOURCE_FILES, sha256
from supplement.run_corrected_baseline_benchmark import _OursStrategy
from paper_query_corrected import corrected_mli_tag
from vsimsearch.mul_build_index import artifact_paths

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKLOAD = ROOT / 'supplement/out/corrected_baseline_20260712/workload.json'
ROSTER = ('Ours', 'STAR-style', 'VQPy-style', 'VOCAL-UDF-style', 'SketchQL',
          'Video-ColBERT-style', 'LAVA-style', 'Exact Frame Scan', 'Craw-Element+Verifier', 'Time-R1')
STRICT_METHODS = ('Exact Frame Scan', 'VQPy-style', 'VOCAL-UDF-style', 'Video-ColBERT-style',
                  'LAVA-style', 'STAR-style', 'Craw-Element+Verifier', 'Ours')
PRUNING_METHODS = tuple(name for name in STRICT_METHODS if name != 'Exact Frame Scan')
SUITE_SOURCES = tuple(dict.fromkeys((*SOURCE_FILES, 'supplement/run_full_baseline_suite.py',
                                   'supplement/generate_corrected_baseline_workload.py',
                                   'supplement/dataset_config.py', 'paper_query_corrected.py')))


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def interval_metrics(predicted, expected):
    key = lambda r: (r.binding, r.start_frame, r.end_frame)
    actual, truth = {key(r) for r in predicted}, {key(r) for r in expected}
    tp, fp, fn = len(actual & truth), len(actual - truth), len(truth - actual)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1)


def check_identity(previous, current):
    for key in ('data_sha256', 'source_sha256', 'workload_sha256', 'repetitions', 'warmups',
                'methods', 'index_sha256', 'native_sha256'):
        if previous.get(key) != current.get(key):
            raise ValueError(f'Resume identity mismatch: {key}; use a fresh output directory')


def collect_identity(dataset, workload_path, repetitions, warmups):
    return dict(data_sha256=sha256(dataset_storage_path(dataset)), workload_sha256=sha256(workload_path),
        source_sha256={name: sha256(ROOT / name) for name in SUITE_SOURCES},
        native_sha256={name: sha256(DEFAULT_CRAW_ROOT / name) for name in NATIVE_FILES},
        index_sha256={str(p): sha256(p) for p in artifact_paths(dataset, .7, output_tag=corrected_mli_tag())[:3]},
        repetitions=repetitions, warmups=warmups, methods=list(STRICT_METHODS))


def checkpoint_validity(directory, identity):
    path = Path(directory) / 'manifest.json'
    if not path.exists():
        return 'missing_manifest'
    previous = json.loads(path.read_text(encoding='utf-8'))
    try:
        check_identity(previous, identity)
    except ValueError:
        return 'stale_identity'
    if not all(previous.get('integrity', {}).get(k) is True for k in ('data','source','native','index')):
        return 'integrity_unverified_or_failed'
    return 'valid'


def method_complete(directory, method, group_paths):
    return all(Path(p).exists() for p in group_paths) and (Path(directory) / 'artifacts' / f'{method}.metadata.json').exists()


def measure_group_safely(*args, **kwargs):
    try:
        return measure_group(*args, **kwargs)
    except Exception as error:
        return dict(status='failed', error=repr(error), traceback=traceback.format_exc(), at_utc=now())


def measure_group(strategy, query, oracle, *, repetitions, warmups, progress=None):
    expected_topk = ordered_result_signature(oracle[:query.topk])
    warmup_verified = True
    for _ in range(warmups):
        warmup_verified &= ordered_result_signature(strategy.query(query)) == expected_topk
    raw = []
    for repetition in range(1, repetitions + 1):
        started = time.perf_counter()
        results = strategy.query(query)
        elapsed = time.perf_counter() - started
        signature = ordered_result_signature(results)
        raw.append(dict(repetition=repetition, elapsed_s=elapsed, signature=signature,
                        ordered_topk_verified=signature == expected_topk))
        if progress:
            progress(raw)
    audit = {}
    full = strategy.query(query, stats=audit, return_all=True)
    times = [row['elapsed_s'] for row in raw]
    return dict(status='measured', raw=raw, median_s=statistics.median(times),
                q1_s=float(np.quantile(times, .25)), q3_s=float(np.quantile(times, .75)),
                min_s=min(times), max_s=max(times), audit=audit,
                full_results=[asdict(r) for r in full], metrics=interval_metrics(full, oracle),
                ordered_topk_verified=warmup_verified and all(r['ordered_topk_verified'] for r in raw),
                full_intervals_verified=ordered_result_signature(full) == ordered_result_signature(oracle),
                oracle_all_signature=ordered_result_signature(oracle),
                oracle_topk_signature=expected_topk)


def csv_write(path, rows):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ['no_verified_results'])
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def serialize_artifact(strategy, path, method):
    payload = strategy.artifact
    if method == 'Craw-Element+Verifier':
        payload = dict(postings=dict(payload.index.index), msu_frames=payload.msu_frames, ranges=payload.ranges)
    with Path(path).open('wb') as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return Path(path).stat().st_size


def dataset_worker(output, dataset, workload_path, repetitions, warmups):
    destination = Path(output) / dataset
    destination.mkdir(parents=True, exist_ok=True)
    records = [r for r in json.loads(Path(workload_path).read_text(encoding='utf-8')) if r['dataset'] == dataset]
    queries = [query_from_dict({k: v for k, v in row.items() if k != 'source_event'}) for row in records]
    if sorted(q.query_id for q in queries) != ['Q1', 'Q2']:
        raise ValueError('Need exactly Q1 and Q2 per dataset')
    data_path = dataset_storage_path(dataset)
    index_paths = artifact_paths(dataset, .7, output_tag=corrected_mli_tag())[:3]
    identity = collect_identity(dataset, workload_path, repetitions, warmups)
    manifest_path = destination / 'manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        check_identity(previous, identity)
        if any(value is False for value in previous.get('integrity', {}).values()):
            raise ValueError('Previously failed integrity check; use a fresh output directory')
        if (previous.get('complete') and checkpoint_validity(destination, identity) == 'valid'
            and all(method_complete(destination, m, [destination/'groups'/f'{m}_{q.query_id}.json' for q in queries]) for m in STRICT_METHODS)):
            print(f'{dataset}: already complete with matching inputs/code', flush=True)
            return
    elif any(destination.glob('oracle_*.json')) or any((destination/'groups').glob('*.json')):
        raise ValueError('Orphan checkpoints without governing manifest; use a fresh output directory')
    width, height = dataset_frame_size(dataset)
    manifest = dict(identity, dataset=dataset, started_utc=now(), complete=False,
                    data_path=str(data_path), frame_dimensions=[width, height],
                    queries=[{k: v for k, v in row.items() if k != 'source_event'} for row in records],
                    python=sys.version, numpy=np.__version__, pandas=pd.__version__, platform=platform.platform(),
                    protocol='strict_QST_continuous_fixed_binding', quantization='W+H; theta10,d8',
                    empty_metric_convention='empty predicted precision=1; empty truth recall=1; both empty F1=1',
                    timing_scope='preloaded online query; one separate full-result instrumented audit',
                    provenance=PROVENANCE, index_parameters={'similarity': .7, 'tag': corrected_mli_tag()})
    write_json(manifest_path, manifest)
    started = time.perf_counter()
    data = load_tracking_table(data_path)
    manifest.update(data_load_s=time.perf_counter()-started, rows=len(data),
                    unique_frames=int(data.frame.nunique()), first_frame=int(data.frame.min()),
                    last_frame=int(data.frame.max()))
    write_json(manifest_path, manifest)
    print(f'{dataset}: {len(data):,} rows, {manifest["unique_frames"]:,} observed sampled frames', flush=True)
    oracles, oracle_stats = {}, {}
    for query in queries:
        oracle_path = destination / f'oracle_{query.query_id}.json'
        if oracle_path.exists():
            saved = json.loads(oracle_path.read_text(encoding='utf-8'))
            full = [QueryResult(tuple(r['binding']), r['start_frame'], r['end_frame']) for r in saved['results']]
            stats = saved['audit']
        else:
            print(f'{dataset} {query.query_id}: full reference scan', flush=True)
            stats = {}
            full = exact_query(data, query, width, height, stats=stats, return_all=True)
            write_json(oracle_path, dict(results=[asdict(r) for r in full], audit=stats,
                                        signature=ordered_result_signature(full)))
        oracles[query.query_id], oracle_stats[query.query_id] = full, stats
    rotation = PREFERRED_VIDEO_ORDER.index(dataset) % len(STRICT_METHODS)
    order = STRICT_METHODS[rotation:] + STRICT_METHODS[:rotation]
    for method_index, method in enumerate(order):
        group_paths = {q.query_id: destination / 'groups' / f'{method}_{q.query_id}.json' for q in queries}
        pending = [q for q in queries if not group_paths[q.query_id].exists()]
        if method_complete(destination, method, group_paths.values()):
            continue
        strategy = None
        try:
            print(f'{dataset} {method}: setup', flush=True)
            setup_start = time.perf_counter()
            setup_kind = 'build'
            if method == 'Exact Frame Scan':
                strategy = ExactFrameScan(data, width, height)
                setup_kind = 'no_index'
            elif method == 'Ours':
                strategy = _OursStrategy(dataset, data, width, height)
                setup_kind = 'prebuilt_index_load_NOT_build'
            elif method == 'Craw-Element+Verifier':
                strategy = CrawElementStrategy(data, width, height,
                    artifact_dir=destination / 'artifacts' / ('craw_' + uuid.uuid4().hex[:10]))
            else:
                pairs = {(q.role_types[s], q.role_types[t]) for q in queries for s, t in q.relations}
                strategy = build_strategy(method, data, width, height,
                                          allowed_type_pairs=pairs if method == 'STAR-style' else None)
            setup_s = time.perf_counter()-setup_start
            print(f'{dataset} {method}: setup {setup_s:.3f}s', flush=True)
            for query in (pending if method_index % 2 == 0 else list(reversed(pending))):
                def progress(raw):
                    write_json(destination / 'progress.json', dict(dataset=dataset, method=method,
                        query_id=query.query_id, raw=raw, updated_utc=now()))
                    print(f'{dataset} {method} {query.query_id}: {len(raw)}/{repetitions} '
                          f'{raw[-1]["elapsed_s"]:.6f}s verified={raw[-1]["ordered_topk_verified"]}', flush=True)
                result = measure_group_safely(strategy, query, oracles[query.query_id],
                                       repetitions=repetitions, warmups=warmups, progress=progress)
                if result['status'] == 'failed':
                    result.update(dataset=dataset, method=method, query_id=query.query_id)
                    write_json(destination / f'failure_{method}_{query.query_id}.json', result)
                    print(f'{dataset} {method} {query.query_id}: FAILED {result["error"]}', flush=True)
                    continue
                audit, reference = result['audit'], oracle_stats[query.query_id]
                result.update(dataset=dataset, query_id=query.query_id, method=method,
                    protocol='strict_QST_continuous_fixed_binding', setup_s=setup_s, setup_kind=setup_kind,
                    source='measured_full_dataset', frame_pruning=1-audit['candidate_frames']/manifest['unique_frames'],
                    binding_frame_pruning=(1-audit['binding_frame_admitted']/reference['binding_frame_enumerated']
                                           if reference['binding_frame_enumerated'] else None),
                    predicate_reduction=(1-audit['spatial_predicate_evaluations']/reference['spatial_predicate_evaluations']
                                         if reference['spatial_predicate_evaluations'] else None),
                    pruning_applicable=method in PRUNING_METHODS, completed_utc=now())
                write_json(group_paths[query.query_id], result)
                print(f'{dataset} {method} {query.query_id}: full F1={result["metrics"]["f1"]:.6f}', flush=True)
            artifact_file = destination / 'artifacts' / f'{method}.pkl'
            artifact_file.parent.mkdir(exist_ok=True)
            size = (sum(p.stat().st_size for p in index_paths) if method == 'Ours' else 0
                    if method == 'Exact Frame Scan' else serialize_artifact(strategy, artifact_file, method))
            write_json(destination / 'artifacts' / f'{method}.metadata.json',
                       dict(method=method, setup_s=setup_s, setup_kind=setup_kind, payload_bytes=size))
        except Exception as error:
            failure = dict(method=method, error=repr(error), traceback=traceback.format_exc(), at_utc=now())
            write_json(destination / f'failure_{method}.json', failure)
            print(f'{dataset} {method}: FAILED {error!r}', flush=True)
        finally:
            if strategy is not None:
                strategy.release()
            del strategy
            gc.collect()
    manifest['finished_utc'] = now()
    manifest['integrity'] = dict(data=sha256(data_path) == identity['data_sha256'],
        source=all(sha256(ROOT / name) == value for name, value in identity['source_sha256'].items()),
        native=all(sha256(DEFAULT_CRAW_ROOT / name) == value for name, value in identity['native_sha256'].items()),
        index=all(sha256(path) == value for path, value in identity['index_sha256'].items()))
    files = list((destination / 'groups').glob('*.json')) if (destination / 'groups').exists() else []
    groups = [json.loads(p.read_text(encoding='utf-8')) for p in files]
    manifest['complete'] = all(method_complete(destination, m,
        [destination/'groups'/f'{m}_{q.query_id}.json' for q in queries]) for m in STRICT_METHODS) and all(manifest['integrity'].values())
    manifest['all_correct'] = manifest['complete'] and all(g['full_intervals_verified'] and g['ordered_topk_verified'] for g in groups)
    write_json(manifest_path, manifest)


def aggregate(output, identities):
    output = Path(output)
    rows, raw, status = [], [], []
    for dataset in PREFERRED_VIDEO_ORDER:
        validity = checkpoint_validity(output / dataset, identities[dataset])
        for query_id in ('Q1', 'Q2'):
            for method in ROSTER:
                path = output / dataset / 'groups' / f'{method}_{query_id}.json'
                if not path.exists() or validity != 'valid' or not (output/dataset/'artifacts'/f'{method}.metadata.json').exists():
                    reason = ('deferred_by_user_no_CUDA' if method == 'Time-R1' else
                              'separate_native_QBE_protocol_pending' if method == 'SketchQL' else
                              validity if validity != 'valid' else 'pending_or_failed')
                    status.append(dict(dataset=dataset, query_id=query_id, method=method, status=reason))
                    continue
                g = json.loads(path.read_text(encoding='utf-8'))
                status.append(dict(dataset=dataset, query_id=query_id, method=method, status='measured'))
                row = {key: g[key] for key in ('dataset','query_id','method','protocol','median_s','q1_s','q3_s',
                    'min_s','max_s','setup_s','setup_kind','source','frame_pruning','binding_frame_pruning',
                    'predicate_reduction','pruning_applicable','ordered_topk_verified','full_intervals_verified')}
                row.update(g['metrics'])
                row.update({key: g['audit'][key] for key in ('candidate_frames','binding_frame_enumerated',
                    'binding_frame_admitted','spatial_predicate_evaluations','candidate_s','verification_s')})
                rows.append(row)
                raw.extend(dict(dataset=dataset, query_id=query_id, method=method, **item) for item in g['raw'])
    csv_write(output / 'summary.csv', rows)
    csv_write(output / 'raw_measurements.csv', raw)
    csv_write(output / 'method_status.csv', status)
    return rows


def suite(output, workload, repetitions=3, warmups=1):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if repetitions < 1 or warmups < 0:
        raise ValueError('Invalid repetition counts')
    identities = {d: collect_identity(d, workload, repetitions, warmups) for d in PREFERRED_VIDEO_ORDER}
    identity_path = output / 'suite_identity.json'
    if identity_path.exists():
        previous = json.loads(identity_path.read_text(encoding='utf-8'))
        if set(previous) != set(identities):
            raise ValueError('Suite dataset coverage changed; use a fresh output directory')
        for dataset in identities:
            check_identity(previous[dataset], identities[dataset])
    elif any((output/d).exists() for d in PREFERRED_VIDEO_ORDER):
        raise ValueError('Dataset checkpoints without suite identity; use a fresh output directory')
    write_json(identity_path, identities)
    write_json(output / 'roster.json', dict(methods=ROSTER, strict_methods=STRICT_METHODS,
        datasets=PREFERRED_VIDEO_ORDER, queries=['Q1','Q2'], repetitions=repetitions, warmups=warmups,
        Time_R1='deferred by user: no GPU environment', SketchQL='native QBE protocol separate'))
    for dataset in PREFERRED_VIDEO_ORDER:
        command = [sys.executable, '-X', 'utf8', '-u', '-m', 'supplement.run_full_baseline_suite',
                   '--output-dir', str(output.resolve()), '--workload', str(Path(workload).resolve()),
                   '--worker', dataset, '--repetitions', str(repetitions), '--warmups', str(warmups)]
        print(f'START DATASET {dataset}', flush=True)
        with (output / f'{dataset}.log').open('a', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding='utf-8', errors='replace')
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end='', flush=True)
            returncode = process.wait()
        if returncode:
            write_json(output / f'{dataset}.worker_error.json', dict(returncode=returncode, at_utc=now()))
        aggregate(output, identities)
    rows = aggregate(output, identities)
    write_json(output / 'suite_status.json', dict(finished_utc=now(), measured_groups=len(rows),
        expected_strict_groups=80, all_strict_groups_measured=len(rows)==80,
        all_strict_correct=len(rows)==80 and all(r['full_intervals_verified'] and r['ordered_topk_verified'] for r in rows),
        Time_R1='deferred_by_user', SketchQL='see native protocol results'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--workload', type=Path, default=DEFAULT_WORKLOAD)
    parser.add_argument('--worker', choices=PREFERRED_VIDEO_ORDER)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--warmups', type=int, default=1)
    args = parser.parse_args()
    if args.worker:
        dataset_worker(args.output_dir, args.worker, args.workload, args.repetitions, args.warmups)
    else:
        suite(args.output_dir, args.workload, args.repetitions, args.warmups)
