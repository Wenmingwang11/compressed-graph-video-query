"""Audited completion plots: ten strict methods and separately labelled SketchQL.

No experiment runs here. The seven unchanged methods are read from the original
suite, while prepared Ours and both adapted systems require fresh measurements.
Raw timings, oracle outputs, input hashes and relevant current sources are
verified before publication. Historical Ours is used only for an optional
before/after plot. Visual-semantic metrics come from the independent scorer;
exact agreement with the structured scan is never shown as semantic accuracy.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('drtest', 'drtrain', 'bdd100kA', 'bdd100kB', 'I-24')
QUERIES = ('Q1', 'Q2')
OLD_METHODS = ('Ours', 'VQPy-style', 'VOCAL-UDF-style', 'Video-ColBERT-style',
               'LAVA-style', 'STAR-style', 'Exact Frame Scan', 'Craw-Element+Verifier')
NEW_METHODS = ('Ours', 'EVA-MV adapted', 'EQUI-VOCAL SQL adapted')
METHODS = (*OLD_METHODS, *NEW_METHODS[1:])
COLORS = dict(zip(METHODS, ('#3b5da3', '#0086c5', '#81a2ed', '#80cfff',
    '#73a57a', '#e58760', '#9a9a9a', '#b184a7', '#d4af52', '#4f9d8b')))
COLORS['SketchQL'] = '#ffc920'
DISPLAY = {'Craw-Element+Verifier': 'Craw-E+Verifier', 'Exact Frame Scan': 'Exact Scan'}
STRICT_PROTOCOL = 'strict_QST_continuous_fixed_binding'
NATIVE_PROTOCOL = 'answer_seeded_QBE_native_all_proposals_plus_QST_verifier'
NATIVE_VARIANT = 'SketchQL-QBE proposals + QST verifier'
WORKLOAD = ROOT/'supplement/out/corrected_baseline_20260712/workload.json'
OLD_SOURCE = ROOT/'supplement/out/full_baseline_suite_20260920'
NEW_SOURCE = ROOT/'supplement/out/completion_suite_20260926'
NATIVE_SOURCE = ROOT/'supplement/out/sketchql_native_full_20260920'
OUTPUT = Path('<LOCAL_USER_HOME>/Desktop/论文/latex-project/figures/baseline_completion_20260926')
CRAW_SOURCE = Path('D:/pycharm/_baseline_prepare_20260919/Craw_Submission-source/sys')
SKETCH_MODEL = Path('D:/pycharm/SketchQL-main/data/model_checkpoint/model_cp.pt')
# This file is not executed by the seven retained historical methods. All
# shared runners, query definitions and baseline implementations are checked.
HISTORICAL_OURS_ONLY_SOURCE = 'vsimsearch/corrected_mli_querying.py'
_HASH_CACHE = {}


def sha256(path):
    path = Path(path)
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if key not in _HASH_CACHE:
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024*1024), b''):
                digest.update(block)
        _HASH_CACHE[key] = digest.hexdigest()
    return _HASH_CACHE[key]


def record_source(path, provenance, role='input'):
    value = dict(file=str(Path(path).resolve()), sha256=sha256(path), role=role)
    if value not in provenance:
        provenance.append(value)


def verify_hash(path, expected, provenance, role='verified_input'):
    if sha256(path) != expected:
        raise ValueError(f'hash mismatch: {path}')
    record_source(path, provenance, role)


def read_json(path, provenance=None, role='measurement'):
    value = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if provenance is not None:
        record_source(path, provenance, role)
    return value


def quantile_linear(values, fraction):
    values = sorted(values)
    location = (len(values)-1)*fraction
    left, right = math.floor(location), math.ceil(location)
    return values[left] + (values[right]-values[left])*(location-left)


def timing_ms(group, repetitions=3):
    if group.get('source') != 'measured_full_dataset':
        raise ValueError('Only measured full-dataset timing may be plotted')
    times = [r['elapsed_s'] for r in group['raw']]
    if len(times) != repetitions or repetitions != 3:
        raise ValueError('Exactly three raw timing repetitions are required')
    if any(not isinstance(t, (int, float)) or not math.isfinite(t) or t <= 0 for t in times):
        raise ValueError('Invalid raw timing')
    median = statistics.median(times)
    q1, q3 = quantile_linear(times, .25), quantile_linear(times, .75)
    for key, value in (('median_s', median), ('q1_s', q1), ('q3_s', q3)):
        if not math.isclose(value, group[key], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f'{key} does not match raw timing repetitions')
    return median*1000, (median-q1)*1000, (q3-median)*1000


def result_signature(rows):
    payload = [[r['binding'], int(r['start_frame']), int(r['end_frame'])] for r in rows]
    return hashlib.sha256(json.dumps(payload, separators=(',', ':')).encode('ascii')).hexdigest()


def validate_strict_group(group, dataset, query, method, oracle, topk):
    if (group.get('dataset'), group.get('query_id'), group.get('method'), group.get('protocol')) != (
            dataset, query, method, STRICT_PROTOCOL):
        raise ValueError(f'Mislabeled strict group: {dataset}/{query}/{method}')
    timing_ms(group)
    expected = oracle['signature']
    top_signature = result_signature(oracle['results'][:topk])
    if (group.get('status') != 'measured' or group.get('full_intervals_verified') is not True or
            group.get('ordered_topk_verified') is not True or
            group.get('oracle_all_signature') != expected or
            group.get('oracle_topk_signature') != top_signature or
            result_signature(group['full_results']) != expected or
            any(r.get('ordered_topk_verified') is not True or r.get('signature') != top_signature
                for r in group['raw'])):
        raise ValueError(f'Unverified full result or Top-k: {dataset}/{query}/{method}')
    metrics = group['metrics']
    if metrics['fp'] != 0 or metrics['fn'] != 0 or metrics['tp'] != len(oracle['results']):
        raise ValueError('Strict execution-agreement counters do not match oracle')


def validate_coverage(rows, methods):
    keys = [(r['dataset'], r['query_id'], r['method']) for r in rows]
    expected = set(itertools.product(DATASETS, QUERIES, methods))
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError(f'Incomplete or duplicate coverage: expected {len(expected)}, got {len(keys)}')


def load_old(source, workload=WORKLOAD, craw_source=CRAW_SOURCE):
    source = Path(source)
    provenance, groups, setup, oracles, manifests = [], [], [], {}, {}
    identities = read_json(source/'suite_identity.json', provenance, 'historical_suite_identity')
    records = read_json(workload, provenance, 'fixed_workload')
    topks = {(q['dataset'], q['query_id']): q['topk'] for q in records}
    for dataset in DATASETS:
        directory = source/dataset
        manifest = read_json(directory/'manifest.json', provenance, 'historical_manifest')
        manifests[dataset] = manifest
        if (manifest.get('complete') is not True or manifest.get('all_correct') is not True or
                not all(manifest.get('integrity', {}).get(k) is True for k in ('data', 'source', 'native', 'index'))):
            raise ValueError(f'Incomplete or unverified old manifest: {dataset}')
        for key, value in identities[dataset].items():
            if manifest.get(key) != value:
                raise ValueError(f'Old suite identity mismatch: {dataset}/{key}')
        verify_hash(manifest['data_path'], manifest['data_sha256'], provenance, 'data')
        verify_hash(workload, manifest['workload_sha256'], provenance, 'workload')
        for name, value in manifest['source_sha256'].items():
            if name != HISTORICAL_OURS_ONLY_SOURCE:
                verify_hash(ROOT/name, value, provenance, 'unchanged_historical_source')
        for name, value in manifest['native_sha256'].items():
            verify_hash(Path(craw_source)/name, value, provenance, 'native_craw_source')
        for query in QUERIES:
            oracle = read_json(directory/f'oracle_{query}.json', provenance, 'scan_oracle')
            if result_signature(oracle['results']) != oracle['signature']:
                raise ValueError('Corrupted historical scan oracle')
            oracles[dataset, query] = oracle
        for method in OLD_METHODS:
            metadata = read_json(directory/'artifacts'/f'{method}.metadata.json', provenance, 'historical_setup')
            setup.append(dict(dataset=dataset, method=method, version='historical', **{
                k: metadata.get(k) for k in ('setup_s', 'setup_kind', 'payload_bytes')}))
            for query in QUERIES:
                group = read_json(directory/'groups'/f'{method}_{query}.json', provenance)
                validate_strict_group(group, dataset, query, method, oracles[dataset, query], topks[dataset, query])
                groups.append(group)
    validate_coverage(groups, OLD_METHODS)
    return groups, manifests, oracles, topks, setup, provenance


def load_completion(source, old_source=OLD_SOURCE, workload=WORKLOAD, craw_source=CRAW_SOURCE):
    old, manifests, oracles, topks, setup, provenance = load_old(old_source, workload, craw_source)
    groups = [g for g in old if g['method'] != 'Ours']
    for dataset, method in itertools.product(DATASETS, NEW_METHODS):
        directory = Path(source)/dataset
        manifest = read_json(directory/f'manifest_{method}.json', provenance, 'completion_manifest')
        identity = manifest['identity']
        if manifest.get('complete') is not True or manifest.get('integrity') is not True:
            raise ValueError(f'Incomplete completion manifest: {dataset}/{method}')
        if (identity.get('dataset'), identity.get('method'), manifest.get('protocol')) != (dataset, method, STRICT_PROTOCOL):
            raise ValueError('Completion manifest labels or protocol mismatch')
        if identity.get('repetitions') != 3 or identity.get('warmups', 0) < 1:
            raise ValueError('Three repetitions and at least one warmup are required')
        for key in ('data_sha256', 'workload_sha256'):
            if identity[key] != manifests[dataset][key]:
                raise ValueError(f'Completion/old input mismatch: {dataset}/{key}')
        for name, value in identity['source_sha256'].items():
            verify_hash(ROOT/name, value, provenance, 'completion_source')
        for name, value in identity.get('index_sha256', {}).items():
            verify_hash(name, value, provenance, 'prebuilt_ours_index')
        for query in QUERIES:
            verify_hash(Path(old_source)/dataset/f'oracle_{query}.json', identity['oracle_sha256'][query], provenance, 'scan_oracle')
            group = read_json(directory/'groups'/f'{method}_{query}.json', provenance)
            validate_strict_group(group, dataset, query, method, oracles[dataset, query], topks[dataset, query])
            groups.append(group)
        setup.append(dict(dataset=dataset, method=method, version='completion',
            setup_s=manifest.get('setup_seconds'), setup_kind='query_independent_preparation',
            payload_bytes=manifest.get('metadata', {}).get('prebuilt_artifact_bytes'),
            metadata=manifest.get('metadata', {})))
    validate_coverage(groups, METHODS)
    return groups, old, manifests, setup, provenance


def load_native(source, old_source, manifests, provenance):
    native = []
    if source is None:
        return native
    for dataset in DATASETS[:-1]:
        directory = Path(source)/dataset
        manifest = read_json(directory/'manifest.json', provenance, 'separate_native_manifest')
        if manifest.get('complete') is not True or manifest.get('integrity') is not True:
            raise ValueError(f'Incomplete native measurement: {dataset}')
        identity = manifest['identity']
        for key in ('data_sha256', 'workload_sha256'):
            if identity[key] != manifests[dataset][key]:
                raise ValueError('Native/strict input mismatch')
        for name, value in identity['source_sha256'].items():
            verify_hash(name, value, provenance, 'native_sketchql_source')
        verify_hash(SKETCH_MODEL, identity['model_sha256'], provenance, 'native_sketchql_checkpoint')
        for query in QUERIES:
            verify_hash(Path(old_source)/dataset/f'oracle_{query}.json', identity['oracle_sha256'][query], provenance, 'scan_oracle')
            group = read_json(directory/f'{query}.json', provenance, 'separate_native_measurement')
            if (group.get('dataset'), group.get('query_id'), group.get('method'), group.get('protocol'),
                    group.get('method_variant')) != (dataset, query, 'SketchQL', NATIVE_PROTOCOL, NATIVE_VARIANT):
                raise ValueError('Mislabeled native group')
            timing_ms(group, identity['repetitions'])
            proposals = read_json(directory/f'{query}.proposals.json', provenance, 'native_proposals')
            if not isinstance(proposals.get('masks'), list) or not isinstance(proposals.get('proposals'), list):
                raise ValueError('Missing native proposal evidence')
            if not isinstance(group.get('full_results'), list):
                raise ValueError('Missing native full-result evidence')
            native.append(group)
    return native


def validate_counts(row):
    counts = ('tp', 'fp', 'fn', 'prediction_count', 'reference_count')
    if any(type(row.get(k)) is not int or row[k] < 0 for k in counts):
        raise ValueError('Semantic counts must be nonnegative integers')
    if row['tp']+row['fp'] != row['prediction_count'] or row['tp']+row['fn'] != row['reference_count']:
        raise ValueError('Semantic counts do not balance')
    for key, numerator, denominator in (('precision', row['tp'], row['prediction_count']),
            ('recall', row['tp'], row['reference_count']),
            ('f1', 2*row['tp'], row['prediction_count']+row['reference_count'])):
        value = row.get(key)
        if denominator == 0:
            if value is not None:
                raise ValueError(f'Undefined {key} must remain null')
        elif value is None or not math.isclose(value, numerator/denominator, abs_tol=1e-12):
            raise ValueError(f'Semantic {key} does not match counts')


def validate_semantic(document, methods=METHODS):
    protocol = document['protocol']
    if (protocol.get('primary_tiou_threshold') != .5 or protocol.get('human_ground_truth') is not False or
            protocol.get('result_scope') != 'all_before_topk' or
            protocol.get('reference_kind') != 'independent_model_assisted_AI_reviewed_tracklet_inventory'):
        raise ValueError('Unexpected semantic reference protocol')
    overall = [r for r in document['overall'] if r['tiou_threshold'] == .5]
    per_query = [r for r in document['per_query'] if r['tiou_threshold'] == .5]
    keys = [(r['method'], r['query_id']) for r in per_query]
    expected = set(itertools.product(methods, (f'SEM{i:02d}' for i in range(1, 7))))
    if len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError('Incomplete semantic six-query coverage')
    if len(overall) != len(methods) or {r['method'] for r in overall} != set(methods):
        raise ValueError('Incomplete semantic micro method coverage')
    for row in [*overall, *per_query]:
        validate_counts(row)
    for row in overall:
        if row.get('aggregation') != 'micro_over_six_queries':
            raise ValueError('Semantic main plot requires micro aggregation')
        selected = [q for q in per_query if q['method'] == row['method']]
        for key in ('tp', 'fp', 'fn', 'prediction_count', 'reference_count'):
            if row[key] != sum(q[key] for q in selected):
                raise ValueError('Semantic micro count differs from per-query counts')
    return overall, per_query


def load_semantic(source, provenance):
    source = Path(source)
    path = source/'metrics.json' if source.is_dir() else source
    manifest = read_json(path.with_name('score_manifest.json'), provenance, 'semantic_score_manifest')
    if manifest.get('status') != 'completed' or manifest.get('human_ground_truth') is not False:
        raise ValueError('Incomplete or mislabeled semantic score manifest')
    for name, value in manifest['sources'].items():
        verify_hash(name, value, provenance, 'semantic_reference_or_prediction_source')
    for name, value in manifest['output_hashes'].items():
        verify_hash(path.parent/name, value, provenance, 'semantic_scoring_output')
    document = read_json(path, provenance, 'semantic_metrics')
    overall, queries = validate_semantic(document)
    return document, overall, queries


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'], 'font.size': 16,
        'axes.linewidth': 1.5, 'axes.unicode_minus': False, 'pdf.fonttype': 42, 'ps.fonttype': 42})


def grouped_figure(series, ylabel, *, log=False, note='', error=None, percent=True):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter
    import numpy as np
    fig, ax = plt.subplots(figsize=(6.9, 3.9))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    large = len(series) > 8
    fig.subplots_adjust(left=.145, right=.99, bottom=.20, top=.70 if large else .77)
    x = np.arange(len(DATASETS), dtype=float)
    width = .80/len(series)
    for number, (method, values, color) in enumerate(series):
        positions = x-.40+(number+.5)*width
        ax.bar(positions, values, width=width*.92, label=DISPLAY.get(method, method), color=color, edgecolor='none')
        if error and method in error:
            ax.errorbar(positions, values, yerr=error[method], fmt='none', ecolor='#333333', elinewidth=.55, capsize=1.2)
        for position, value in zip(positions, values):
            if not math.isfinite(value):
                ax.text(position, .03, 'N/A', transform=ax.get_xaxis_transform(), rotation=90, ha='center', fontsize=7)
            elif value == 0 and not log:
                ax.text(position, 1.2, '0', ha='center', va='bottom', fontsize=8, color=color)
    finite = [v for _, values, _ in series for v in values if math.isfinite(v)]
    if log:
        if error:
            for method, values, _ in series:
                if method in error:
                    finite.extend(v+e for v, e in zip(values, error[method][1]) if math.isfinite(v+e))
                    finite.extend(v-e for v, e in zip(values, error[method][0]) if math.isfinite(v-e))
        if not finite or min(finite) <= 0:
            raise ValueError('Logarithmic plot requires positive measured values')
        ax.set_yscale('log')
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(10**math.floor(math.log10(min(finite)*.8)), 10**math.ceil(math.log10(max(finite)*1.15)))
    elif percent:
        lower = min(0, 25*math.floor(min(finite, default=0)/25))
        upper = max(105, 25*math.ceil(max(finite, default=0)/25))
        ax.set_ylim(lower, upper)
        ax.set_yticks(list(range(lower, int(upper)+1, 25)))
    ax.set_ylabel(ylabel)
    ax.set_xticks(x, DATASETS)
    ax.set_xlim(-.5, len(DATASETS)-.5)
    ax.tick_params(axis='x', labelsize=14, length=0, pad=10)
    ax.tick_params(axis='y', which='major', labelsize=15, length=0)
    ax.tick_params(axis='y', which='minor', length=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
    handles, labels = ax.get_legend_handles_labels()
    columns = 3 if large else min(4, len(series))
    order = [r*columns+c for c in range(columns) for r in range(math.ceil(len(series)/columns)) if r*columns+c < len(series)]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc='upper center',
        bbox_to_anchor=(.5, 1.56 if large else 1.28), ncol=columns, frameon=False,
        fontsize=8.6 if large else 9.5, columnspacing=.85, handletextpad=.4, handlelength=1.25)
    fig.text(.5, .025, note, ha='center', fontsize=8.6)
    return fig


def semantic_figure(rows, metric):
    import matplotlib.pyplot as plt
    import numpy as np
    index = {r['method']: r for r in rows}
    methods = [m for m in METHODS if m in index]
    if set(index) - set(METHODS):
        raise ValueError('Unknown method in semantic figure')
    values = [index[m][metric] for m in methods]
    fig, ax = plt.subplots(figsize=(6.9, 3.9))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    fig.subplots_adjust(left=.32, right=.96, bottom=.25, top=.95)
    y = np.arange(len(methods))
    ax.barh(y, [float('nan') if v is None else v*100 for v in values], height=.68,
            color=[COLORS[m] for m in methods], edgecolor='none')
    ax.set_yticks(y, [DISPLAY.get(m, m) for m in methods], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel(f'Model-reference {metric} (%)', fontsize=15)
    ax.tick_params(axis='both', length=0)
    for position, value in zip(y, values):
        ax.text(1 if value is None else value*100+1.4, position,
            'N/A' if value is None else f'{value*100:.2f}', va='center', fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
    equal = len({index[m][metric] for m in methods}) == 1
    fig.text(.5, .06, 'I-24 first 60 s | Six-query micro average | Event tIoU >= 0.5', ha='center', fontsize=8.5)
    fig.text(.5, .02, 'Independent model reference; not human ground truth.' +
             (' All methods have the same score.' if equal else ''), ha='center', fontsize=8.5)
    return fig


def save_figure(fig, output, stem, book, outputs):
    import matplotlib.pyplot as plt
    for suffix in ('pdf', 'png'):
        path = output/f'{stem}.{suffix}'
        fig.savefig(path, dpi=300, facecolor='white', transparent=False)
        outputs.append(path)
    book.savefig(fig, facecolor='white')
    plt.close(fig)


def measurement_rows(groups, native):
    rows = []
    by_key = {(g['dataset'], g['query_id'], g['method']): g for g in [*groups, *native]}
    for dataset, query, method in itertools.product(DATASETS, QUERIES, (*METHODS, 'SketchQL')):
        group = by_key.get((dataset, query, method))
        row = dict(dataset=dataset, query_id=query, method=method,
            protocol=NATIVE_PROTOCOL if method == 'SketchQL' else STRICT_PROTOCOL,
            status=('deferred' if dataset == 'I-24' and method == 'SketchQL' else 'not_requested') if group is None
                else 'measured_separate_answer_seeded_QBE' if method == 'SketchQL' else 'measured_strict_QST')
        if group:
            median, lower, upper = timing_ms(group)
            row.update(median_ms=median, q1_ms=median-lower, q3_ms=median+upper, repetitions=3,
                raw_seconds=[r['elapsed_s'] for r in group['raw']],
                execution_correctness_not_semantic_accuracy=True)
            for key in ('frame_pruning', 'binding_frame_pruning', 'predicate_reduction',
                        'ordered_topk_verified', 'full_intervals_verified', 'setup_s', 'setup_kind'):
                row[key] = group.get(key)
        rows.append(row)
    return rows


def render(source, old_source, output, native_source=NATIVE_SOURCE, semantic_source=None,
           workload=WORKLOAD, craw_source=CRAW_SOURCE, include_speedup=False):
    groups, old, manifests, setup, provenance = load_completion(source, old_source, workload, craw_source)
    native = load_native(native_source, old_source, manifests, provenance)
    semantic = load_semantic(semantic_source, provenance) if semantic_source else None
    output = Path(output)
    if output.resolve() in {Path(source).resolve(), Path(old_source).resolve()}:
        raise ValueError('Figure output cannot be a measurement source directory')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Refusing to mix or overwrite prior figures; choose an empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    style()
    from matplotlib.backends.backend_pdf import PdfPages
    index = {(g['dataset'], g['query_id'], g['method']): g for g in groups}
    outputs, omissions = [], []
    book_path = output/'completion_measured_20260926.pdf'
    with PdfPages(book_path) as book:
        for query in QUERIES:
            series, errors = [], {}
            for method in METHODS:
                times = [timing_ms(index[d, query, method]) for d in DATASETS]
                series.append((method, [v[0] for v in times], COLORS[method]))
                errors[method] = [[v[1] for v in times], [v[2] for v in times]]
            fig = grouped_figure(series, 'Query Time (ms)', log=True, error=errors,
                note=f'{query} | Full sampled data; online query only; 3 runs, IQR bars')
            save_figure(fig, output, f'query_time_{query.lower()}_ms', book, outputs)
            specs = [('frame_pruning', 'Candidate Frame\nPruning (%)', [m for m in OLD_METHODS if m != 'Exact Frame Scan']),
                     ('binding_frame_pruning', 'Binding-Frame\nReduction (%)', ['Ours', 'STAR-style']),
                     ('predicate_reduction', 'Exact Predicate\nReduction (%)', ['Ours', 'STAR-style'])]
            for key, label, candidates in specs:
                series = []
                for method in candidates:
                    selected = [index[d, query, method] for d in DATASETS]
                    if not all(g.get('pruning_applicable') is True and isinstance(g.get(key), (int, float))
                               and math.isfinite(g[key]) for g in selected):
                        omissions.append(dict(query_id=query, metric=key, method=method, reason='No complete comparable measured counter'))
                        continue
                    series.append((method, [g[key]*100 for g in selected], COLORS[method]))
                if series:
                    note = f'{query} | Separate instrumented audit; complete scan denominator'
                    fig = grouped_figure(series, label, note=note)
                    save_figure(fig, output, f'{key}_{query.lower()}', book, outputs)
        if native:
            ni = {(g['dataset'], g['query_id']): g for g in native}
            series, errors = [], {}
            for query, color in [('Q1', COLORS['Ours']), ('Q2', COLORS['SketchQL'])]:
                times = [timing_ms(ni[d, query]) if (d, query) in ni else (math.nan, math.nan, math.nan) for d in DATASETS]
                series.append((query, [t[0] for t in times], color))
                errors[query] = [[t[1] for t in times], [t[2] for t in times]]
            fig = grouped_figure(series, 'Query Time (ms)', log=True, error=errors,
                note='SketchQL + Verifier | Separate answer-seeded QBE protocol; I-24 deferred')
            save_figure(fig, output, 'sketchql_separate_qbe_query_time_ms', book, outputs)
        if include_speedup:
            oi = {(g['dataset'], g['query_id']): g for g in old if g['method'] == 'Ours'}
            series = [(q, [oi[d, q]['median_s']/index[d, q, 'Ours']['median_s'] for d in DATASETS], color)
                      for q, color in [('Q1', COLORS['Ours']), ('Q2', COLORS['Video-ColBERT-style'])]]
            fig = grouped_figure(series, 'Ours Speedup (times)', log=True,
                note='Historical / optimized median online query time; same complete outputs')
            save_figure(fig, output, 'ours_before_after_speedup', book, outputs)
        if semantic:
            document, overall, queries = semantic
            for metric in ('precision', 'recall'):
                fig = semantic_figure(overall, metric)
                save_figure(fig, output, f'model_reference_{metric}_micro', book, outputs)
    outputs.append(book_path)
    tables = {'query_time_and_pruning.csv': measurement_rows(groups, native),
              'offline_preparation.csv': setup}
    if semantic:
        tables['semantic_primary_micro.csv'] = semantic[1]
        tables['semantic_primary_per_query.csv'] = semantic[2]
        tables['semantic_tiou_sensitivity.csv'] = semantic[0]['per_query']
    for filename, rows in tables.items():
        write_csv(output/filename, rows)
        outputs.append(output/filename)
    record_source(Path(__file__), provenance, 'renderer_source')
    manifest = dict(generated_at_utc=datetime.now(timezone.utc).isoformat(), strict_groups=len(groups),
        strict_methods=list(METHODS), separate_native_groups=len(native),
        semantic_primary_method_count=len(semantic[1]) if semantic else 0,
        figure_size_inches=[6.9, 3.9], png_dpi=300, font_family='Times New Roman', method_colors=COLORS,
        sources=provenance, outputs=[dict(file=str(p.resolve()), sha256=sha256(p)) for p in outputs],
        timing_scope='Preloaded online query, excluding independently measured offline preparation; no substituted timings.',
        native_scope='SketchQL has answer-derived exemplars and a different QBE protocol; shown separately. I-24 deferred.',
        semantic_scope=semantic[0]['protocol'] if semantic else 'Not requested',
        semantic_reference_is_human_ground_truth=False,
        pruning_omissions=omissions,
        newly_added_pruning_policy='EVA-MV and EQUI-VOCAL excluded: no comparable measured pruning counter.',
        historical_source_verification_exception=dict(file=HISTORICAL_OURS_ONLY_SOURCE,
            reason='Historical Ours only; historical manifest/suite identity and exact results verified. The other seven sources are checked against current files.'),
        style_reference='supplement/render_full_baseline_suite.py; figures/baseline_20260926',
        status='complete')
    (output/'figure_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(status='complete', output=str(output), strict_groups=len(groups),
        native_groups=len(native), individual_plots=sum(p.suffix == '.png' for p in outputs)), ensure_ascii=True))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=NEW_SOURCE)
    parser.add_argument('--old-source', type=Path, default=OLD_SOURCE)
    parser.add_argument('--native-source', type=Path, default=NATIVE_SOURCE)
    parser.add_argument('--semantic-source', type=Path, help='Completed semantic metrics.json or its directory')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--workload', type=Path, default=WORKLOAD)
    parser.add_argument('--craw-source', type=Path, default=CRAW_SOURCE)
    parser.add_argument('--without-native', action='store_true')
    parser.add_argument('--include-speedup', action='store_true')
    args = parser.parse_args()
    render(args.source, args.old_source, args.output, None if args.without_native else args.native_source,
           args.semantic_source, args.workload, args.craw_source, args.include_speedup)
