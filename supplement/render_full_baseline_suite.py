"""Publication plots from verified full-dataset measurements only; no overrides."""
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

DATASETS = ['drtest','drtrain','bdd100kA','bdd100kB','I-24']
METHODS = ['Ours','VQPy-style','VOCAL-UDF-style','Video-ColBERT-style','LAVA-style',
           'STAR-style','Exact Frame Scan','Craw-Element+Verifier']
COLORS = {'Ours':'#3b5da3','VQPy-style':'#0086c5','VOCAL-UDF-style':'#81a2ed',
          'Video-ColBERT-style':'#80cfff','LAVA-style':'#73a57a','STAR-style':'#e58760',
          'Exact Frame Scan':'#9a9a9a','Craw-Element+Verifier':'#b184a7','SketchQL':'#ffc920'}
DISPLAY = {'Craw-Element+Verifier':'Craw-E+Verifier','Exact Frame Scan':'Exact Scan'}
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path('<LOCAL_USER_HOME>/Desktop/论文/latex-project/figures')


def validate_coverage(rows,datasets,queries,methods):
    actual = [(r['dataset'],r['query_id'],r['method']) for r in rows]
    expected = set(itertools.product(datasets,queries,methods))
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError(f'Incomplete/duplicate coverage: expected {len(expected)}, got {len(actual)}')


def verified_median(group,repetitions):
    if group.get('source') != 'measured_full_dataset':
        raise ValueError('Only full-dataset measured data may be plotted')
    times = [r['elapsed_s'] for r in group['raw']]
    if len(times)!=repetitions or any(not math.isfinite(t) or t<=0 for t in times):
        raise ValueError('Invalid raw timing repetitions')
    median = statistics.median(times)
    if not math.isclose(median,group['median_s'],rel_tol=1e-12,abs_tol=1e-12):
        raise ValueError('Saved median does not match raw measurements')
    return median


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def load_groups(source,native_source=None):
    source = Path(source)
    identities = json.loads((source/'suite_identity.json').read_text(encoding='utf-8'))
    groups,provenance = [],[]
    for dataset in DATASETS:
        directory = source/dataset
        manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
        if not manifest.get('complete') or not all(manifest.get('integrity',{}).get(k) is True for k in ('data','source','native','index')):
            raise ValueError(f'{dataset}: incomplete or unverified manifest')
        for key,value in identities[dataset].items():
            if manifest.get(key) != value:
                raise ValueError(f'{dataset}: manifest identity mismatch {key}')
        for method,query in itertools.product(METHODS,['Q1','Q2']):
            metadata = directory/'artifacts'/f'{method}.metadata.json'
            if not metadata.exists():
                raise ValueError(f'{dataset} {method}: missing artifact metadata')
            path = directory/'groups'/f'{method}_{query}.json'
            group = json.loads(path.read_text(encoding='utf-8'))
            verified_median(group,manifest['repetitions'])
            if (group['dataset'],group['query_id'],group['method']) != (dataset,query,method):
                raise ValueError('Mislabeled group file')
            groups.append(group)
            provenance.append(dict(file=str(path.resolve()),sha256=sha256(path)))
    validate_coverage(groups,DATASETS,['Q1','Q2'],METHODS)
    native = []
    if native_source:
        for dataset in DATASETS:
            directory = Path(native_source)/dataset
            manifest_path = directory/'manifest.json'
            if not manifest_path.exists():
                continue
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            if manifest.get('integrity') is not True:
                continue
            native_identity = manifest['identity']
            for key in ('data_sha256','workload_sha256'):
                if native_identity[key] != identities[dataset][key]:
                    raise ValueError(f'Native/strict input mismatch: {dataset} {key}')
            for query in ['Q1','Q2']:
                from supplement.sketchql_native_adapter import native_group_valid
                if not native_group_valid(directory,dataset,query,manifest['identity']['repetitions']):
                    continue
                if native_identity['oracle_sha256'][query] != sha256(source/dataset/f'oracle_{query}.json'):
                    raise ValueError(f'Native/strict oracle mismatch: {dataset} {query}')
                path = directory/f'{query}.json'
                group = json.loads(path.read_text(encoding='utf-8'))
                if (group['dataset'],group['query_id'],group['method'],group['protocol']) != (
                    dataset,query,'SketchQL','answer_seeded_QBE_native_all_proposals_plus_QST_verifier'):
                    raise ValueError('Mislabeled native group file')
                if group.get('method_variant') != 'SketchQL-QBE proposals + QST verifier':
                    raise ValueError('Native variant label missing or changed')
                verified_median(group,manifest['identity']['repetitions'])
                native.append(group)
                provenance.append(dict(file=str(path.resolve()),sha256=sha256(path)))
    return groups,native,provenance


def save_figure(fig,output,stem,pdf_book,outputs):
    pdf, png = output/f'{stem}.pdf',output/f'{stem}.png'
    fig.savefig(pdf,facecolor='white',transparent=False)
    fig.savefig(png,dpi=300,facecolor='white',transparent=False)
    pdf_book.savefig(fig,facecolor='white')
    outputs.extend([str(pdf),str(png)])
    import matplotlib.pyplot as plt
    plt.close(fig)


def grouped_figure(series,ylabel,*,log=False,note='',error=None):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter
    import numpy as np
    fig,ax = plt.subplots(figsize=(6.9,3.9))
    fig.patch.set_facecolor('white'); ax.set_facecolor('white')
    fig.subplots_adjust(left=.145,right=.99,bottom=.20,top=.77)
    x = np.arange(len(DATASETS),dtype=float)
    width = .80/len(series)
    for index,(name,values,color) in enumerate(series):
        positions = x-.40+(index+.5)*width
        ax.bar(positions,values,width=width*.92,label=DISPLAY.get(name,name),color=color,edgecolor='none')
        if error and name in error:
            ax.errorbar(positions,values,yerr=error[name],fmt='none',ecolor='#333333',elinewidth=.55,capsize=1.2)
        for position,value in zip(positions,values):
            if not np.isfinite(value):
                ax.text(position,.03,'N/A',transform=ax.get_xaxis_transform(),rotation=90,ha='center',fontsize=7)
            elif value == 0 and not log:
                ax.text(position,1.2,'0',ha='center',va='bottom',fontsize=8,color=color)
    if log:
        finite = [v for _,values,_ in series for v in values if np.isfinite(v) and v>0]
        if error:
            for name,values,_ in series:
                if name in error:
                    finite.extend(v+e for v,e in zip(values,error[name][1]) if np.isfinite(v+e) and v+e>0)
                    finite.extend(v-e for v,e in zip(values,error[name][0]) if np.isfinite(v-e) and v-e>0)
        ax.set_yscale('log')
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(10**math.floor(math.log10(min(finite)*.8)),10**math.ceil(math.log10(max(finite)*1.15)))
    else:
        ax.set_ylim(0,105)
        ax.set_yticks([0,25,50,75,100])
    ax.set_ylabel(ylabel)
    ax.set_xticks(x,DATASETS); ax.set_xlim(-.5,len(DATASETS)-.5)
    ax.tick_params(axis='x',labelsize=14,length=0,pad=10)
    ax.tick_params(axis='y',which='major',labelsize=15,length=0)
    ax.tick_params(axis='y',which='minor',length=0)
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_linewidth(1.5)
    handles,labels = ax.get_legend_handles_labels()
    columns = 4 if len(series)>4 else len(series)
    order = [r*columns+c for c in range(columns) for r in range(math.ceil(len(series)/columns)) if r*columns+c<len(series)]
    ax.legend([handles[i] for i in order],[labels[i] for i in order],loc='upper center',bbox_to_anchor=(.5,1.28),
              ncol=columns,frameon=False,fontsize=9.5,columnspacing=.85,handletextpad=.4,handlelength=1.25)
    fig.text(.5,.025,note,ha='center',fontsize=9)
    return fig


def render(source,output,native_source=None,figure_set='paper'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import numpy as np
    if figure_set not in ('paper','all'):
        raise ValueError('figure_set must be paper or all')
    groups,native,provenance = load_groups(source,native_source)
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'font.size':16,
        'axes.linewidth':1.5,'axes.unicode_minus':False,'pdf.fonttype':42,'ps.fonttype':42})
    index = {(g['dataset'],g['query_id'],g['method']):g for g in groups}
    outputs = []
    book = output/'baseline_full_suite_measured_20260920.pdf'
    metric_specs = [('median_s','Query Time (s)',METHODS,True),
        ('frame_pruning','Candidate Frame\nPruning (%)',[m for m in METHODS if m!='Exact Frame Scan'],False),
        ('binding_frame_pruning','Binding-Frame\nReduction (%)',['Ours','STAR-style'],False),
        ('predicate_reduction','Verifier Predicate\nReduction (%)',['Ours','STAR-style'],False),
        ('precision','Interval Precision (%)',METHODS,False),('recall','Interval Recall (%)',METHODS,False),
        ('f1','Interval F1 (%)',METHODS,False)]
    if figure_set == 'paper':
        metric_specs = metric_specs[:4]
    with PdfPages(book) as pdf_book:
        for query in ('Q1','Q2'):
            for metric,label,methods,log in metric_specs:
                series,error = [],{}
                for method in methods:
                    selected = [index[d,query,method] for d in DATASETS]
                    values = [(g['metrics'][metric] if metric in ('precision','recall','f1') else g[metric]) for g in selected]
                    values = [np.nan if v is None else v*(1 if log else 100) for v in values]
                    series.append((method,values,COLORS[method]))
                    if log:
                        error[method] = [[g['median_s']-g['q1_s'] for g in selected],
                                         [g['q3_s']-g['median_s'] for g in selected]]
                note = f'{query} | Full sampled-frame data; '
                note += ('median of 3 runs, IQR bars' if log else 'exact interval match against scan' if metric in ('precision','recall','f1')
                         else 'exact verifier calls; separate audit' if metric=='predicate_reduction'
                         else 'separate instrumented query audit')
                fig = grouped_figure(series,label,log=log,note=note,error=error)
                save_figure(fig,output,f'baseline_{metric}_{query.lower()}_measured_20260920',pdf_book,outputs)
            if figure_set == 'all':
                # These diagnostic charts are excluded from the user-selected paper set.
                selected = [index[d,query,'Ours'] for d in DATASETS]
                series = [(name,[g['audit'][key] for g in selected],color) for name,key,color in
                          [('Candidate generation','candidate_s','#80cfff'),('Exact verification','verification_s','#e58760')]]
                fig = grouped_figure(series,'Ours Phase Time (s)',log=True,
                    note=f'{query} | One instrumented audit; index construction excluded')
                save_figure(fig,output,f'baseline_ours_phases_{query.lower()}_measured_20260920',pdf_book,outputs)
        if native and figure_set == 'all':
            ni = {(g['dataset'],g['query_id']):g for g in native}
            def native_values(query,metric):
                values = [(ni[d,query]['metrics'][metric] if metric in ('precision','recall','f1') else ni[d,query][metric])
                        if (d,query) in ni else np.nan for d in DATASETS]
                # The raw metric uses an explicit empty-set convention. An empty
                # prediction has undefined precision, so do not draw a 100% bar.
                if metric=='precision':
                    values = [value if (d,query) not in ni or
                              ni[d,query]['metrics']['tp']+ni[d,query]['metrics']['fp']>0 else np.nan
                              for d,value in zip(DATASETS,values)]
                return [np.nan if value is None else value for value in values]
            series = [(q,native_values(q,'median_s'),c) for q,c in [('Q1','#4f9d8b'),('Q2','#80cfff')]]
            error = {q:[[ni[d,q]['median_s']-ni[d,q]['q1_s'] if (d,q) in ni else np.nan for d in DATASETS],
                        [ni[d,q]['q3_s']-ni[d,q]['median_s'] if (d,q) in ni else np.nan for d in DATASETS]]
                     for q in ('Q1','Q2')}
            fig = grouped_figure(series,'Query Time (s)',log=True,error=error,
                note='SketchQL + Verifier | CPU, answer-seeded QBE; 3 runs, IQR bars')
            save_figure(fig,output,'baseline_sketchql_qbe_time_measured_20260920',pdf_book,outputs)
            series = [(f'{q} {metric.upper()}',[v*100 for v in native_values(q,metric)],color)
                      for q,colors in [('Q1',['#3b5da3','#81a2ed','#80cfff']),('Q2',['#4f9d8b','#73a57a','#ffc920'])]
                      for metric,color in zip(['precision','recall','f1'],colors)]
            fig = grouped_figure(series,'Interval P / R / F1 (%)',
                note='SketchQL + Verifier | Answer-seeded QBE; undefined Precision: N/A')
            save_figure(fig,output,'baseline_sketchql_qbe_accuracy_measured_20260920',pdf_book,outputs)
            series = [(f'{q} {label}',[v*100 for v in native_values(q,metric)],color)
                      for q,colors in [('Q1',['#3b5da3','#80cfff']),('Q2',['#4f9d8b','#ffc920'])]
                      for metric,label,color in zip(['frame_pruning','predicate_reduction'],['Frames','Predicates'],colors)]
            fig = grouped_figure(series,'Reduction (%)',
                note='SketchQL + Verifier | Answer-seeded QBE; interpret with recall')
            save_figure(fig,output,'baseline_sketchql_qbe_pruning_measured_20260920',pdf_book,outputs)
    outputs.append(str(book))
    status,measurements = [],[]
    all_groups = {(g['dataset'],g['query_id'],g['method']):g for g in [*groups,*native]}
    for dataset,query in itertools.product(DATASETS,['Q1','Q2']):
        for method in [*METHODS,'SketchQL']:
            measured = method in METHODS or any(g['dataset']==dataset and g['query_id']==query for g in native)
            status.append(dict(dataset=dataset,query_id=query,method=method,
                method_variant='SketchQL-QBE proposals + QST verifier' if method=='SketchQL' else method,
                status='measured_strict_QST' if method in METHODS else 'measured_separate_QBE' if measured
                else 'not_measured_or_incomplete'))
            group = all_groups.get((dataset,query,method),{})
            row = dict(status[-1],protocol=group.get('protocol',''),
                       repetitions=len(group['raw']) if group else '',
                       oracle_intervals=group['metrics']['tp']+group['metrics']['fn'] if group else '')
            row.update({k:group.get(k,'') for k in ('median_s','q1_s','q3_s','frame_pruning','binding_frame_pruning',
                                                  'predicate_reduction','ordered_topk_verified','full_intervals_verified')})
            row.update({k:group.get('metrics',{}).get(k,'') for k in ('tp','fp','fn','precision','recall','f1')})
            row['precision_defined'] = group['metrics']['tp']+group['metrics']['fp']>0 if group else ''
            row['recall_defined'] = group['metrics']['tp']+group['metrics']['fn']>0 if group else ''
            # Keep the source convention auditable, but do not export undefined
            # precision as a seemingly valid 100% result in the plotting table.
            row['precision_raw'] = row['precision']
            if group and not row['precision_defined']:
                row['precision'] = ''
            if group and not row['recall_defined']:
                row['recall'] = ''
            row.update({k:group.get('audit',{}).get(k,'') for k in ('candidate_s','verification_s',
                'candidate_frames','binding_frame_admitted','spatial_predicate_evaluations',
                'windows_total','windows_after_type_filter')})
            measurements.append(row)
    with (output/'baseline_full_suite_method_status_20260920.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(status[0])); writer.writeheader(); writer.writerows(status)
    with (output/'baseline_full_suite_results_20260920.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(measurements[0])); writer.writeheader(); writer.writerows(measurements)
    setup_rows = []
    for dataset,method in itertools.product(DATASETS,METHODS):
        metadata = json.loads((Path(source)/dataset/'artifacts'/f'{method}.metadata.json').read_text(encoding='utf-8'))
        setup_rows.append(dict(dataset=dataset,method=method,setup_kind=metadata['setup_kind'],
                               setup_s=metadata['setup_s'],index_payload_bytes=metadata['payload_bytes']))
    with (output/'baseline_full_suite_setup_20260920.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(setup_rows[0])); writer.writeheader(); writer.writerows(setup_rows)
    (output/'baseline_full_suite_plot_manifest_20260920.json').write_text(json.dumps(dict(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        figure_set=figure_set,individual_plot_count=(len(outputs)-1)//2,
        strict_groups=len(groups),native_groups=len(native),source_files=provenance,outputs=outputs,
        figure_size_inches=[6.9,3.9],png_dpi=300,font_family='Times New Roman',method_colors=COLORS,
        accuracy_scope='Exact structured interval agreement with scan; not visual-semantic accuracy.',
        missing_groups=[r for r in status if r['status']=='not_measured_or_incomplete'],
        style_reference='supplement/render_baseline_structured_query_figures.py: baseline_num2_query_time_full; '
                        'D:/pycharm/实验图代码/baseline_query_time_final.py: multi-method layout',
        measurement_note='No embedded timing arrays, scaling, estimates or manual overrides.'),indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(dict(strict_groups=len(groups),native_groups=len(native),files=len(outputs),book=str(book)),ensure_ascii=False))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--native-source',type=Path)
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--figure-set',choices=('paper','all'),default='paper',
                        help='paper keeps the 8 time/pruning plots selected by the user; all includes diagnostics')
    args = parser.parse_args()
    render(args.source,args.output_dir,args.native_source,args.figure_set)
