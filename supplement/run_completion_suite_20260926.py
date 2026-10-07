"""Verified supplementary runs for prepared Ours, EVA-MV and EQUI-VOCAL.

Existing unaffected timings remain separate immutable evidence. Only independent
preparation is outside query timing; no workload answers are materialized here.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import pickle
import sys
import time

from paper_query_corrected import corrected_mli_tag
from supplement.corrected_query_model import QueryResult, ordered_result_signature
from supplement.dataset_config import dataset_frame_size, dataset_storage_path, PREFERRED_VIDEO_ORDER
from supplement.generate_corrected_baseline_workload import load_tracking_table, query_from_dict
from supplement.run_baseline_audit import sha256
from supplement.run_full_baseline_suite import measure_group, write_json, now
from vsimsearch.mul_build_index import artifact_paths

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'supplement/out/full_baseline_suite_20260920'
WORKLOAD=ROOT/'supplement/out/corrected_baseline_20260712/workload.json'
OUTPUT=ROOT/'supplement/out/completion_suite_20260926'
METHODS=('Ours','EVA-MV adapted','EQUI-VOCAL SQL adapted')
COMMON=('supplement/corrected_query_model.py','supplement/generate_corrected_baseline_workload.py',
        'supplement/run_full_baseline_suite.py','supplement/run_completion_suite_20260926.py')
METHOD_SOURCES={
    'Ours':('vsimsearch/prepared_mli_querying.py','vsimsearch/corrected_mli_querying.py'),
    'EVA-MV adapted':('supplement/eva_symbolic_mv_adapter.py',),
    'EQUI-VOCAL SQL adapted':('supplement/equivocal_postgres_adapter.py',),
}


class PreparedOurs:
    def __init__(self,dataset,data,width,height):
        from vsimsearch.prepared_mli_querying import PreparedMLIQueryEngine
        started=time.perf_counter()
        paths=artifact_paths(dataset,.7,output_tag=corrected_mli_tag())[:3]
        with paths[0].open('rb') as f: index=pickle.load(f)
        with paths[1].open('rb') as f: graphs=pickle.load(f)
        with paths[2].open('rb') as f: edges=pickle.load(f)
        loaded=time.perf_counter()
        self.engine=PreparedMLIQueryEngine(index,graphs,data,width,height,
            window_edge_indexes=edges,legacy_single_direction=True,
            indexed_frame_width=width,indexed_frame_height=height)
        self.metadata=dict(self.engine.metadata,index_load_seconds=loaded-started,
            prebuilt_artifact_bytes=sum(p.stat().st_size for p in paths),
            setup_seconds=time.perf_counter()-started,
            protocol='original MLI graph/window edges; query-independent access preparation; raw exact candidate verification')

    def query(self,*args,**kwargs): return self.engine.query(*args,**kwargs)
    def release(self): self.engine.release()


def strategy_for(method,dataset,data,width,height):
    if method=='Ours': return PreparedOurs(dataset,data,width,height)
    if method=='EVA-MV adapted':
        from supplement.eva_symbolic_mv_adapter import EVASymbolicMVStrategy
        return EVASymbolicMVStrategy(data,width,height)
    if method=='EQUI-VOCAL SQL adapted':
        from supplement.equivocal_postgres_adapter import EquivocalPostgresStrategy
        return EquivocalPostgresStrategy(data,width,height)
    raise ValueError(method)


def source_hashes(method):
    return {name:sha256(ROOT/name) for name in (*COMMON,*METHOD_SOURCES[method])}


def run_dataset(dataset,methods,output,repetitions,warmups):
    destination=output/dataset
    destination.mkdir(parents=True,exist_ok=True)
    old_manifest=json.loads((OLD/dataset/'manifest.json').read_text(encoding='utf-8'))
    if not old_manifest['complete'] or not old_manifest['all_correct']:
        raise ValueError('unverified old oracle run')
    data_path=dataset_storage_path(dataset)
    data_hash=sha256(data_path)
    if data_hash!=old_manifest['data_sha256'] or sha256(WORKLOAD)!=old_manifest['workload_sha256']:
        raise ValueError('cannot reuse oracle for changed data/workload')
    width,height=dataset_frame_size(dataset)
    data=load_tracking_table(data_path)
    records=[r for r in json.loads(WORKLOAD.read_text(encoding='utf-8')) if r['dataset']==dataset]
    queries=[query_from_dict({k:v for k,v in r.items() if k!='source_event'}) for r in records]
    if sorted(q.query_id for q in queries)!=['Q1','Q2']: raise ValueError('Q1/Q2 required')
    oracles={}
    for query in queries:
        path=OLD/dataset/f'oracle_{query.query_id}.json'
        saved=json.loads(path.read_text(encoding='utf-8'))
        full=[QueryResult(tuple(r['binding']),r['start_frame'],r['end_frame']) for r in saved['results']]
        if ordered_result_signature(full)!=saved['signature']: raise ValueError('oracle signature mismatch')
        oracles[query.query_id]=(full,saved['audit'],sha256(path))
    for method in methods:
        hashes=source_hashes(method)
        identity=dict(dataset=dataset,method=method,data_sha256=data_hash,workload_sha256=sha256(WORKLOAD),
            source_sha256=hashes,repetitions=repetitions,warmups=warmups,
            index_sha256=({str(p):sha256(p) for p in artifact_paths(dataset,.7,output_tag=corrected_mli_tag())[:3]} if method=='Ours' else {}),
            oracle_sha256={qid:row[2] for qid,row in oracles.items()})
        manifest_path=destination/f'manifest_{method}.json'
        paths={q.query_id:destination/'groups'/f'{method}_{q.query_id}.json' for q in queries}
        if manifest_path.exists():
            previous=json.loads(manifest_path.read_text(encoding='utf-8'))
            if previous['identity']!=identity: raise ValueError(f'{method}: checkpoint identity changed; choose fresh output')
            if previous.get('complete') and all(p.exists() for p in paths.values()):
                print(f'{dataset} {method}: verified checkpoint complete',flush=True)
                continue
        manifest=dict(identity=identity,started_utc=now(),complete=False,
            protocol='strict_QST_continuous_fixed_binding',
            timing_scope='preloaded online query including all needed predicates, temporal merging and ordered return; preparation excluded',
            frame_dimensions=[width,height],rows=len(data),unique_frames=int(data.frame.nunique()))
        write_json(manifest_path,manifest)
        strategy=None
        try:
            started=time.perf_counter()
            print(f'{dataset} {method}: setup',flush=True)
            strategy=strategy_for(method,dataset,data,width,height)
            setup_s=time.perf_counter()-started
            print(f'{dataset} {method}: setup {setup_s:.3f}s',flush=True)
            for query in queries:
                oracle,scan_audit,_=oracles[query.query_id]
                def progress(raw):
                    write_json(destination/'progress.json',dict(dataset=dataset,method=method,query_id=query.query_id,raw=raw,updated_utc=now()))
                    print(f'{dataset} {method} {query.query_id} rep{len(raw)} {raw[-1]["elapsed_s"]*1000:.3f}ms correct={raw[-1]["ordered_topk_verified"]}',flush=True)
                result=measure_group(strategy,query,oracle,repetitions=repetitions,warmups=warmups,progress=progress)
                if not result['ordered_topk_verified'] or not result['full_intervals_verified']:
                    write_json(destination/f'FAILED_{method}_{query.query_id}.json',result)
                    raise ValueError(f'{method}/{query.query_id}: exact result mismatch')
                audit=result['audit']
                result.update(dataset=dataset,method=method,query_id=query.query_id,
                    protocol=manifest['protocol'],source='measured_full_dataset',setup_s=setup_s,
                    setup_kind='query_independent_preparation',completed_utc=now(),
                    frame_pruning=(1-audit['candidate_frames']/manifest['unique_frames'] if 'candidate_frames' in audit else None),
                    binding_frame_pruning=(1-audit['binding_frame_admitted']/scan_audit['binding_frame_enumerated']
                        if 'binding_frame_admitted' in audit and scan_audit['binding_frame_enumerated'] else None),
                    predicate_reduction=(1-audit['spatial_predicate_evaluations']/scan_audit['spatial_predicate_evaluations']
                        if 'spatial_predicate_evaluations' in audit and scan_audit['spatial_predicate_evaluations'] else None),
                    pruning_applicable=method=='Ours',
                    execution_correctness_not_semantic_accuracy=True)
                write_json(paths[query.query_id],result)
            manifest['metadata']=getattr(strategy,'metadata',getattr(strategy,'artifact',{}))
            manifest['setup_seconds']=setup_s
            manifest['integrity']=source_hashes(method)==hashes and sha256(data_path)==data_hash
            manifest['complete']=manifest['integrity']
            if not manifest['integrity']: raise ValueError('source/data changed during measurement')
        except Exception as error:
            import traceback
            manifest.update(error=repr(error),traceback=traceback.format_exc())
            raise
        finally:
            if strategy is not None: strategy.release()
            manifest['finished_utc']=now()
            write_json(manifest_path,manifest)
            del strategy
            gc.collect()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--datasets',nargs='+',choices=list(PREFERRED_VIDEO_ORDER),default=list(PREFERRED_VIDEO_ORDER))
    p.add_argument('--methods',nargs='+',choices=list(METHODS),default=list(METHODS))
    p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--repetitions',type=int,default=3)
    p.add_argument('--warmups',type=int,default=1)
    args=p.parse_args()
    if args.repetitions<1 or args.warmups<0: p.error('invalid repetitions/warmups')
    for dataset in args.datasets: run_dataset(dataset,args.methods,args.output,args.repetitions,args.warmups)
