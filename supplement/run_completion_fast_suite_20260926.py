"""Final MLI type-directory runs, retaining unchanged EVA/EQUI measurements.

The v1 runner and all v1 evidence stay immutable. This runner selects the fast
engine explicitly and records both engine modules plus this configuration in
each new source identity. Reused baseline files are copied byte-for-byte only
after their source/data/workload/oracle identities have been verified.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pickle
import shutil
import time

from supplement import run_completion_suite_20260926 as base
from vsimsearch.prepared_mli_querying_fast import FastPreparedMLIQueryEngine

OUTPUT=base.ROOT/'supplement/out/completion_suite_20260926_fast'


class FinalPreparedOurs:
    def __init__(self,dataset,data,width,height):
        started=time.perf_counter()
        paths=base.artifact_paths(dataset,.7,output_tag=base.corrected_mli_tag())[:3]
        values=[]
        for path in paths:
            with path.open('rb') as stream: values.append(pickle.load(stream))
        loaded=time.perf_counter()
        self.engine=FastPreparedMLIQueryEngine(values[0],values[1],data,width,height,
            window_edge_indexes=values[2],legacy_single_direction=True,
            indexed_frame_width=width,indexed_frame_height=height)
        self.metadata=dict(self.engine.metadata,index_load_seconds=loaded-started,
            prebuilt_artifact_bytes=sum(p.stat().st_size for p in paths),setup_seconds=time.perf_counter()-started,
            protocol='MLI original window pair maps plus type directory; same-query bucket compilation; raw exact verification')

    def query(self,*args,**kwargs): return self.engine.query(*args,**kwargs)
    def release(self): self.engine.release()


def reuse_unchanged(dataset,output):
    records=[]
    for method in ('EVA-MV adapted','EQUI-VOCAL SQL adapted'):
        source=base.OUTPUT/dataset
        manifest_path=source/f'manifest_{method}.json'
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        identity=manifest['identity']
        if not manifest.get('complete') or not manifest.get('integrity'): raise ValueError('incomplete baseline source')
        if (base.sha256(base.dataset_storage_path(dataset))!=identity['data_sha256'] or
                base.sha256(base.WORKLOAD)!=identity['workload_sha256'] or
                base.source_hashes(method)!=identity['source_sha256']):
            raise ValueError('reused baseline inputs or code changed')
        for q in ('Q1','Q2'):
            if base.sha256(base.OLD/dataset/f'oracle_{q}.json')!=identity['oracle_sha256'][q]:
                raise ValueError('reused oracle changed')
        paths=[manifest_path,*[source/'groups'/f'{method}_{q}.json' for q in ('Q1','Q2')]]
        for path in paths:
            destination=output/dataset/path.relative_to(source)
            destination.parent.mkdir(parents=True,exist_ok=True)
            if destination.exists() and base.sha256(destination)!=base.sha256(path):
                raise ValueError('cannot replace nonidentical existing measurement')
            if not destination.exists(): shutil.copy2(path,destination)
            records.append(dict(source=str(path.resolve()),destination=str(destination.resolve()),sha256=base.sha256(path)))
    base.write_json(output/dataset/'unchanged_baseline_reuse.json',dict(files=records,protocol='byte-identical verified reuse; no new timing samples'))


def run(datasets,output):
    # Explicit strategy selection; no modification of either executor's source.
    base.PreparedOurs=FinalPreparedOurs
    base.METHOD_SOURCES=dict(base.METHOD_SOURCES)
    base.METHOD_SOURCES['Ours']=(*base.METHOD_SOURCES['Ours'],
        'vsimsearch/prepared_mli_querying_fast.py','supplement/run_completion_fast_suite_20260926.py')
    for dataset in datasets:
        reuse_unchanged(dataset,output)
        base.run_dataset(dataset,['Ours'],output,3,1)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--datasets',nargs='+',choices=list(base.PREFERRED_VIDEO_ORDER),default=list(base.PREFERRED_VIDEO_ORDER))
    p.add_argument('--output',type=Path,default=OUTPUT)
    args=p.parse_args();run(args.datasets,args.output)
