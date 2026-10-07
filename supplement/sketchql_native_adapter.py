"""Full-input native SketchQL QBE proposal search plus exact QST verification.

This is a separate, answer-seeded example-query protocol. Learned scores are
computed by the native model, but no tuned score threshold is imposed: every
native proposal is verified. Do not label this identical-input structured QST.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import pickle
import random
import statistics
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
NATIVE = Path('D:/pycharm/SketchQL-main')
DATASETS = ('drtest','drtrain','bdd100kA','bdd100kB','I-24')
NATIVE_SOURCES = ('search/search.py','model/model.py','data/utils.py',
                  'data/video_data.py','data/visual_query.py')
PROTOCOL = 'answer_seeded_QBE_native_all_proposals_plus_QST_verifier'
VARIANT = 'SketchQL-QBE proposals + QST verifier'


def validate_records(records):
    if sorted(r['query_id'] for r in records) != ['Q1','Q2']:
        raise ValueError('Native workload requires exactly one Q1 and one Q2')


def native_group_valid(directory,dataset,query_id,repetitions):
    try:
        group = json.loads((Path(directory)/f'{query_id}.json').read_text(encoding='utf-8'))
        proposals = json.loads((Path(directory)/f'{query_id}.proposals.json').read_text(encoding='utf-8'))
        return ((group['dataset'],group['query_id'],group['protocol'],group['method_variant']) ==
                (dataset,query_id,PROTOCOL,VARIANT) and len(group['raw'])==repetitions
                and all(row['elapsed_s']>0 for row in group['raw'])
                and all(k in group['audit'] for k in ('candidate_frames','spatial_predicate_evaluations'))
                and all(k in group['metrics'] for k in ('precision','recall','f1'))
                and isinstance(group['full_results'],list) and isinstance(proposals['masks'],list)
                and isinstance(proposals['proposals'],list))
    except (OSError,ValueError,KeyError,TypeError):
        return False


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
    tmp.replace(path)


def proposal_masks(records, role_types):
    grouped_roles = defaultdict(list)
    for role, label in enumerate(role_types):
        grouped_roles[label].append(role)
    native_order = [role for roles in grouped_roles.values() for role in roles]
    masks = defaultdict(set)
    for proposal in records:
        if len(proposal) != len(role_types):
            raise ValueError('Native proposal role count mismatch')
        windows = {tuple(obj['frame_start_end']) for obj in proposal}
        if len(windows) != 1:
            raise ValueError('Native proposal has inconsistent object windows')
        start,end = next(iter(windows))
        if start < 0 or end <= start:
            raise ValueError('Invalid native half-open interval')
        binding = [None]*len(role_types)
        for role,obj in zip(native_order,proposal):
            binding[role] = int(obj['video_obj_id'])
        if len(set(binding)) != len(binding):
            continue
        masks[tuple(binding)].update(range(int(start)+1,int(end)+1))
    return dict(masks)


def metrics(predicted, truth):
    key = lambda r:(r.binding,r.start_frame,r.end_frame)
    p,t = {key(r) for r in predicted},{key(r) for r in truth}
    tp,fp,fn = len(p&t),len(p-t),len(t-p)
    return dict(tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else 1.,
                recall=tp/(tp+fn) if tp+fn else 1.,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.)


def prepare_input(dataset, destination, records):
    import pandas as pd
    import data.video_data as vd
    from supplement.dataset_config import dataset_frame_size, dataset_storage_path
    from supplement.generate_corrected_baseline_workload import RAW_COLUMNS
    raw = pd.read_csv(dataset_storage_path(dataset),names=RAW_COLUMNS)
    raw = raw.sort_values(['frame','track_id'],kind='stable')
    maximum = int(raw.frame.max())
    primitives = [dict(frame_id=f,labels=[],ids=[],bboxes=[]) for f in range(1,maximum+1)]
    for row in raw.itertuples(index=False):
        frame = primitives[int(row.frame)-1]
        frame['labels'].append(int(row.class_id))
        frame['ids'].append(int(row.track_id))
        frame['bboxes'].append((float(row.left),float(row.top),float(row.left+row.width),float(row.top+row.height)))
    path = destination/'full_primitives.pkl'
    with path.open('wb') as handle:
        pickle.dump(primitives,handle,pickle.HIGHEST_PROTOCOL)
    # Numeric labels are equality tokens. Do not map BDD's ids through COCO names.
    labels = {int(c):f'class_{int(c)}' for c in raw.class_id.unique()}
    width,height = dataset_frame_size(dataset)
    with patch.object(vd,'CLASSES',labels):
        objects,at_frame = vd.read_primitives_old(str(path),width,height)
    query_objects = {}
    for record in records:
        event = record['source_event']
        start,end = int(event['start_frame']),int(event['end_frame'])
        queries = {}
        for role,(track_id,label) in enumerate(zip(event['binding'],record['role_types'])):
            obj = vd.ObjInVideo(f'role{role}')
            obj.frame_start,obj.frame_end = 0,end-start+1 # VisualQuery uses exclusive end.
            for f in range(start,end+1):
                frame = primitives[f-1]
                matches = [b for i,b in zip(frame['ids'],frame['bboxes']) if i==track_id]
                if not matches:
                    raise ValueError(f'Missing exemplar role {track_id} at frame {f}')
                obj.boxes.append(matches[-1])
            obj.types = [f'class_{int(label)}']*len(obj.boxes)
            queries[obj.id] = obj
        query_objects[record['query_id']] = queries
    data = pd.DataFrame(dict(frame=raw.frame.astype(int),track_id=raw.track_id.astype(int),
        class_id=raw.class_id.astype(int),cx=raw.left+raw.width/2,cy=raw.top+raw.height/2))
    return data,objects,at_frame,query_objects


def run_dataset(dataset, output, strict_output, workload, repetitions=3, warmups=1, threads=4):
    import numpy as np
    import pandas as pd
    import torch
    from supplement.corrected_query_model import QueryResult, exact_query, ordered_result_signature
    from supplement.dataset_config import dataset_storage_path,dataset_frame_size
    from supplement.generate_corrected_baseline_workload import query_from_dict
    if str(NATIVE) not in sys.path:
        sys.path.insert(0,str(NATIVE))
    import search.search as native_search
    from model.model import EncoderCentroidLarger, EncoderModelWrapper
    from data.visual_query import VisualQuery
    checkpoint = NATIVE/'data/model_checkpoint/model_cp.pt'
    destination = Path(output)/dataset
    destination.mkdir(parents=True,exist_ok=True)
    records = [r for r in json.loads(Path(workload).read_text(encoding='utf-8')) if r['dataset']==dataset]
    validate_records(records)
    sources = [Path(__file__),ROOT/'supplement/corrected_query_model.py',ROOT/'supplement/generate_corrected_baseline_workload.py',
               ROOT/'supplement/dataset_config.py',*[NATIVE/name for name in NATIVE_SOURCES]]
    identity = dict(data_sha256=sha256(dataset_storage_path(dataset)),workload_sha256=sha256(workload),
        model_sha256=sha256(checkpoint),source_sha256={str(p):sha256(p) for p in sources},
        repetitions=repetitions,warmups=warmups,threads=threads,
        oracle_sha256={r['query_id']:sha256(Path(strict_output)/dataset/f'oracle_{r["query_id"]}.json') for r in records})
    reference_manifest = json.loads((Path(strict_output)/dataset/'manifest.json').read_text(encoding='utf-8'))
    if (not reference_manifest.get('complete')
        or not all(reference_manifest.get('integrity',{}).get(k) is True for k in ('data','source','native','index'))
        or any(reference_manifest.get(k)!=identity[k] for k in ('data_sha256','workload_sha256'))
        or reference_manifest['source_sha256']['supplement/corrected_query_model.py'] != sha256(ROOT/'supplement/corrected_query_model.py')):
        raise ValueError('Native run requires a complete, matching strict reference experiment')
    manifest_path = destination/'manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        if previous['identity'] != identity or previous.get('integrity') is False:
            raise ValueError('Native checkpoint identity/integrity mismatch; use a fresh output directory')
        if previous.get('complete') and all(native_group_valid(destination,dataset,r['query_id'],repetitions) for r in records):
            return
    elif any(destination.glob('Q*.json')):
        raise ValueError('Native orphan group files without manifest')
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    random.seed(20260920); np.random.seed(20260920); torch.manual_seed(20260920)
    manifest = dict(identity=identity,dataset=dataset,complete=False,protocol=PROTOCOL,
        method='SketchQL',method_variant=VARIANT,python=sys.version,torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,
        device='cpu',native_inference_batch=64,query_seed='workload.source_event; additional answer-derived exemplar input',
        filtering='native stride=10, scales=[0.5,0.8,1,1.2,1.5,2], motion and overlap filters; no score threshold',
        candidate_scope='all scored native proposals; learned scores recorded but not thresholded',
        timing_scope='preloaded model/tracks; native search + proposal mapping + exact verification; fresh track copy before each run',
        input_scope='all original sampled-frame rows; native reader boundary removal, interpolation and majority-track class retained')
    save(manifest_path,manifest)
    started = time.perf_counter()
    data,objects,at_frame,query_objects = prepare_input(dataset,destination,records)
    manifest['input_prepare_s'] = time.perf_counter()-started
    manifest.update(rows=len(data),unique_frames=int(data.frame.nunique()),last_frame=int(data.frame.max()),
                    native_track_count=len(objects),native_frame_slots=len(at_frame))
    started = time.perf_counter()
    model = EncoderModelWrapper(str(checkpoint),EncoderCentroidLarger(),torch.device('cpu'),False,n_att=4)
    original_predict = model.predict_embed
    model.predict_embed = lambda seq,batch_size=64:original_predict(seq,batch_size=64)
    manifest['model_load_s'] = time.perf_counter()-started
    save(manifest_path,manifest)
    width,height = dataset_frame_size(dataset)
    for record in records:
        query_id = record['query_id']
        group_path = destination/f'{query_id}.json'
        if native_group_valid(destination,dataset,query_id,repetitions):
            continue
        try:
            query = query_from_dict({k:v for k,v in record.items() if k!='source_event'})
            saved = json.loads((Path(strict_output)/dataset/f'oracle_{query_id}.json').read_text(encoding='utf-8'))
            oracle = [QueryResult(tuple(r['binding']),r['start_frame'],r['end_frame']) for r in saved['results']]
            native_query = VisualQuery(query_objects[query_id],relation=[])
            raw = []
            for iteration in range(warmups+repetitions+1):
                audit = iteration==warmups+repetitions
                # Native search smooths boxes in place; each trial gets the same fresh input.
                tracks = copy.deepcopy(objects)
                random.seed(20260920); np.random.seed(20260920); torch.manual_seed(20260920)
                stats = {} if audit else None
                started = time.perf_counter()
                with torch.inference_mode(), patch.object(native_search,'EncoderCentroidLarger',lambda:None), \
                     patch.object(native_search,'EncoderModelWrapper',lambda **kwargs:model):
                    proposals,features,scores,centroids,timing = native_search.execute_qeury_multi_obj_sim_search_learned_model(
                        tracks,at_frame,None,native_query,str(checkpoint),torch.device('cpu'))
                if len(scores) != len(proposals) or not np.isfinite(scores).all():
                    raise ValueError('Native model returned missing or non-finite proposal scores')
                # These auxiliary visualization tensors are not used by retrieval
                # verification. Keep every scored proposal, but release these copies.
                del features,centroids
                candidate_finished = time.perf_counter()
                masks = proposal_masks(proposals,query.role_types)
                mapping_finished = time.perf_counter()
                results = exact_query(data,query,width,height,candidate_binding_frames=masks,stats=stats,return_all=audit)
                elapsed = time.perf_counter()-started
                print(f'{dataset} SketchQL {query_id}: iteration={iteration} audit={audit} {elapsed:.6f}s proposals={len(proposals)}',flush=True)
                if not audit and iteration>=warmups:
                    raw.append(dict(repetition=iteration-warmups+1,elapsed_s=elapsed,
                        native_search_s=candidate_finished-started,mapping_s=mapping_finished-candidate_finished,
                        result_signature=ordered_result_signature(results),native_proposals=len(proposals)))
                    save(destination/f'{query_id}.progress.json',dict(raw=raw))
                if audit:
                    stats.update(candidate_s=mapping_finished-started,native_search_s=candidate_finished-started,
                                 proposal_mapping_s=mapping_finished-candidate_finished)
                    # Store every scored proposal, not just the final verified intervals.
                    compact = [dict(binding=list(b),frames=sorted(frames)) for b,frames in masks.items()]
                    save(destination/f'{query_id}.proposals.json',dict(masks=compact,
                        proposals=[dict(objects=[int(o['video_obj_id']) for o in p],window=list(p[0]['frame_start_end']),
                                        score=float(s)) for p,s in zip(proposals,scores)]))
                    values = [r['elapsed_s'] for r in raw]
                    group = dict(dataset=dataset,query_id=query_id,method='SketchQL',method_variant=VARIANT,protocol=manifest['protocol'],
                        source='measured_full_dataset',median_s=statistics.median(values),q1_s=float(np.quantile(values,.25)),
                        q3_s=float(np.quantile(values,.75)),min_s=min(values),max_s=max(values),raw=raw,audit=stats,
                        metrics=metrics(results,oracle),full_results=[asdict(r) for r in results],native_timing=timing,
                        native_proposal_count=len(proposals),unique_proposal_bindings=len(masks),
                        ordered_topk_verified=all(r['result_signature']==ordered_result_signature(oracle[:query.topk]) for r in raw),
                        full_intervals_verified=ordered_result_signature(results)==ordered_result_signature(oracle),
                        frame_pruning=1-stats['candidate_frames']/manifest['unique_frames'],
                        predicate_reduction=1-stats['spatial_predicate_evaluations']/saved['audit']['spatial_predicate_evaluations']
                            if saved['audit']['spatial_predicate_evaluations'] else None)
                    save(group_path,group)
                del tracks,proposals,scores,masks,results
                gc.collect()
        except Exception as error:
            save(destination/f'failure_{query_id}.json',dict(error=repr(error),traceback=traceback.format_exc()))
            print(f'{dataset} SketchQL {query_id}: FAILED {error!r}',flush=True)
    manifest['integrity'] = (sha256(dataset_storage_path(dataset))==identity['data_sha256']
        and sha256(workload)==identity['workload_sha256'] and sha256(checkpoint)==identity['model_sha256']
        and all(sha256(p)==h for p,h in identity['source_sha256'].items())
        and all(sha256(Path(strict_output)/dataset/f'oracle_{q}.json')==h for q,h in identity['oracle_sha256'].items()))
    manifest['complete'] = manifest['integrity'] and all(native_group_valid(destination,dataset,r['query_id'],repetitions) for r in records)
    save(manifest_path,manifest)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=DATASETS,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--strict-output',type=Path,required=True)
    parser.add_argument('--workload',type=Path,default=ROOT/'supplement/out/corrected_baseline_20260712/workload.json')
    parser.add_argument('--repetitions',type=int,default=3)
    parser.add_argument('--warmups',type=int,default=1)
    parser.add_argument('--threads',type=int,default=4)
    args = parser.parse_args()
    run_dataset(args.dataset,args.output_dir,args.strict_output,args.workload,args.repetitions,args.warmups,args.threads)
