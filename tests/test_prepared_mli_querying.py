import importlib.util
import math
import unittest

import igraph as ig
import pandas as pd

from supplement.corrected_query_model import (QuerySpec, RelationConstraint, exact_query,
    ordered_result_signature, quantize_relation)


def engine(data, windows=None, missing=(), graph_only=False):
    spec = importlib.util.find_spec('vsimsearch.prepared_mli_querying')
    assert spec is not None, 'PreparedMLIQueryEngine is not implemented'
    from vsimsearch.prepared_mli_querying import PreparedMLIQueryEngine
    windows = windows or {1: set(data.frame)}
    types, graphs, edges = {}, {}, {}
    for wid, frames in windows.items():
        part = data[data.frame.isin(frames)]
        types[wid] = {int(c): set(map(int, g.track_id)) for c,g in part.groupby('class_id')}
        pairs = {}
        for frame, group in part.groupby('frame'):
            rows = sorted(group.itertuples(index=False), key=lambda r:(r.cx,r.cy,r.track_id))
            for i,a in enumerate(rows):
                for b in rows[i+1:]:
                    if (int(frame),a.track_id,b.track_id) in missing:
                        continue
                    td = quantize_relation((a.cx,a.cy),(b.cx,b.cy),100,100)
                    pairs.setdefault((a.track_id,b.track_id),{}).setdefault(td,set()).add(int(frame))
        g = ig.Graph(directed=True)
        g.add_vertices(1)
        g.vs[0]['name'] = tuple(part.track_id.unique())
        g.vs[0]['combinations'] = pairs if graph_only else {}
        graphs[wid] = g
        edges[wid] = pairs
    return PreparedMLIQueryEngine(types,graphs,data,100,100,
        window_edge_indexes=None if graph_only else edges, legacy_single_direction=True)


def query(types=(0,1), relations=None, df=1, k=10):
    return QuerySpec('test', types, relations or {(0,1):RelationConstraint(0,7,frozenset(range(-10,11)))},df,k)


class PreparedMLITests(unittest.TestCase):
    def data(self, rows):
        return pd.DataFrame(rows,columns=['frame','track_id','class_id','cx','cy'])

    def assert_oracle(self, data, q, **kwargs):
        e = engine(data, **kwargs)
        actual = e.query(q,return_all=True)
        expected = exact_query(data,q,100,100,return_all=True)
        self.assertEqual(ordered_result_signature(actual),ordered_result_signature(expected))
        self.assertEqual(e.query(q),expected[:q.topk])
        return e

    def test_cross_window_continuity_and_missing_frame(self):
        d=self.data([(f,i,c,x,20.) for f in (1,2,3,5,6) for i,c,x in ((10,0,10.),(20,1,60.))])
        self.assert_oracle(d,query(df=2),windows={1:{1,2},2:{3,5,6}})

    def test_changing_class_is_checked_per_frame(self):
        d=self.data([(f,i,c,x,20.) for f in range(1,6) for i,c,x in ((10,0 if f!=3 else 2,10.),(20,1,60.))])
        self.assert_oracle(d,query(df=2))

    def test_three_roles_require_all_relations_in_same_frame(self):
        d=self.data([(f,i,c,x,y) for f in range(1,6) for i,c,x,y in
            ((10,0,10.,10.),(20,1,40. if f<4 else 90.,10.),(30,2,40.,40. if f>2 else 90.))])
        q=query((0,1,2),{(0,1):RelationConstraint(1,1,frozenset({0})),
                         (1,2):RelationConstraint(1,1,frozenset({5}))})
        self.assert_oracle(d,q)

    def test_reverse_horizontal_and_zero_distance(self):
        d=self.data([(f,i,c,x,y) for f in (1,2) for i,c,x,y in
            ((10,0,20.,20.),(20,1,20.,20.),(30,2,70.,20.))])
        self.assert_oracle(d,query((1,0),{(0,1):RelationConstraint(0,0,frozenset({0}))}))
        self.assert_oracle(d,query((2,0),{(0,1):RelationConstraint(2,2,frozenset({10}))}))

    def test_angle_quantization_boundaries(self):
        for angle in (math.pi/10,-math.pi/10,math.pi/2,-math.pi/2,math.pi-1e-12):
            d=self.data([(1,10,0,50.,50.),(1,20,1,50.+20*math.cos(angle),50.+20*math.sin(angle))])
            td=quantize_relation((d.iloc[1].cx,d.iloc[1].cy),(50.,50.),100,100)
            self.assert_oracle(d,query((1,0),{(0,1):RelationConstraint(td[1],td[1],frozenset({td[0]}))}))

    def test_missing_sidecar_pair_frame_falls_back(self):
        d=self.data([(f,i,c,x,20.) for f in (1,2,3) for i,c,x in ((10,0,10.),(20,1,60.))])
        e=self.assert_oracle(d,query(df=3),missing={(2,10,20)})
        self.assertIn(2,e.metadata['fallback_frames'])

    def test_graph_only_uses_graph_relations(self):
        d=self.data([(f,i,c,x,20.) for f in (1,2) for i,c,x in ((10,0,10.),(20,1,60.))])
        e=self.assert_oracle(d,query(),graph_only=True)
        self.assertGreater(e.metadata['graph_only_windows'],0)

    def test_three_roles_no_cartesian_fallback_when_coverage_complete(self):
        d=self.data([(f,i,c,x,20.) for f in range(1,5) for i,c,x in ((10,0,10.),(20,1,40.),(30,2,70.))])
        e=self.assert_oracle(d,query((0,1,2),{(0,1):RelationConstraint(1,1,frozenset({0})),
            (1,2):RelationConstraint(1,1,frozenset({0}))}))
        self.assertEqual(e.metadata['fallback_frames'],[])

    def test_empty_constraints_and_single_role(self):
        d=self.data([(f,i,c,x,20.) for f in (1,2,4) for i,c,x in ((10,0,10.),(20,1,60.))])
        q=QuerySpec('empty',(0,),{},2,1)
        self.assert_oracle(d,q)

    def test_repeated_types_distinct_bindings_and_stable_topk(self):
        d=self.data([(f,i,0,x,20.) for f in (1,2,3) for i,x in ((10,10.),(20,40.),(30,70.))])
        self.assert_oracle(d,query((0,0),k=2))

    def test_random_small_inputs_match_exact_scan(self):
        import random
        rng=random.Random(73)
        for n in (2,3,4):
            rows=[]
            for frame in range(1,9):
                for i in range(5):
                    if rng.random()>.2:
                        rows.append((frame,i,i%3,float(rng.randrange(100)),float(rng.randrange(100))))
            d=self.data(rows)
            roles=tuple(i%3 for i in range(n))
            relations={(i,i+1):RelationConstraint(0,2,frozenset(range(-7,7))) for i in range(n-1)}
            self.assert_oracle(d,query(roles,relations,df=1),windows={1:{1,2,3,4},2:{5,6,7,8}})

    def test_nonlegacy_partial_direction_uses_explicit_fallback(self):
        from vsimsearch.prepared_mli_querying import PreparedMLIQueryEngine
        d=self.data([(1,10,0,10.,20.),(1,20,1,60.,20.)])
        e=PreparedMLIQueryEngine({1:{0:{10},1:{20}}},{},d,100,100,
            window_edge_indexes={1:{(10,20):{(0,2):{1}}}},legacy_single_direction=False)
        q=query((1,0),{(0,1):RelationConstraint(2,2,frozenset({10}))})
        self.assertEqual(e.query(q),exact_query(d,q,100,100))

    def test_overlapping_window_counts_cannot_hide_missing_pair(self):
        from vsimsearch.prepared_mli_querying import PreparedMLIQueryEngine
        d=self.data([(1,10,0,10.,20.),(1,20,1,40.,20.),(1,30,2,70.,20.)])
        e=PreparedMLIQueryEngine({}, {},d,100,100,window_edge_indexes={
            1:{(10,20):{(0,1):{1}},(20,30):{(0,1):{1}}},
            2:{(10,20):{(0,1):{1}}}},legacy_single_direction=True)
        q=query((0,2),{(0,1):RelationConstraint(2,2,frozenset({0}))})
        self.assertEqual(e.query(q),exact_query(d,q,100,100))

    def test_audited_bucket_boundaries_do_not_admit_every_neighbor_bucket(self):
        d=self.data([(1,10,0,10.,20.),(1,20,1,60.,20.),(1,30,1,90.,20.)])
        e=engine(d)
        audit={}
        q=query((0,1),{(0,1):RelationConstraint(2,2,frozenset({0}))})
        self.assertEqual(e.query(q,stats=audit),exact_query(d,q,100,100))
        self.assertEqual(audit['binding_frame_enumerated'],1)

    def test_graph_entries_supplement_incomplete_sidecar_without_mutation(self):
        from vsimsearch.prepared_mli_querying import PreparedMLIQueryEngine
        d=self.data([(f,i,c,x,20.) for f in (1,2) for i,c,x in ((10,0,10.),(20,1,60.))])
        graph=ig.Graph(directed=True)
        graph.add_vertices(1)
        graph.vs[0]['combinations']={(10,20):{(0,2):{1,2}}}
        sidecar={1:{(10,20):{(0,2):{1}}}}
        e=PreparedMLIQueryEngine({}, {1:graph}, d,100,100,
            window_edge_indexes=sidecar,legacy_single_direction=True)
        q=query(df=2)
        self.assertEqual(e.query(q),exact_query(d,q,100,100))
        self.assertEqual(e.metadata['graph_sidecar_supplement_frames'],1)
        self.assertEqual(sidecar[1][(10,20)][(0,2)],{1})
        self.assertEqual(e.metadata['fallback_frames'],[])

    def test_wrong_bins_are_explicitly_reported_and_fall_back(self):
        from vsimsearch.prepared_mli_querying import PreparedMLIQueryEngine
        d=self.data([(1,10,0,10.,20.),(1,20,1,60.,20.)])
        e=PreparedMLIQueryEngine({}, {},d,100,100,
            window_edge_indexes={1:{(10,20):{(7,9):{1}}}},legacy_single_direction=True)
        self.assertEqual(e.query(query()),exact_query(d,query(),100,100))
        self.assertEqual(e.metadata['fallback_frames'],[1])

    def test_indexed_dimensions_can_differ_from_query_dimensions(self):
        from vsimsearch.prepared_mli_querying import PreparedMLIQueryEngine
        d=self.data([(1,10,0,10.,20.),(1,20,1,60.,20.)])
        e=PreparedMLIQueryEngine({}, {},d,200,200,
            window_edge_indexes={1:{(10,20):{(0,2):{1}}}},legacy_single_direction=True,
            indexed_frame_width=100,indexed_frame_height=100)
        q=query(relations={(0,1):RelationConstraint(1,1,frozenset({0}))})
        self.assertEqual(e.query(q),exact_query(d,q,200,200))
        self.assertEqual(e.metadata['fallback_frames'],[])

    def test_incompatible_quantization_is_rejected(self):
        d=self.data([(1,10,0,10.,20.),(1,20,1,60.,20.)])
        e=engine(d)
        q=QuerySpec('wrong-parts',(0,1),{(0,1):RelationConstraint(0,7,frozenset({0}))},1,10,8,8)
        with self.assertRaisesRegex(ValueError,'quantization'):
            e.query(q)

    def test_duplicate_track_uses_all_classes_and_last_coordinate_like_oracle(self):
        from vsimsearch.prepared_mli_querying import PreparedMLIQueryEngine
        d=self.data([(1,-1,0,10.,20.),(1,-1,2,30.,20.),(1,20,1,60.,20.)])
        e=PreparedMLIQueryEngine({}, {},d,100,100,
            window_edge_indexes={1:{(-1,20):{(0,2):{1}}}},legacy_single_direction=True)
        for role_type in (0,2):
            q=query((role_type,1),{(0,1):RelationConstraint(1,1,frozenset({0}))})
            expected=exact_query(d,q,100,100)
            self.assertTrue(expected)
            self.assertEqual(e.query(q),expected)
        self.assertEqual(e.metadata['duplicate_observation_rows'],1)
        self.assertEqual(e.metadata['fallback_frames'],[1])

    def test_preparation_reports_additional_serialized_structure_size(self):
        d=self.data([(1,10,0,10.,20.),(1,20,1,60.,20.)])
        e=engine(d)
        self.assertGreater(e.metadata['prepared_auxiliary_pickle_bytes'],0)
        self.assertGreaterEqual(e.metadata['setup_s'],e.metadata['size_measurement_s'])


if __name__=='__main__':
    unittest.main()
