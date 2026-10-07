import importlib.util
import pickle
import unittest
from unittest.mock import patch

import pandas as pd

from supplement.corrected_query_model import QuerySpec, RelationConstraint, exact_query, ordered_result_signature
from tests.test_prepared_mli_querying import engine as make_original, query


def make_fast(data, **kwargs):
    assert importlib.util.find_spec('vsimsearch.prepared_mli_querying_fast') is not None, 'fast MLI variant not implemented'
    from vsimsearch.prepared_mli_querying_fast import FastPreparedMLIQueryEngine
    original = make_original(data, **kwargs)
    edges = {wid: values[0] for wid, values in original.windows.items()}
    before = pickle.dumps(edges)
    fast = FastPreparedMLIQueryEngine(original.index, {}, data, 100, 100,
        window_edge_indexes=edges, legacy_single_direction=True)
    return original, fast, edges, before


class FastPreparedMLITests(unittest.TestCase):
    def data(self, rows):
        return pd.DataFrame(rows, columns=['frame','track_id','class_id','cx','cy'])

    def assert_queries(self, data, queries, **kwargs):
        original, fast, edges, before = make_fast(data, **kwargs)
        for q in queries:
            expected = exact_query(data, q, 100, 100, return_all=True)
            self.assertEqual(ordered_result_signature(fast.query(q, return_all=True)), ordered_result_signature(expected))
            self.assertEqual(fast.query(q), expected[:q.topk])
            self.assertEqual(fast.query(q), original.query(q))
        self.assertEqual(pickle.dumps(edges), before)
        return fast

    def test_changing_classes_reverse_relations_and_cross_window_intervals(self):
        data = self.data([(f,i,c,x,20.) for f in (1,2,3,5,6) for i,c,x in
            ((10,0 if f != 3 else 2,10.),(20,1,40.),(30,0,70.))])
        self.assert_queries(data, [query((0,1),df=2), query((1,0),df=2),query((0,0),k=2)],
            windows={1:{1,2},2:{3,5,6}})

    def test_simultaneous_three_role_constraints(self):
        data = self.data([(f,i,c,x,y) for f in range(1,6) for i,c,x,y in
            ((10,0,10.,10.),(20,1,40. if f<4 else 90.,10.),(30,2,40.,40. if f>2 else 90.))])
        self.assert_queries(data, [query((0,1,2), {
            (0,1):RelationConstraint(1,1,frozenset({0})),
            (1,2):RelationConstraint(1,1,frozenset({5}))})])

    def test_duplicate_id_and_missing_pair_fallback_preserve_oracle(self):
        data = self.data([(1,-1,0,10.,20.),(1,-1,2,30.,20.),(1,20,1,60.,20.),
                          (2,10,0,10.,20.),(2,20,1,60.,20.)])
        self.assert_queries(data, [query((0,1)),query((2,1)),QuerySpec('single',(0,),{},1,10)],
            missing={(2,10,20)})

    def test_zero_distance_and_reverse_horizontal_boundaries(self):
        data = self.data([(f,i,c,x,y) for f in (1,2) for i,c,x,y in
            ((10,0,20.,20.),(20,1,20.,20.),(30,2,70.,20.))])
        self.assert_queries(data, [
            query((1,0),{(0,1):RelationConstraint(0,0,frozenset({0}))}),
            query((2,0),{(0,1):RelationConstraint(2,2,frozenset({10}))})])

    def test_constraint_compilation_reused_within_query_only(self):
        data = self.data([(f,i,c,x,20.) for f in (1,2,3,4) for i,c,x in ((10,0,10.),(20,1,60.))])
        _, fast, _, _ = make_fast(data, windows={1:{1,2},2:{3,4}})
        q = query()
        with patch.object(fast, '_compile_allowed_bins', wraps=fast._compile_allowed_bins) as compiled:
            fast.query(q)
            self.assertEqual(compiled.call_count,1)
            fast.query(q)
            self.assertEqual(compiled.call_count,2)

    def test_metadata_reports_type_directory_cost_separately(self):
        data = self.data([(1,10,0,10.,20.),(1,20,1,60.,20.)])
        _, fast, _, _ = make_fast(data)
        self.assertEqual(fast.metadata['typed_pair_directory_entries'],2)
        self.assertGreater(fast.metadata['typed_pair_directory_pickle_bytes'],0)
        self.assertGreaterEqual(fast.metadata['setup_s'],fast.metadata['typed_pair_directory_setup_s'])

    def test_random_small_inputs_match_all_results(self):
        import random
        rng = random.Random(517)
        for size in (2,3,4):
            data = self.data([(f,i,i%3,float(rng.randrange(100)),float(rng.randrange(100)))
                for f in range(1,9) for i in range(6) if rng.random()>.2])
            relations = {(i,i+1):RelationConstraint(0,2,frozenset(range(-7,7))) for i in range(size-1)}
            self.assert_queries(data,[query(tuple(i%3 for i in range(size)),relations)],
                windows={1:{1,2,3,4},2:{5,6,7,8}})


if __name__ == '__main__':
    unittest.main()
