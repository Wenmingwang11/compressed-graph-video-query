"""Tests of symbolic MV coverage plus exact event-query semantics."""
import importlib.util
import math
import random
import unittest

import pandas as pd

from supplement.corrected_query_model import QuerySpec, RelationConstraint, exact_query


def data_table(rows):
    return pd.DataFrame(rows, columns=['frame', 'track_id', 'class_id', 'cx', 'cy'])


class EVASymbolicMVTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('supplement.eva_symbolic_mv_adapter'),
                             'EVA symbolic MV adaptation is not implemented')
        from supplement import eva_symbolic_mv_adapter
        return eva_symbolic_mv_adapter

    def test_symbolic_domains_compute_partial_overlap_difference_union_without_rows(self):
        Domain = self.module().PredicateDomain
        first = Domain.from_ranges({0: [(1, 5)], 1: [(10, 20)]})
        second = Domain.from_ranges({0: [(4, 8)], 1: [(15, 17)], 2: [(2, 4)]})
        self.assertEqual((first & second).ranges, ((0, ((4, 5),)), (1, ((15, 17),))))
        self.assertEqual((second - first).ranges, ((0, ((6, 8),)), (2, ((2, 4),))))
        self.assertEqual((first | second).ranges, ((0, ((1, 8),)), (1, ((10, 20),)), (2, ((2, 4),))))
        self.assertTrue((first - first).is_empty)

    def test_udf_signatures_are_alias_independent_and_keep_ordered_parameter_values(self):
        Signature = self.module().UDFSignature
        self.assertEqual(Signature.create('Feature', ('source',), {'b': 2, 'a': 1}),
                         Signature.create('Feature', ('source',), {'a': 1, 'b': 2}))
        self.assertNotEqual(Signature.create('Feature', ('source',), {'roles': [0, 1]}),
                            Signature.create('Feature', ('source',), {'roles': [1, 0]}))
        self.assertNotEqual(Signature.create('Feature', ('source',), {}),
                            Signature.create('Feature', ('other-source',), {}))

    def test_conditional_apply_evaluates_only_uncovered_input_and_reuses_overlap(self):
        m = self.module()
        data = data_table([(f, 10, 0, float(f), 0.) for f in range(1, 9)])
        manager = m.MaterializedFeatureManager(data)
        manager.materialize(m.PredicateDomain.from_ranges({0: [(1, 5)]}))
        stats = {}
        answer = manager.conditional_apply(m.PredicateDomain.from_ranges({0: [(4, 8)]}), stats=stats)
        self.assertEqual(answer.frame.tolist(), [4, 5, 6, 7, 8])
        self.assertEqual(stats['mv_reused_input_rows'], 2)
        self.assertEqual(stats['mv_computed_input_rows'], 3)
        again = {}
        manager.conditional_apply(m.PredicateDomain.from_ranges({0: [(4, 8)]}), stats=again)
        self.assertEqual(again['mv_computed_input_rows'], 0)
        self.assertEqual(again['mv_reused_input_rows'], 5)

    def strategy(self, data, **kwargs):
        s = self.module().EVASymbolicMVStrategy(data, 100, 100, **kwargs)
        self.addCleanup(s.release)
        return s

    def assert_parity(self, data, q, **kwargs):
        s = self.strategy(data, **kwargs)
        expected = exact_query(data, q, 100, 100, return_all=True)
        self.assertEqual(s.query(q, return_all=True), expected)
        self.assertEqual(s.query(q), expected[:q.topk])

    def test_partial_offline_materialization_is_reset_between_repeated_queries(self):
        data = data_table([(f, oid, cls, x, 0.) for f in range(1, 9)
                           for oid, cls, x in [(10, 0, 0.), (20, 1, 50.)]])
        s = self.strategy(data, preload_fraction=.5)
        q = QuerySpec('one', (0, 1), {(0, 1): RelationConstraint(2, 2, frozenset({0}))}, 3, 10)
        first, second = {}, {}
        a = s.query(q, stats=first, return_all=True)
        b = s.query(q, stats=second, return_all=True)
        self.assertEqual(a, exact_query(data, q, 100, 100, return_all=True))
        self.assertEqual(a, b)
        self.assertGreater(first['mv_computed_input_rows'], 0)
        self.assertEqual(first['mv_computed_input_rows'], second['mv_computed_input_rows'])
        self.assertEqual(first['mv_reused_input_rows'], second['mv_reused_input_rows'])

    def test_three_roles_class_change_missing_frames_and_chunk_boundaries(self):
        rows = [(f, oid, cls if not (f == 5 and oid == 33) else 2, x, 0.)
                for f in [1, 2, 3, 4, 5, 6, 8, 9, 10, 11]
                for oid, cls, x in [(33, 0, 0.), (44, 0, 25.), (55, 1, 50.)]]
        q = QuerySpec('three', (0, 0, 1), {
            (0, 1): RelationConstraint(1, 1, frozenset({0})),
            (1, 2): RelationConstraint(1, 1, frozenset({0})),
            (2, 0): RelationConstraint(2, 2, frozenset({10})),
        }, 3, 1)
        self.assert_parity(data_table(rows), q, frame_batch_size=2)

    def test_angle_boundaries_zero_distance_and_same_query_id_do_not_cache_answers(self):
        vectors = [(0., 0.), (50., 0.), (-50., 0.), (0., -50.)]
        vectors += [(49 * math.cos(i * math.pi / 10), 49 * math.sin(i * math.pi / 10)) for i in range(-10, 11)]
        data = data_table([(i, 10, 0, 0., 0.) for i in range(len(vectors))] +
                          [(i, 20, 1, x, y) for i, (x, y) in enumerate(vectors)])
        s = self.strategy(data)
        for angle in range(-10, 11):
            q = QuerySpec('same', (0, 1), {(0, 1): RelationConstraint(0, 2, frozenset({angle}))}, 1, 100)
            self.assertEqual(s.query(q, return_all=True), exact_query(data, q, 100, 100, return_all=True))

    def test_empty_unary_and_no_relation_queries(self):
        data = data_table([(1, 100, 0, 0., 0.), (2, 100, 0, 0., 0.), (4, 200, 1, 1., 0.)])
        for roles in [(0,), (0, 0), (0, 1), (99,)]:
            self.assert_parity(data, QuerySpec('noedge', roles, {}, 1, 5))
        self.assert_parity(data_table([]), QuerySpec('empty', (0,), {}, 1, 10))

    def test_conflicting_duplicate_track_preserves_all_classes_and_last_position(self):
        rows = [(f, oid, cls, x, 0.) for f in [1, 2, 3]
                for oid, cls, x in [(-1, 0, 0.), (-1, 1, 80.), (-1, 0, 10.),
                                    (-1, 0, 0.), (20, 2, 50.)]]
        for classes in [(0, 2), (1, 2), (0, 1)]:
            with self.subTest(classes=classes):
                q = QuerySpec('duplicates', classes, {(0, 1): RelationConstraint(2, 2, frozenset({0}))}, 2, 10)
                self.assert_parity(data_table(rows), q)

    def test_random_queries_match_all_results_and_topk(self):
        rng = random.Random(71)
        rows = [(f, oid, (oid + (f == 5)) % 2, rng.randrange(100), rng.randrange(100))
                for f in range(10) for oid in range(7) if rng.random() > .1]
        data = data_table(rows)
        s = self.strategy(data, frame_batch_size=3)
        for i in range(20):
            roles = (0, 1) if i % 2 else (0, 1, 0)
            relations = {(0, 1): RelationConstraint(i % 2, 4, frozenset(range(-7, 8)))}
            if len(roles) == 3 and i % 4 == 0:
                relations[(2, 0)] = RelationConstraint(0, 2, frozenset(range(-10, 11)))
            q = QuerySpec(str(i), roles, relations, 1 + i % 3, 5)
            expected = exact_query(data, q, 100, 100, return_all=True)
            self.assertEqual(s.query(q, return_all=True), expected)
            self.assertEqual(s.query(q), expected[:5])


if __name__ == '__main__':
    unittest.main()
