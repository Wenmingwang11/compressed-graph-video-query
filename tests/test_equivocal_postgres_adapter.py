"""Real PostgreSQL parity tests for the EQUI-VOCAL execution adaptation."""
import importlib.util
import math
import random
import unittest

import pandas as pd

from supplement.corrected_query_model import (
    QuerySpec, RelationConstraint, exact_query,
)


def table(rows):
    return pd.DataFrame(rows, columns=['frame', 'track_id', 'class_id', 'cx', 'cy'])


class EquivocalPostgresTests(unittest.TestCase):
    def strategy(self, data):
        self.assertIsNotNone(
            importlib.util.find_spec('supplement.equivocal_postgres_adapter'),
            'The EQUI-VOCAL PostgreSQL execution adapter is not implemented',
        )
        from supplement.equivocal_postgres_adapter import EquivocalPostgresStrategy
        strategy = EquivocalPostgresStrategy(data, 100, 100)
        self.addCleanup(strategy.release)
        return strategy

    def assert_parity(self, data, query):
        strategy = self.strategy(data)
        try:
            expected = exact_query(data, query, 100, 100, return_all=True)
            self.assertEqual(strategy.query(query, return_all=True), expected)
            self.assertEqual(strategy.query(query), expected[:query.topk])
        finally:
            strategy.release()

    def test_missing_frames_split_and_duration_windows_merge_to_maximal_intervals(self):
        rows = []
        for frame in [1, 2, 4, 5, 6, 7, 10, 11, 12]:
            rows += [(frame, 10, 0, 0., 0.), (frame, 20, 1, 50., 0.)]
        q = QuerySpec('gaps', (0, 1), {(0, 1): RelationConstraint(2, 2, frozenset({0}))}, 3, 1)
        self.assert_parity(table(rows), q)

    def test_role_direction_and_zero_distance(self):
        rows = [(f, oid, cls, x, y) for f in [0, 1, 2]
                for oid, cls, x, y in [(7, 0, 50., 0.), (9, 1, 0., 0.), (12, 1, 50., 0.)]]
        for bins, distance in [(frozenset({10}), 2), (frozenset({0}), 0), (frozenset({0}), 2)]:
            with self.subTest(bins=bins, distance=distance):
                q = QuerySpec('direction', (0, 1), {(0, 1): RelationConstraint(distance, distance, bins)}, 1, 50)
                self.assert_parity(table(rows), q)

    def test_classes_are_checked_each_frame(self):
        rows = [(f, 10, 0 if f != 3 else 2, 0., 0.) for f in range(1, 7)]
        rows += [(f, 20, 1, 50., 0.) for f in range(1, 7)]
        q = QuerySpec('class-change', (0, 1), {(0, 1): RelationConstraint(2, 2, frozenset({0}))}, 2, 20)
        self.assert_parity(table(rows), q)

    def test_three_roles_are_injective_and_bindings_keep_original_ids(self):
        rows = [(f, oid, cls, x, 0.) for f in range(2, 7)
                for oid, cls, x in [(101, 0, 0.), (309, 0, 25.), (801, 1, 50.)]]
        q = QuerySpec('three', (0, 0, 1), {
            (0, 1): RelationConstraint(1, 1, frozenset({0})),
            (1, 2): RelationConstraint(1, 1, frozenset({0})),
            (2, 0): RelationConstraint(2, 2, frozenset({10})),
        }, 3, 10)
        self.assert_parity(table(rows), q)

    def test_no_relations_empty_results_and_single_role(self):
        data = table([(1, 100, 0, 0., 0.), (2, 100, 0, 0., 0.), (4, 200, 1, 1., 0.)])
        for roles in [(0,), (0, 0), (0, 1), (99,)]:
            with self.subTest(roles=roles):
                self.assert_parity(data, QuerySpec('unary', roles, {}, 1, 2))
        self.assert_parity(table([]), QuerySpec('empty', (0,), {}, 1, 10))

    def test_quantization_boundaries_are_checked_in_postgres(self):
        vectors = [(50., 0.), (-50., 0.), (0., -50.), (0., 50.), (0., 0.)]
        for index in range(-10, 11):
            angle = index * math.pi / 10
            vectors.append((49. * math.cos(angle), 49. * math.sin(angle)))
        rows = [(i, 10, 0, 0., 0.) for i in range(len(vectors))]
        rows += [(i, 20, 1, x, y) for i, (x, y) in enumerate(vectors)]
        data = table(rows)
        for angle in range(-10, 11):
            with self.subTest(angle=angle):
                q = QuerySpec('boundary', (0, 1), {(0, 1): RelationConstraint(0, 2, frozenset({angle}))}, 1, 100)
                self.assert_parity(data, q)

    def test_repeated_query_id_does_not_reuse_answer_from_other_predicate(self):
        data = table([(f, oid, cls, x, 0.) for f in range(5)
                      for oid, cls, x in [(1, 0, 0.), (2, 1, 50.)]])
        strategy = self.strategy(data)
        for bins in [frozenset({0}), frozenset({10}), frozenset({0})]:
            q = QuerySpec('same-id', (0, 1), {(0, 1): RelationConstraint(2, 2, bins)}, 2, 10)
            self.assertEqual(strategy.query(q, return_all=True), exact_query(data, q, 100, 100, return_all=True))

    def test_conflicting_duplicate_track_preserves_all_classes_and_last_position(self):
        rows = [(f, oid, cls, x, 0.) for f in [1, 2, 3]
                for oid, cls, x in [(-1, 0, 0.), (-1, 1, 80.), (-1, 0, 10.),
                                    (-1, 0, 0.), (20, 2, 50.)]]
        for classes in [(0, 2), (1, 2), (0, 1)]:
            with self.subTest(classes=classes):
                q = QuerySpec('duplicates', classes, {(0, 1): RelationConstraint(2, 2, frozenset({0}))}, 2, 10)
                self.assert_parity(table(rows), q)

    def test_random_multirole_queries_agree_with_exhaustive_oracle(self):
        rng = random.Random(31)
        rows = [(f, oid, (oid + (f == 4)) % 2, rng.randrange(100), rng.randrange(100))
                for f in range(8) for oid in range(7) if rng.random() > .1]
        data = table(rows)
        strategy = self.strategy(data)
        for i in range(12):
            roles = (0, 1) if i % 2 else (0, 1, 0)
            q = QuerySpec(str(i), roles, {(0, 1): RelationConstraint(i % 2, 4, frozenset(range(-7, 8)))}, 1 + i % 3, 5)
            self.assertEqual(strategy.query(q, return_all=True), exact_query(data, q, 100, 100, return_all=True))


if __name__ == '__main__':
    unittest.main()
