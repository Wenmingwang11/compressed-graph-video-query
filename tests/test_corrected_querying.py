import unittest

import pandas as pd

from supplement.corrected_query_model import QuerySpec, RelationConstraint, exact_query, result_signature
from vsimsearch.corrected_querying import build_directed_relation_index, query_directed_relation_index


class CorrectedQueryingTests(unittest.TestCase):
    def setUp(self):
        rows = []
        for frame in (1, 2, 3, 5, 6):
            rows.extend(
                [
                    (frame, 10, 0, 10.0, 50.0),
                    (frame, 20, 1, 60.0, 50.0),
                    (frame, 30, 2, 60.0, 80.0),
                ]
            )
        self.data = pd.DataFrame(rows, columns=["frame", "track_id", "class_id", "cx", "cy"])

    def test_index_keeps_both_directions_and_compresses_runs(self):
        index = build_directed_relation_index(self.data, 100, 100)

        forward = index.postings[(0, 1, 0, 2)][(10, 20)]
        reverse = index.postings[(1, 0, 10, 2)][(20, 10)]
        self.assertEqual(forward, ((1, 3), (5, 6)))
        self.assertEqual(reverse, ((1, 3), (5, 6)))

    def test_query_matches_oracle_for_two_roles(self):
        query = QuerySpec(
            query_id="q1",
            role_types=(0, 1),
            relations={
                (0, 1): RelationConstraint(2, 2, frozenset({0})),
                (1, 0): RelationConstraint(2, 2, frozenset({10})),
            },
            min_consecutive_frames=3,
            topk=10,
        )
        index = build_directed_relation_index(self.data, 100, 100)

        actual = query_directed_relation_index(index, self.data, query)
        expected = exact_query(self.data, query, 100, 100)

        self.assertEqual(result_signature(actual), result_signature(expected))

    def test_query_matches_oracle_for_three_roles(self):
        query = QuerySpec(
            query_id="q2",
            role_types=(0, 1, 2),
            relations={
                (0, 1): RelationConstraint(2, 2, frozenset({0})),
                (1, 0): RelationConstraint(2, 2, frozenset({10})),
                (1, 2): RelationConstraint(1, 1, frozenset({5})),
                (2, 1): RelationConstraint(1, 1, frozenset({-5})),
            },
            min_consecutive_frames=3,
            topk=10,
        )
        index = build_directed_relation_index(self.data, 100, 100)

        actual = query_directed_relation_index(index, self.data, query)
        expected = exact_query(self.data, query, 100, 100)

        self.assertEqual(result_signature(actual), result_signature(expected))

    def test_index_rejects_query_with_different_quantization(self):
        query = QuerySpec(
            query_id="bad-parts",
            role_types=(0, 1),
            relations={(0, 1): RelationConstraint(2, 2, frozenset({0}))},
            min_consecutive_frames=1,
            topk=10,
            theta_parts=12,
            distance_parts=8,
        )
        index = build_directed_relation_index(self.data, 100, 100)

        with self.assertRaisesRegex(ValueError, "quantization"):
            query_directed_relation_index(index, self.data, query)

    def test_builder_can_limit_offline_edges_to_declared_type_pairs(self):
        index = build_directed_relation_index(
            self.data,
            100,
            100,
            allowed_type_pairs={(0, 1), (1, 0)},
        )

        self.assertTrue(index.postings)
        self.assertEqual({key[:2] for key in index.postings}, {(0, 1), (1, 0)})


if __name__ == "__main__":
    unittest.main()
