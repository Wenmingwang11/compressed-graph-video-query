import unittest

import pandas as pd

from supplement.corrected_query_model import QuerySpec, RelationConstraint, exact_query
from supplement.relocate_qst_benchmark import RelocateStyleStrategy, build_region_vector


class RelocateQSTBenchmarkTests(unittest.TestCase):
    def setUp(self):
        rows = []
        for frame in range(1, 6):
            rows.extend(
                [
                    (frame, 10, 0, 10.0 + frame, 40.0, 12.0, 20.0, 0.9),
                    (frame, 20, 1, 60.0 + frame, 40.0, 14.0, 22.0, 0.8),
                    (frame, 30, 2, 60.0 + frame, 70.0, 10.0, 16.0, 0.7),
                ]
            )
        self.data = pd.DataFrame(
            rows,
            columns=["frame", "track_id", "class_id", "cx", "cy", "width", "height", "confidence"],
        )

    def test_region_vector_is_finite_and_normalized(self):
        vector = build_region_vector(self.data[self.data["track_id"] == 10], 100, 100)

        self.assertEqual(vector.ndim, 1)
        self.assertTrue(pd.notna(vector).all())
        self.assertAlmostEqual(float((vector * vector).sum()), 1.0, places=6)

    def test_two_role_query_matches_exact_oracle(self):
        query = QuerySpec(
            query_id="Q1",
            role_types=(0, 1),
            relations={(0, 1): RelationConstraint(2, 2, frozenset({0}))},
            min_consecutive_frames=5,
            topk=10,
        )
        strategy = RelocateStyleStrategy(self.data, 100, 100)

        actual = strategy.query(query, binding=(10, 20), start_frame=1, end_frame=5)
        expected = exact_query(self.data, query, 100, 100)

        self.assertEqual(actual, expected)
        self.assertEqual(len(strategy.last_role_scores), 2)

    def test_three_role_query_supports_repeated_classes(self):
        duplicate = self.data[self.data["track_id"] == 20].copy()
        duplicate["track_id"] = 21
        duplicate["cx"] += 5.0
        data = pd.concat([self.data, duplicate], ignore_index=True)
        query = QuerySpec(
            query_id="Q2",
            role_types=(0, 1, 1),
            relations={
                (0, 1): RelationConstraint(2, 2, frozenset({0})),
                (0, 2): RelationConstraint(2, 2, frozenset({0})),
            },
            min_consecutive_frames=5,
            topk=10,
        )
        strategy = RelocateStyleStrategy(data, 100, 100)

        actual = strategy.query(query, binding=(10, 20, 21), start_frame=1, end_frame=5)
        expected = exact_query(data, query, 100, 100)

        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
