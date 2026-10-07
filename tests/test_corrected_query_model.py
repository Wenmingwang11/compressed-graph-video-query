import unittest

import pandas as pd

from supplement.corrected_query_model import (
    QuerySpec,
    RelationConstraint,
    exact_query,
    maximal_consecutive_segments,
    quantize_relation,
    result_signature,
)


class CorrectedQueryModelTests(unittest.TestCase):
    def test_relation_constraint_uses_closed_distance_interval_and_angle_set(self):
        constraint = RelationConstraint(d_min=2, d_max=4, theta_bins=frozenset({1, 2}))

        self.assertTrue(constraint.matches(theta_bin=1, distance_bin=2))
        self.assertTrue(constraint.matches(theta_bin=2, distance_bin=4))
        self.assertFalse(constraint.matches(theta_bin=0, distance_bin=3))
        self.assertFalse(constraint.matches(theta_bin=1, distance_bin=5))

    def test_quantized_relation_is_directed(self):
        forward = quantize_relation(
            source_xy=(10.0, 50.0),
            target_xy=(60.0, 50.0),
            frame_width=100,
            frame_height=100,
            theta_parts=10,
            distance_parts=8,
        )
        reverse = quantize_relation(
            source_xy=(60.0, 50.0),
            target_xy=(10.0, 50.0),
            frame_width=100,
            frame_height=100,
            theta_parts=10,
            distance_parts=8,
        )
        upward = quantize_relation(
            source_xy=(50.0, 50.0),
            target_xy=(50.0, 20.0),
            frame_width=100,
            frame_height=100,
            theta_parts=10,
            distance_parts=8,
        )

        self.assertEqual(forward, (0, 2))
        self.assertEqual(reverse, (10, 2))
        self.assertEqual(upward, (-5, 1))

    def test_maximal_segments_do_not_bridge_missing_frames(self):
        self.assertEqual(
            maximal_consecutive_segments([5, 2, 1, 4, 9]),
            [(1, 2), (4, 5), (9, 9)],
        )

    def test_exact_query_enforces_injective_bindings_and_consecutive_threshold(self):
        rows = []
        for frame in (1, 2, 4, 5, 6):
            rows.extend(
                [
                    (frame, 10, 0, 10.0, 50.0),
                    (frame, 20, 1, 60.0, 50.0),
                ]
            )
        data = pd.DataFrame(rows, columns=["frame", "track_id", "class_id", "cx", "cy"])
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

        results = exact_query(data, query, frame_width=100, frame_height=100)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].binding, (10, 20))
        self.assertEqual((results[0].start_frame, results[0].end_frame), (4, 6))
        self.assertEqual(results[0].duration_frames, 3)

    def test_repeated_role_types_cannot_bind_the_same_track_twice(self):
        data = pd.DataFrame(
            [
                (1, 10, 0, 10.0, 10.0),
                (2, 10, 0, 10.0, 10.0),
            ],
            columns=["frame", "track_id", "class_id", "cx", "cy"],
        )
        query = QuerySpec(
            query_id="same-type",
            role_types=(0, 0),
            relations={},
            min_consecutive_frames=1,
            topk=10,
        )

        self.assertEqual(exact_query(data, query, frame_width=100, frame_height=100), [])

    def test_candidate_frames_preserve_original_frame_numbers(self):
        rows = []
        for frame in (1, 2, 4):
            rows.extend([(frame, 10, 0, 10.0, 50.0), (frame, 20, 1, 60.0, 50.0)])
        data = pd.DataFrame(rows, columns=["frame", "track_id", "class_id", "cx", "cy"])
        query = QuerySpec(
            query_id="candidate-frames",
            role_types=(0, 1),
            relations={(0, 1): RelationConstraint(2, 2, frozenset({0}))},
            min_consecutive_frames=3,
            topk=10,
        )

        results = exact_query(
            data,
            query,
            frame_width=100,
            frame_height=100,
            candidate_frames={1, 2, 4},
        )

        self.assertEqual(results, [])

    def test_topk_order_and_signature_are_deterministic(self):
        rows = []
        for frame in (1, 2, 3, 4):
            rows.extend([(frame, 10, 0, 10.0, 50.0), (frame, 20, 1, 60.0, 50.0)])
        for frame in (1, 2, 3):
            rows.extend([(frame, 11, 0, 10.0, -100.0), (frame, 21, 1, 60.0, -100.0)])
        data = pd.DataFrame(rows, columns=["frame", "track_id", "class_id", "cx", "cy"])
        query = QuerySpec(
            query_id="topk",
            role_types=(0, 1),
            relations={(0, 1): RelationConstraint(2, 2, frozenset({0}))},
            min_consecutive_frames=2,
            topk=2,
        )

        first = exact_query(data, query, frame_width=100, frame_height=100)
        second = exact_query(data.sample(frac=1.0, random_state=7), query, 100, 100)

        self.assertEqual([item.binding for item in first], [(10, 20), (11, 21)])
        self.assertEqual(result_signature(first), result_signature(second))

    def test_topk_tie_breaks_by_binding_before_start_frame(self):
        rows = []
        for frame in (5, 6):
            rows.extend([(frame, 10, 0, 10.0, 50.0), (frame, 20, 1, 60.0, 50.0)])
        for frame in (1, 2):
            rows.extend([(frame, 11, 0, 10.0, -100.0), (frame, 21, 1, 60.0, -100.0)])
        data = pd.DataFrame(rows, columns=["frame", "track_id", "class_id", "cx", "cy"])
        query = QuerySpec(
            query_id="tie",
            role_types=(0, 1),
            relations={(0, 1): RelationConstraint(2, 2, frozenset({0}))},
            min_consecutive_frames=2,
            topk=1,
        )

        results = exact_query(data, query, frame_width=100, frame_height=100)

        self.assertEqual(results[0].binding, (10, 20))


if __name__ == "__main__":
    unittest.main()
