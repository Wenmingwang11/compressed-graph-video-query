import unittest

from supplement.relocate_m2_benchmark import (
    DetectionRow,
    build_frame_spans,
    find_pair_sequences,
    find_query_instance,
    find_query_sequence_instance,
    pair_satisfies_constraint,
    parse_pair_constraint,
)


class RelocateM2BenchmarkTests(unittest.TestCase):
    def test_build_frame_spans_uses_image_count_minus_one(self):
        spans = build_frame_spans(
            [
                ("MVI_1", 5),
                ("MVI_2", 4),
            ]
        )

        self.assertEqual(len(spans), 2)
        self.assertEqual((spans[0].global_start, spans[0].global_end), (1, 4))
        self.assertEqual((spans[1].global_start, spans[1].global_end), (5, 7))

    def test_parse_pair_constraint(self):
        constraint = parse_pair_constraint("0-1:10,8")

        self.assertEqual((constraint.left_role, constraint.right_role), (0, 1))
        self.assertEqual((constraint.theta_lt, constraint.d_ratio_lt), (10.0, 8.0))

    def test_pair_satisfies_constraint(self):
        left = DetectionRow(frame=1, object_id=10, left=10.0, top=10.0, width=20.0, height=20.0, cls=2)
        right = DetectionRow(frame=1, object_id=11, left=40.0, top=10.0, width=20.0, height=20.0, cls=8)

        self.assertTrue(
            pair_satisfies_constraint(
                left,
                right,
                theta_lt=10.0,
                d_ratio_lt=8.0,
                frame_width=100,
                frame_height=50,
            )
        )
        self.assertFalse(
            pair_satisfies_constraint(
                left,
                right,
                theta_lt=10.0,
                d_ratio_lt=0.05,
                frame_width=100,
                frame_height=50,
            )
        )

    def test_find_query_instance_returns_valid_pair(self):
        detections = [
            DetectionRow(frame=1, object_id=1, left=10.0, top=10.0, width=20.0, height=20.0, cls=2),
            DetectionRow(frame=1, object_id=2, left=40.0, top=10.0, width=20.0, height=20.0, cls=8),
            DetectionRow(frame=2, object_id=3, left=50.0, top=50.0, width=10.0, height=10.0, cls=2),
        ]

        query = find_query_instance(
            detections,
            query_types=[2, 8],
            pair_constraint=parse_pair_constraint("0-1:10,8"),
            frame_width=100,
            frame_height=50,
        )

        self.assertEqual(query.frame, 1)
        self.assertEqual(query.roles[0].cls, 2)
        self.assertEqual(query.roles[1].cls, 8)

    def test_find_pair_sequences_requires_consecutive_10_frames(self):
        detections = []
        for frame in range(1, 11):
            detections.append(
                DetectionRow(frame=frame, object_id=1, left=10.0, top=10.0, width=20.0, height=20.0, cls=2)
            )
            detections.append(
                DetectionRow(frame=frame, object_id=2, left=40.0, top=10.0, width=20.0, height=20.0, cls=8)
            )
        for frame in range(20, 29):
            detections.append(
                DetectionRow(frame=frame, object_id=3, left=10.0, top=10.0, width=20.0, height=20.0, cls=2)
            )
            detections.append(
                DetectionRow(frame=frame, object_id=4, left=40.0, top=10.0, width=20.0, height=20.0, cls=8)
            )

        sequences = find_pair_sequences(
            detections,
            query_types=[2, 8],
            pair_constraint=parse_pair_constraint("0-1:10,8"),
            frame_width=100,
            frame_height=50,
            temporal_length=10,
        )

        self.assertEqual(len(sequences), 1)
        self.assertEqual(sequences[0].start_frame, 1)
        self.assertEqual(sequences[0].end_frame, 10)
        self.assertEqual((sequences[0].left_object_id, sequences[0].right_object_id), (1, 2))

    def test_find_query_sequence_instance_returns_earliest_10_frame_sequence(self):
        detections = []
        for frame in range(1, 11):
            detections.append(
                DetectionRow(frame=frame, object_id=1, left=10.0, top=10.0, width=20.0, height=20.0, cls=2)
            )
            detections.append(
                DetectionRow(frame=frame, object_id=2, left=40.0, top=10.0, width=20.0, height=20.0, cls=8)
            )
        for frame in range(12, 22):
            detections.append(
                DetectionRow(frame=frame, object_id=5, left=10.0, top=10.0, width=20.0, height=20.0, cls=2)
            )
            detections.append(
                DetectionRow(frame=frame, object_id=6, left=40.0, top=10.0, width=20.0, height=20.0, cls=8)
            )

        query = find_query_sequence_instance(
            detections,
            query_types=[2, 8],
            pair_constraint=parse_pair_constraint("0-1:10,8"),
            frame_width=100,
            frame_height=50,
            temporal_length=10,
        )

        self.assertEqual((query.start_frame, query.end_frame), (1, 10))
        self.assertEqual(len(query.frames), 10)
        self.assertEqual((query.left_object_id, query.right_object_id), (1, 2))


if __name__ == "__main__":
    unittest.main()
