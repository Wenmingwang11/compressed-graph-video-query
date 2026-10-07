import unittest

import pandas as pd

from supplement.corrected_query_model import exact_query
from supplement.generate_corrected_baseline_workload import (
    derive_query_from_event,
    query_from_dict,
    query_to_dict,
    select_real_event,
)


class CorrectedBaselineWorkloadTests(unittest.TestCase):
    def test_selects_earliest_deterministic_binding_with_required_streak(self):
        rows = []
        for frame in range(1, 8):
            rows.extend(
                [
                    (frame, 10, 2, 10.0, 10.0),
                    (frame, 20, 8, 40.0, 10.0),
                    (frame, 21, 8, 70.0, 10.0),
                ]
            )
        data = pd.DataFrame(rows, columns=["frame", "track_id", "class_id", "cx", "cy"])

        event = select_real_event(data, role_types=(2, 8), min_consecutive_frames=5)

        self.assertEqual(event.binding, (10, 20))
        self.assertEqual((event.start_frame, event.end_frame), (1, 5))

    def test_missing_frame_breaks_event_streak(self):
        rows = []
        for frame in (1, 2, 4, 5, 6):
            rows.extend([(frame, 10, 2, 10.0, 10.0), (frame, 20, 8, 40.0, 10.0)])
        data = pd.DataFrame(rows, columns=["frame", "track_id", "class_id", "cx", "cy"])

        event = select_real_event(data, role_types=(2, 8), min_consecutive_frames=3)

        self.assertEqual((event.start_frame, event.end_frame), (4, 6))

    def test_derived_query_uses_observed_directed_bins_and_matches_source_event(self):
        rows = []
        for frame, target_x in ((1, 35.0), (2, 40.0), (3, 45.0)):
            rows.extend(
                [
                    (frame, 10, 2, 10.0, 50.0),
                    (frame, 20, 8, target_x, 50.0),
                ]
            )
        data = pd.DataFrame(rows, columns=["frame", "track_id", "class_id", "cx", "cy"])
        event = select_real_event(data, role_types=(2, 8), min_consecutive_frames=3)

        query = derive_query_from_event(
            data,
            event,
            query_id="drtest-q1",
            frame_width=100,
            frame_height=100,
            topk=10,
        )

        self.assertEqual(set(query.relations), {(0, 1)})
        self.assertEqual(query.relations[(0, 1)].theta_bins, frozenset({0}))
        matches = exact_query(data, query, frame_width=100, frame_height=100)
        self.assertEqual(matches[0].binding, (10, 20))
        self.assertGreaterEqual(matches[0].duration_frames, 3)

        restored = query_from_dict(query_to_dict("synthetic", event, query))
        self.assertEqual(restored, query)

    def test_three_role_event_requires_three_distinct_tracks(self):
        rows = []
        for frame in (1, 2, 3):
            rows.extend(
                [
                    (frame, 10, 3, 10.0, 10.0),
                    (frame, 11, 3, 20.0, 10.0),
                    (frame, 30, 81, 40.0, 10.0),
                    (frame, 40, 82, 60.0, 10.0),
                ]
            )
        data = pd.DataFrame(rows, columns=["frame", "track_id", "class_id", "cx", "cy"])

        event = select_real_event(data, role_types=(3, 81, 82), min_consecutive_frames=3)

        self.assertEqual(event.binding, (10, 30, 40))
        self.assertEqual(len(set(event.binding)), 3)


if __name__ == "__main__":
    unittest.main()
