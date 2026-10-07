from __future__ import annotations

import pandas as pd
import unittest

from structured_query_common import parse_pair_constraints, run_query_dataframe


def _toy_detections() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "frame": 1,
                "track_id": 10,
                "class_id": 2,
                "x1": 0.0,
                "y1": 0.0,
                "x2": 10.0,
                "y2": 10.0,
                "cx": 5.0,
                "cy": 5.0,
                "width": 10.0,
                "height": 10.0,
            },
            {
                "frame": 1,
                "track_id": 20,
                "class_id": 8,
                "x1": 20.0,
                "y1": 0.0,
                "x2": 30.0,
                "y2": 10.0,
                "cx": 25.0,
                "cy": 5.0,
                "width": 10.0,
                "height": 10.0,
            },
            {
                "frame": 2,
                "track_id": 10,
                "class_id": 2,
                "x1": 0.0,
                "y1": 0.0,
                "x2": 10.0,
                "y2": 10.0,
                "cx": 5.0,
                "cy": 5.0,
                "width": 10.0,
                "height": 10.0,
            },
        ]
    )


class StructuredQueryCommonTest(unittest.TestCase):
    def test_parse_pair_constraints_requires_all_role_pairs(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing pairs"):
            parse_pair_constraints("0-1:1,1", role_count=3)

    def test_run_query_dataframe_matches_quantized_pair_relation(self) -> None:
        constraints = parse_pair_constraints("0-1:1,1", role_count=2)

        result = run_query_dataframe(
            df=_toy_detections(),
            query_types=[2, 8],
            pair_constraints=constraints,
            frame_threshold=0,
            frame_width=100,
            frame_height=100,
            temporal_mode="union_count",
            topk=10,
        )

        self.assertEqual(len(result.matches), 1)
        self.assertEqual(result.matches[0].binding, (10, 20))
        self.assertEqual(result.matches[0].frame_count, 1)

    def test_run_query_dataframe_rejects_failed_pair_relation(self) -> None:
        constraints = parse_pair_constraints("0-1:0,1", role_count=2)

        result = run_query_dataframe(
            df=_toy_detections(),
            query_types=[2, 8],
            pair_constraints=constraints,
            frame_threshold=0,
            frame_width=100,
            frame_height=100,
            temporal_mode="union_count",
            topk=10,
        )

        self.assertEqual(result.matches, [])


if __name__ == "__main__":
    unittest.main()
