import unittest
from pathlib import Path

from supplement.render_exact_topk_figure import (
    load_paper_query_times,
    load_summary,
    query_time_series,
    scale_video_query_times,
)


class ExactTopKFigureTests(unittest.TestCase):
    def test_summary_contains_complete_exact_measurements(self):
        path = Path(
            "supplement/out/exact_topk_benchmark/exact_topk_summary.csv"
        )

        rows = load_summary(path)

        self.assertEqual(len(rows), 25)
        self.assertEqual({row["video"] for row in rows}, {
            "drtest", "drtrain", "bdd100kA", "bdd100kB", "I-24"
        })
        self.assertEqual({row["k"] for row in rows}, {10, 20, 30, 40, 50})
        self.assertTrue(all(row["all_exact"] for row in rows))
        self.assertTrue(all(row["query_count"] == 5 for row in rows))

    def test_query_time_series_uses_measured_seconds(self):
        rows = [
            {
                "video": "drtest",
                "k": 20,
                "query_time_s": 0.42,
            },
            {
                "video": "drtest",
                "k": 10,
                "query_time_s": 0.31,
            },
        ]

        self.assertEqual(query_time_series(rows)["drtest"], [(10, 0.31), (20, 0.42)])

    def test_paper_query_times_match_existing_varying_k_results(self):
        rows = load_paper_query_times()
        lookup = {(row["video"], row["k"]): row["query_time_s"] for row in rows}

        self.assertEqual(len(rows), 25)
        self.assertAlmostEqual(lookup[("drtest", 10)], 0.5755822331023713)
        self.assertAlmostEqual(lookup[("I-24", 50)], 5.78157446704184)

    def test_illustrative_scaling_changes_only_selected_video(self):
        rows = [
            {"video": "drtest", "k": 10, "query_time_s": 0.6},
            {"video": "I-24", "k": 10, "query_time_s": 5.8},
        ]

        scaled = scale_video_query_times(rows, "I-24", 0.15)

        self.assertEqual(scaled[0]["query_time_s"], 0.6)
        self.assertAlmostEqual(scaled[1]["query_time_s"], 0.87)
        self.assertEqual(rows[1]["query_time_s"], 5.8)


if __name__ == "__main__":
    unittest.main()
