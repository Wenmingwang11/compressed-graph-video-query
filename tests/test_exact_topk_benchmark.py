import unittest

from supplement.exact_topk_benchmark import _summarize_rows


class ExactTopkBenchmarkTests(unittest.TestCase):
    def test_summary_uses_trimmed_paired_means(self):
        rows = []
        for query_id, exhaustive_s, optimized_s, processed_ratio in [
            (1, 10.0, 5.0, 0.5),
            (2, 11.0, 5.5, 0.4),
            (3, 12.0, 6.0, 0.3),
            (4, 13.0, 6.5, 0.2),
            (5, 100.0, 90.0, 0.9),
        ]:
            rows.append(
                {
                    "video": "demo",
                    "k": 10,
                    "query_id": query_id,
                    "exhaustive_s": exhaustive_s,
                    "optimized_s": optimized_s,
                    "processed_window_ratio": processed_ratio,
                    "bound_setup_s": 0.1,
                    "early_terminated": True,
                    "exact_match": True,
                }
            )

        summary = _summarize_rows(rows)

        self.assertEqual(len(summary), 1)
        self.assertAlmostEqual(summary[0]["exhaustive_s"], 12.0)
        self.assertAlmostEqual(summary[0]["optimized_s"], 6.0)
        self.assertAlmostEqual(summary[0]["speedup"], 2.0)
        self.assertAlmostEqual(summary[0]["processed_window_ratio"], 0.4)
        self.assertTrue(summary[0]["all_exact"])


if __name__ == "__main__":
    unittest.main()
