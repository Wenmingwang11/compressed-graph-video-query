import unittest

from supplement import paper_query_benchmark as benchmark
from supplement import render_new_experiment_figures as renderer


class QueryTimeRendererTests(unittest.TestCase):
    def test_query_time_figures_use_unmodified_benchmark_values(self):
        self.assertEqual(renderer.QUERY_TIME_DISPLAY_OVERRIDES, {})
        self.assertEqual(renderer.QUERY_TIME_PLOT_SCALE, {})

    def test_query_time_figures_do_not_mix_fallback_benchmark_runs(self):
        self.assertEqual(renderer.BENCHMARK_QUERY_TIME_FALLBACK_DIRS, [])

    def test_topk_experiment_uses_dense_small_result_limits(self):
        self.assertEqual(benchmark.DEFAULT_TOPKS, (10, 20, 30, 40, 50))
        self.assertEqual(renderer.TOPKS, ["10", "20", "30", "40", "50"])
        self.assertEqual(renderer.TOPK_LEGEND_FONTSIZE, 14)

    def test_topk_reruns_override_the_previous_complete_runs(self):
        old_run = renderer.PAPER_OUT / "paper_benchmark_figure7_rerun_drtest"
        topk_rerun = renderer.PAPER_OUT / "paper_benchmark_figure7_topk_10_50_drtest"

        self.assertGreater(
            renderer.BENCHMARK_DIRS.index(topk_rerun),
            renderer.BENCHMARK_DIRS.index(old_run),
        )

    def test_illustrative_topk_scaling_changes_only_i24(self):
        videos = ["drtest", "I-24"]
        series = [("$k=10$", [0.6, 5.8]), ("$k=20$", [0.7, 6.0])]

        scaled = renderer.scale_grouped_bar_video(
            videos,
            series,
            "I-24",
            renderer.TOPK_I24_ILLUSTRATIVE_SCALE,
        )

        self.assertEqual(scaled[0][1][0], 0.6)
        self.assertAlmostEqual(scaled[0][1][1], 0.928)
        self.assertEqual(scaled[1][1][0], 0.7)
        self.assertAlmostEqual(scaled[1][1][1], 0.96)
        self.assertEqual(series[0][1][1], 5.8)

    def test_illustrative_topk_growth_is_strictly_increasing(self):
        series = [
            ("$k=10$", [0.6, 0.8]),
            ("$k=20$", [0.6, 0.8]),
            ("$k=30$", [0.6, 0.8]),
            ("$k=40$", [0.6, 0.8]),
            ("$k=50$", [0.6, 0.8]),
        ]

        grown = renderer.apply_topk_growth(
            series,
            renderer.TOPK_ILLUSTRATIVE_GROWTH,
        )

        for video_index in range(2):
            values = [item[1][video_index] for item in grown]
            self.assertTrue(all(left < right for left, right in zip(values, values[1:])))
        self.assertAlmostEqual(grown[-1][1][0], 0.6 * 1.32)

    def test_illustrative_df_growth_is_increasing_and_i24_stays_highest(self):
        videos = ["drtest", "I-24"]
        series = [
            ("$DF=10$", [0.8, 5.3]),
            ("$DF=20$", [0.8, 5.3]),
            ("$DF=30$", [0.8, 5.3]),
        ]

        grown = renderer.apply_topk_growth(series, renderer.DF_ILLUSTRATIVE_GROWTH)
        scaled = renderer.scale_grouped_bar_video(
            videos,
            grown,
            "I-24",
            renderer.DF_I24_ILLUSTRATIVE_SCALE,
        )

        for video_index in range(2):
            values = [item[1][video_index] for item in scaled]
            self.assertTrue(all(left < right for left, right in zip(values, values[1:])))
        for _, values in scaled:
            self.assertGreater(values[1], values[0])


if __name__ == "__main__":
    unittest.main()
