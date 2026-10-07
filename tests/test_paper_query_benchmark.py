import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from supplement.paper_query_benchmark import (
    QueryBenchRow,
    _append_csv_rows,
    _choose_ordered_query_family,
    _estimate_query_complexity,
    _enumerate_obj_num_query_families,
    _format_progress,
    _format_query_trace,
    _has_minimum_topk_results,
    _sample_benchmark_query_types,
    _select_structural_query_family,
)


class PaperQueryBenchmarkTests(unittest.TestCase):
    def test_topk_query_requires_enough_eligible_results(self):
        self.assertTrue(_has_minimum_topk_results(SimpleNamespace(top_paths=list(range(50))), 50))
        self.assertFalse(_has_minimum_topk_results(SimpleNamespace(top_paths=list(range(49))), 50))

    def test_structural_query_family_selection_does_not_probe_runtime(self):
        families = [[0, 1, 2, 3], [4, 5, 6, 7]]

        self.assertEqual(_select_structural_query_family(families), [0, 1, 2, 3])
        self.assertIsNone(_select_structural_query_family([]))

    def test_append_csv_rows_writes_header_once(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rows.csv"
            _append_csv_rows(path, [{"a": 1, "b": 2}])
            _append_csv_rows(path, [{"a": 3, "b": 4}])

            with path.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.reader(f))

            self.assertEqual(rows, [["a", "b"], ["1", "2"], ["3", "4"]])

    def test_format_progress_includes_position_and_stage(self):
        message = _format_progress(7, 520, "drtest", "vary_s", "s=0.7")

        self.assertIn("[7/520]", message)
        self.assertIn("drtest", message)
        self.assertIn("vary_s", message)
        self.assertIn("s=0.7", message)

    def test_format_query_trace_includes_query_and_result_fields(self):
        row = QueryBenchRow(
            experiment="vary_s",
            video="drtest",
            s="0.7",
            obj_num=3,
            query_types="0,1,2",
            theta_lt=1.0,
            d_lt=2.0,
            frame_threshold=10,
            topk=10,
            windows_total=100,
            windows_after_type_filter=5,
            candidate_vertices_total=7,
            bindings_total=8,
            bindings_matched=2,
            pair_lookups=11,
            theta_d_bins_scanned=12,
            t_load_s=0.01,
            t_type_filter_s=0.02,
            t_candidate_vertex_s=0.03,
            t_spatial_match_s=0.04,
            t_prefix_tree_s=0.05,
            t_total_s=0.14,
            top_path_count=1,
            top1_frame_count=6,
        )

        message = _format_query_trace(7, 520, row)

        self.assertIn("[7/520]", message)
        self.assertIn("types=0,1,2", message)
        self.assertIn("theta_lt=1.0", message)
        self.assertIn("d_lt=2.0", message)
        self.assertIn("windows_after_type_filter=5", message)
        self.assertIn("candidate_vertices_total=7", message)
        self.assertIn("t_total_s=0.1400", message)

    def test_sample_benchmark_query_types_respects_allowed_types(self):
        import random

        index = {
            1: {0: {1}, 1: {2}, 2: {3}, 9: {4}},
            2: {0: {5}, 2: {6}, 5: {7}},
        }

        query_types = _sample_benchmark_query_types(index, 3, random.Random(42), {0, 1, 2})

        self.assertEqual(set(query_types), {0, 1, 2})

    def test_sample_benchmark_query_types_raises_when_allowed_types_have_no_valid_query(self):
        import random

        index = {
            1: {0: {1}, 9: {2}},
            2: {2: {3}, 9: {4}},
        }

        with self.assertRaises(ValueError):
            _sample_benchmark_query_types(index, 2, random.Random(42), {0, 2})

    def test_estimate_query_complexity_counts_windows_and_pair_work(self):
        index = {
            1: {0: {1, 2}, 1: {3}, 2: {4, 5}},
            2: {0: {6}, 2: {7}},
            3: {0: {8}, 1: {9}, 2: {10}},
        }

        windows, binding_upper_bound, pair_work = _estimate_query_complexity(index, [0, 1, 2])

        self.assertEqual(windows, 2)
        self.assertEqual(binding_upper_bound, 5)
        self.assertEqual(pair_work, 15)

    def test_estimate_query_complexity_respects_repeated_type_counts(self):
        index = {
            1: {0: {1, 2, 3}, 1: {4}},
            2: {0: {5}, 1: {6}},
        }

        windows, binding_upper_bound, pair_work = _estimate_query_complexity(
            index, [0, 1, 0]
        )

        self.assertEqual(windows, 1)
        self.assertEqual(binding_upper_bound, 6)
        self.assertEqual(pair_work, 18)

    def test_sample_benchmark_query_types_retries_until_complexity_within_limit(self):
        import random

        index = {
            1: {0: {1, 2, 3}, 1: {4, 5}, 2: {6, 7}},
            2: {0: {8}, 3: {9}, 5: {10}},
        }

        with patch(
            "supplement.paper_query_benchmark.sample_query_types",
            side_effect=[[0, 1, 2], [0, 3, 5]],
        ):
            query_types = _sample_benchmark_query_types(
                index,
                3,
                random.Random(42),
                None,
                max_estimated_pair_work=10,
                max_sampling_attempts=2,
            )

        self.assertEqual(query_types, [0, 3, 5])

    def test_choose_ordered_query_family_returns_monotonic_prefix_bindings(self):
        index = {
            1: {0: {1}, 1: {2, 3}, 2: {4, 5}},
            2: {0: {6}, 1: {7}, 2: {8}},
        }

        ordered_family, prefix_stats = _choose_ordered_query_family(
            index,
            [0, 1, 2],
            [2, 3],
            min_candidate_windows=1,
            max_binding_upper_bound=10,
        )

        self.assertEqual(set(ordered_family), {0, 1, 2})
        bindings = [binding_upper_bound for _, _, binding_upper_bound, _ in prefix_stats]
        self.assertEqual(bindings, sorted(bindings))

    def test_enumerate_obj_num_query_families_returns_valid_family(self):
        index = {
            1: {0: {1}, 1: {2}, 2: {3}, 3: {4}},
            2: {0: {5}, 1: {6}, 2: {7}, 3: {8}},
            3: {0: {9}, 1: {10}, 2: {11}, 3: {12}, 9: {13}},
        }

        families = _enumerate_obj_num_query_families(
            index,
            [2, 3, 4],
            allowed_types={0, 1, 2, 3, 9},
            family_type_pool_size=5,
            min_candidate_windows=2,
            max_binding_upper_bound=10,
        )

        self.assertTrue(families)
        self.assertEqual(set(families[0]), {0, 1, 2, 3})

    def test_enumerate_obj_num_query_families_can_repeat_supported_types(self):
        index = {
            1: {0: {1, 2, 3}, 1: {4, 5}, 2: {6, 7}},
            2: {0: {8, 9}, 1: {10, 11}, 2: {12, 13}},
        }

        families = _enumerate_obj_num_query_families(
            index,
            [2, 3, 4],
            allowed_types={0, 1, 2},
            family_type_pool_size=3,
            min_candidate_windows=2,
            max_binding_upper_bound=100,
        )

        self.assertTrue(families)
        self.assertEqual(len(families[0]), 4)
        self.assertLess(len(set(families[0])), 4)


if __name__ == "__main__":
    unittest.main()
