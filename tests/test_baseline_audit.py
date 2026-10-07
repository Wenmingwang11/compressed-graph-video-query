import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch
from dataclasses import replace

import pandas as pd

from supplement.corrected_query_model import QuerySpec, RelationConstraint, exact_query
from supplement.corrected_baseline_strategies import build_strategy


def fixture():
    rows = []
    for frame in (1, 2, 4, 5, 6):
        rows.extend([(frame, 10, 1, 10., 50.), (frame, 20, 2, 60., 50.),
                     (frame, 21, 2, 10., 90.)])
    return pd.DataFrame(rows, columns=['frame', 'track_id', 'class_id', 'cx', 'cy'])


def query():
    return QuerySpec('audit', (1, 2), {(0, 1): RelationConstraint(2, 2, frozenset({0}))}, 2, 1)


def tuples(results):
    return [(r.binding, r.start_frame, r.end_frame) for r in results]


class BaselineAuditTests(unittest.TestCase):
    def test_runner_writes_measured_and_audited_results_without_overwriting(self):
        from supplement.run_baseline_audit import run
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'data.txt'
            source.write_text('fixture', encoding='utf-8')
            workload = root / 'workload.json'
            workload.write_text(json.dumps([{'dataset': 'drtest', 'query_id': 'audit',
                'role_types': [1, 2], 'relations': [{'source_role': 0, 'target_role': 1,
                'd_min': 2, 'd_max': 2, 'theta_bins': [0]}], 'min_consecutive_frames': 2,
                'topk': 1, 'theta_parts': 10, 'distance_parts': 8,
                'source_event': {'binding': [999, 999]}}]), encoding='utf-8')
            output = root / 'result'
            with patch('supplement.run_baseline_audit.dataset_storage_path', return_value=source), \
                 patch('supplement.run_baseline_audit.load_tracking_table', return_value=fixture()), \
                 patch('supplement.run_baseline_audit.dataset_frame_size', return_value=(100, 100)):
                run(output, dataset='drtest', query_id='audit', workload_path=workload,
                    methods=('Exact Frame Scan', 'VQPy-style'), repetitions=1, warmups=0)
                manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
                self.assertTrue(manifest['complete'])
                self.assertTrue(manifest['all_correct'])
                self.assertNotIn('source_event', manifest['query'])
                self.assertEqual(len(json.loads((output / 'results.json').read_text())), 2)
                with self.assertRaises(FileExistsError):
                    run(output, workload_path=workload)

    def test_craw_native_partition_lookup_optimization_preserves_ranges(self):
        from supplement.craw_element_adapter import load_native_craw, fast_splitter_class
        native = load_native_craw()
        frames = {str(f): [{'id': 1, 'cls': 1}, {'id': 2 if f < 55 else 3, 'cls': 2}]
                  for f in range(1, 151)}
        expected = native.MSUSplitter().split(frames)
        self.assertEqual(fast_splitter_class(native)().split(frames), expected)

    def test_craw_keeps_cross_msu_events_and_changing_short_track_classes(self):
        from supplement.craw_element_adapter import CrawElementStrategy
        rows = []
        for frame in range(1, 101):
            rows.extend([(frame, 10, 1, 10., 50.), (frame, 20, 2, 60., 50.)])
        # Same track changes class for only two frames; default native guidance
        # filters short tracks, but the retrieval posting must retain these classes.
        rows.extend([(f, 30, 3 if f in (70, 71) else 4, 60., 50.) for f in range(65, 75)])
        rows.extend([(80, 40, 5, 60., 50.), (81, 40, 5, 60., 50.)])
        data = pd.DataFrame(rows, columns=fixture().columns)
        with tempfile.TemporaryDirectory() as directory:
            strategy = CrawElementStrategy(data, 100, 100, artifact_dir=directory)
            actual = strategy.query(query(), return_all=True)
            self.assertEqual(tuples(actual), [((10, 20), 1, 100)])
            short = replace(query(), role_types=(1, 3))
            self.assertEqual(tuples(strategy.query(short)), [((10, 30), 70, 71)])
            self.assertEqual(tuples(strategy.query(replace(query(), role_types=(1, 5)))),
                             [((10, 40), 80, 81)])
            strategy.release()

    def test_ours_multiclass_track_membership_does_not_drop_earlier_class_event(self):
        import igraph as ig
        from vsimsearch.corrected_mli_querying import query_corrected_mli
        rows = []
        for frame in (1, 2, 3):
            rows.extend([(frame, 10, 1 if frame < 3 else 3, 10., 50.),
                         (frame, 20, 2, 60., 50.)])
        data = pd.DataFrame(rows, columns=fixture().columns)
        for use_window_fallback in (False, True):
            with self.subTest(window_fallback=use_window_fallback):
                graph = ig.Graph(directed=True)
                graph.add_vertices(1)
                graph.vs[0]['name'] = (10, 20)
                pairs = {(10, 20): {(0, 2): {1, 2, 3}}}
                graph.vs[0]['combinations'] = {} if use_window_fallback else pairs
                actual = query_corrected_mli({1: {1: {10}, 2: {20}, 3: {10}}},
                    {1: graph}, data, query(), 100, 100,
                    window_edge_indexes={1: pairs} if use_window_fallback else None,
                    return_all=True)
                self.assertEqual(tuples(actual), [((10, 20), 1, 2)])

    def test_ours_horizontal_reverse_edge_keeps_positive_pi_endpoint(self):
        import igraph as ig
        from vsimsearch.corrected_mli_querying import query_corrected_mli
        graph = ig.Graph(directed=True)
        graph.add_vertices(1)
        graph.vs[0]['name'] = (10, 20)
        graph.vs[0]['combinations'] = {(10, 20): {(0, 2): {1, 2, 4, 5, 6}}}
        reverse = replace(query(), relations={(1, 0): RelationConstraint(2, 2, frozenset({10}))})
        actual = query_corrected_mli({1: {1: {10}, 2: {20, 21}}}, {1: graph},
            fixture(), reverse, 100, 100, legacy_single_direction=True)
        self.assertEqual(tuples(actual), [((10, 20), 4, 6)])

    def test_full_results_preserve_fixed_binding_gaps_and_exact_df(self):
        stats = {}
        actual = exact_query(fixture(), query(), 100, 100, stats=stats, return_all=True)
        self.assertEqual(tuples(actual), [((10, 20), 4, 6), ((10, 20), 1, 2)])
        self.assertEqual(tuples(exact_query(fixture(), query(), 100, 100)), [((10, 20), 4, 6)])
        self.assertEqual(stats['candidate_frames'], 5)
        self.assertEqual(stats['binding_frame_enumerated'], 10)
        self.assertEqual(stats['binding_frame_admitted'], 10)
        self.assertEqual(stats['spatial_predicate_evaluations'], 10)
        self.assertEqual(stats['spatial_match_binding_frames'], 5)
        self.assertEqual(stats['qualifying_intervals'], 2)
        for name in ('verification_s', 'spatial_scan_s', 'temporal_merge_s', 'topk_s'):
            self.assertGreaterEqual(stats[name], 0)

    def test_enumeration_and_admission_are_distinct_units(self):
        stats = {}
        actual = exact_query(fixture(), query(), 100, 100, candidate_frames={4, 5, 6},
                             candidate_bindings={(10, 20)}, stats=stats, return_all=True)
        self.assertEqual(tuples(actual), [((10, 20), 4, 6)])
        self.assertEqual(stats['candidate_frames'], 3)
        self.assertEqual(stats['binding_frame_enumerated'], 6)
        self.assertEqual(stats['binding_frame_admitted'], 3)
        self.assertEqual(stats['spatial_predicate_evaluations'], 3)

    def test_restricted_star_artifact_does_not_reject_unindexed_pair(self):
        strategy = build_strategy('star', fixture(), 100, 100, allowed_type_pairs={(1, 2)})
        reverse = replace(query(), relations={(1, 0): RelationConstraint(2, 2, frozenset({10}))})
        self.assertEqual(tuples(strategy.query(reverse)), [((10, 20), 4, 6)])

    def test_five_strategies_full_correctness_and_audit_agree(self):
        for name in ('star', 'vqpy', 'vocal-udf', 'video-colbert', 'lava'):
            with self.subTest(name=name):
                strategy = build_strategy(name, fixture(), 100, 100)
                stats = {}
                actual = strategy.query(query(), stats=stats, return_all=True)
                self.assertEqual(tuples(actual), [((10, 20), 4, 6), ((10, 20), 1, 2)])
                self.assertGreaterEqual(stats['candidate_s'], 0)
                self.assertEqual(stats['spatial_match_binding_frames'], 5)
                self.assertLessEqual(stats['spatial_predicate_evaluations'], 10)

    def test_ordered_signature_detects_wrong_topk_order(self):
        from supplement.corrected_query_model import ordered_result_signature
        results = exact_query(fixture(), query(), 100, 100, return_all=True)
        self.assertNotEqual(ordered_result_signature(results), ordered_result_signature(reversed(results)))


if __name__ == '__main__':
    unittest.main()
