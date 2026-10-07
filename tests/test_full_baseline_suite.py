import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from supplement.corrected_query_model import QueryResult


class FullBaselineSuiteTests(unittest.TestCase):
    def test_empty_csv_clears_stale_rows(self):
        import csv
        from supplement.run_full_baseline_suite import csv_write
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'summary.csv'
            csv_write(path,[{'value':123}])
            csv_write(path,[])
            with path.open(encoding='utf-8-sig') as handle:
                self.assertEqual(list(csv.DictReader(handle)),[])

    def test_binding_specific_frame_masks_do_not_cross_join_proposals(self):
        import pandas as pd
        from supplement.corrected_query_model import QuerySpec, exact_query
        data = pd.DataFrame([dict(frame=f,track_id=i,class_id=c,cx=i,cy=0)
                             for f in range(1,5) for i,c in [(1,1),(2,2),(3,1),(4,2)]])
        query = QuerySpec('Q',(1,2),{},2,10)
        masks = {(1,2):{1,2}, (3,4):{3,4}}
        results = exact_query(data,query,100,100,candidate_binding_frames=masks,return_all=True)
        self.assertEqual(results,[QueryResult((1,2),1,2),QueryResult((3,4),3,4)])

    def test_aggregation_rejects_stale_or_integrity_failed_manifest(self):
        from supplement.run_full_baseline_suite import checkpoint_validity
        identity = {'data_sha256': 'a', 'source_sha256': {'x':'b'}, 'workload_sha256':'c',
                    'repetitions':3, 'warmups':1, 'methods':['Exact Frame Scan'], 'index_sha256':{}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'manifest.json'
            path.write_text(json.dumps(dict(identity, integrity={'data':True, 'source':True,
                                                                 'native':True, 'index':True})))
            self.assertEqual(checkpoint_validity(Path(directory), identity), 'valid')
            self.assertEqual(checkpoint_validity(Path(directory), dict(identity,repetitions=1)), 'stale_identity')
            path.write_text(json.dumps(dict(identity, integrity={'data':False,'source':True,
                                                                'native':True,'index':True})))
            self.assertEqual(checkpoint_validity(Path(directory), identity), 'integrity_unverified_or_failed')

    def test_group_files_without_artifact_metadata_are_not_method_complete(self):
        from supplement.run_full_baseline_suite import method_complete
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            groups = [root/'Q1.json', root/'Q2.json']
            for path in groups:
                path.write_text('{}')
            self.assertFalse(method_complete(root, 'Exact Frame Scan', groups))
            (root/'artifacts').mkdir()
            (root/'artifacts/Exact Frame Scan.metadata.json').write_text('{}')
            self.assertTrue(method_complete(root, 'Exact Frame Scan', groups))

    def test_query_failure_does_not_prevent_next_group(self):
        from supplement.run_full_baseline_suite import measure_group_safely
        with patch('supplement.run_full_baseline_suite.measure_group', side_effect=[ValueError('Q1 fails'), {'status':'measured'}]):
            results = [measure_group_safely(None, None, [], repetitions=1, warmups=0) for _ in range(2)]
        self.assertEqual(results[0]['status'], 'failed')
        self.assertEqual(results[1]['status'], 'measured')

    def test_interval_metrics_hand_computed_false_positive_and_missing_result(self):
        from supplement.run_full_baseline_suite import interval_metrics
        truth = [QueryResult((1, 2), 1, 10), QueryResult((3, 4), 20, 30)]
        predicted = [truth[0], QueryResult((5, 6), 40, 50)]
        metrics = interval_metrics(predicted, truth)
        self.assertEqual((metrics['precision'], metrics['recall'], metrics['f1']), (.5, .5, .5))
        self.assertEqual((metrics['tp'], metrics['fp'], metrics['fn']), (1, 1, 1))

    def test_empty_result_metrics_are_defined_and_not_missing_measurements(self):
        from supplement.run_full_baseline_suite import interval_metrics
        result = QueryResult((1, 2), 1, 10)
        self.assertEqual(interval_metrics([], [])['f1'], 1.)
        self.assertEqual(interval_metrics([], [result])['f1'], 0.)
        self.assertEqual(interval_metrics([result], [])['f1'], 0.)

    def test_checkpoint_resume_rejects_input_or_source_changes(self):
        from supplement.run_full_baseline_suite import check_identity
        previous = {'data_sha256': 'a', 'source_sha256': {'x': 'b'}, 'workload_sha256': 'c',
                    'repetitions': 3, 'warmups': 1, 'methods': ['Exact Frame Scan'], 'index_sha256': {}}
        check_identity(previous, dict(previous))
        for key in ('data_sha256', 'source_sha256', 'workload_sha256', 'repetitions', 'index_sha256'):
            altered = dict(previous, **{key: 'changed'})
            with self.subTest(key=key), self.assertRaises(ValueError):
                check_identity(previous, altered)

    def test_ten_roster_entries_and_only_applicable_pruning_methods(self):
        from supplement.run_full_baseline_suite import ROSTER, PRUNING_METHODS
        self.assertEqual(len(ROSTER), 10)
        self.assertIn('Time-R1', ROSTER)
        self.assertIn('SketchQL', ROSTER)
        self.assertNotIn('Exact Frame Scan', PRUNING_METHODS)
        self.assertNotIn('Time-R1', PRUNING_METHODS)
        self.assertIn('Ours', PRUNING_METHODS)

    def test_group_measurement_records_all_repetitions_and_missing_reference_events(self):
        from supplement.run_full_baseline_suite import measure_group
        truth = [QueryResult((1, 2), 1, 10), QueryResult((3, 4), 20, 30)]
        class Strategy:
            def query(self, query, *, stats=None, return_all=False):
                if stats is not None:
                    stats.update(candidate_frames=10, binding_frame_enumerated=10,
                        binding_frame_admitted=10, spatial_predicate_evaluations=10,
                        spatial_match_binding_frames=10, qualifying_intervals=1,
                        candidate_s=.1, verification_s=.2, spatial_scan_s=.19,
                        temporal_merge_s=.009, topk_s=.001)
                return truth[:1]
        from supplement.corrected_query_model import QuerySpec
        group = measure_group(Strategy(), QuerySpec('Q1',(1,2),{},1,10), truth,
                              repetitions=2, warmups=1)
        self.assertEqual(len(group['raw']), 2)
        self.assertEqual(group['metrics']['recall'], .5)
        self.assertFalse(group['full_intervals_verified'])
        self.assertFalse(group['ordered_topk_verified'])
        self.assertTrue(all(row['elapsed_s'] >= 0 for row in group['raw']))


if __name__ == '__main__':
    unittest.main()
