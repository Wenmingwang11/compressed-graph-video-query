"""Integrity tests for publication data, rather than screenshot implementation."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


MODULE = Path(__file__).with_name('render_completion_suite_20260926.py')


class PublicationIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not MODULE.exists():
            return
        spec = importlib.util.spec_from_file_location('completion_renderer', MODULE)
        cls.renderer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.renderer)

    def require_renderer(self):
        self.assertTrue(MODULE.exists(), 'completion renderer is not implemented')
        return self.renderer

    def timing(self):
        return dict(source='measured_full_dataset', status='measured',
                    raw=[dict(elapsed_s=x) for x in (.01, .02, .06)],
                    median_s=.02, q1_s=.015, q3_s=.04)

    def test_ms_conversion_uses_raw_median_and_asymmetric_iqr(self):
        r = self.require_renderer()
        median, lower, upper = r.timing_ms(self.timing(), 3)
        self.assertAlmostEqual(median, 20)
        self.assertAlmostEqual(lower, 5)
        self.assertAlmostEqual(upper, 20)

    def test_fabricated_saved_quartile_or_missing_repetition_is_rejected(self):
        r = self.require_renderer()
        g = self.timing()
        g['q1_s'] = .001
        with self.assertRaisesRegex(ValueError, 'q1_s'):
            r.timing_ms(g, 3)
        with self.assertRaisesRegex(ValueError, 'repetitions'):
            r.timing_ms(self.timing(), 2)

    def test_source_hash_tampering_is_rejected(self):
        r = self.require_renderer()
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder)/'source.py'
            file.write_text('original', encoding='utf-8')
            expected = r.sha256(file)
            file.write_text('changed', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                r.verify_hash(file, expected, [])

    def test_strict_method_names_keep_adaptation_and_style_disclosure(self):
        r = self.require_renderer()
        self.assertEqual(len(r.METHODS), 10)
        self.assertIn('EVA-MV adapted', r.METHODS)
        self.assertIn('EQUI-VOCAL SQL adapted', r.METHODS)
        self.assertNotIn('Time-R1', r.METHODS)
        self.assertIn('VOCAL-UDF-style', r.METHODS)
        self.assertNotIn('SketchQL', r.METHODS)

    def semantic(self):
        per_query = []
        for q in range(1, 7):
            per_query.append(dict(method='Ours', query_id=f'SEM{q:02d}', tiou_threshold=.5,
                tp=1, fp=1, fn=3, prediction_count=2, reference_count=4,
                precision=.5, recall=.25, f1=1/3))
        return dict(protocol=dict(primary_tiou_threshold=.5, human_ground_truth=False,
            result_scope='all_before_topk', reference_kind='independent_model_assisted_AI_reviewed_tracklet_inventory'),
            per_query=per_query,
            overall=[dict(method='Ours', tiou_threshold=.5, aggregation='micro_over_six_queries',
                tp=6, fp=6, fn=18, prediction_count=12, reference_count=24,
                precision=.5, recall=.25, f1=1/3)])

    def test_semantic_micro_cannot_be_replaced_by_arbitrary_average(self):
        r = self.require_renderer()
        doc = self.semantic()
        overall, queries = r.validate_semantic(doc, ('Ours',))
        self.assertEqual(overall[0]['tp'], 6)
        self.assertEqual(len(queries), 6)
        bad = copy.deepcopy(doc)
        bad['overall'][0]['precision'] = 1
        with self.assertRaisesRegex(ValueError, 'precision'):
            r.validate_semantic(bad, ('Ours',))

    def test_zero_predictions_remain_undefined_not_one_hundred_percent(self):
        r = self.require_renderer()
        row = dict(tp=0, fp=0, fn=2, prediction_count=0, reference_count=2,
                   precision=None, recall=0, f1=0)
        r.validate_counts(row)
        row['precision'] = 1
        with self.assertRaisesRegex(ValueError, 'precision'):
            r.validate_counts(row)


if __name__ == '__main__':
    unittest.main()
