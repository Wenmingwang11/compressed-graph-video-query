import unittest


class FullBaselineFigureTests(unittest.TestCase):
    def test_native_data_identity_must_match_current_strict_run(self):
        import tempfile
        import json
        from pathlib import Path
        from supplement.render_full_baseline_suite import load_groups,DATASETS,METHODS
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            strict,native = root/'strict',root/'native'
            strict.mkdir(); native.mkdir()
            identities = {d:dict(data_sha256='input',workload_sha256='workload',repetitions=3) for d in DATASETS}
            (strict/'suite_identity.json').write_text(json.dumps(identities))
            for dataset in DATASETS:
                target = strict/dataset
                (target/'groups').mkdir(parents=True)
                (target/'artifacts').mkdir()
                (target/'manifest.json').write_text(json.dumps(dict(identities[dataset],complete=True,
                    integrity=dict(data=True,source=True,native=True,index=True))))
                for method in METHODS:
                    (target/'artifacts'/f'{method}.metadata.json').write_text('{}')
                    for query in ['Q1','Q2']:
                        group = dict(dataset=dataset,query_id=query,method=method,median_s=2.,
                            source='measured_full_dataset',raw=[dict(elapsed_s=x) for x in [1.,2.,3.]])
                        (target/'groups'/f'{method}_{query}.json').write_text(json.dumps(group))
            (native/DATASETS[0]).mkdir()
            (native/DATASETS[0]/'manifest.json').write_text(json.dumps(dict(complete=True,integrity=True,
                identity=dict(data_sha256='wrong input',workload_sha256='workload'))))
            with self.assertRaisesRegex(ValueError,'Native/strict input mismatch'):
                load_groups(strict,native)

    def test_validate_coverage_rejects_duplicate_and_missing_cells(self):
        from supplement.render_full_baseline_suite import validate_coverage
        rows = [dict(dataset='d',query_id='Q1',method='m')]
        validate_coverage(rows,['d'],['Q1'],['m'])
        for invalid in [[],rows+rows]:
            with self.assertRaises(ValueError):
                validate_coverage(invalid,['d'],['Q1'],['m'])

    def test_time_values_are_recomputed_from_raw_repetitions(self):
        from supplement.render_full_baseline_suite import verified_median
        group = dict(raw=[dict(elapsed_s=x) for x in (1.,4.,2.)],median_s=2.,source='measured_full_dataset')
        self.assertEqual(verified_median(group,3),2.)
        group['median_s']=100.
        with self.assertRaises(ValueError):
            verified_median(group,3)
        group.update(median_s=2.,source='estimated_full_query')
        with self.assertRaises(ValueError):
            verified_median(group,3)


if __name__=='__main__':
    unittest.main()
