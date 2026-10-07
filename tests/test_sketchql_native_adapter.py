import unittest


class NativeSketchQLMappingTests(unittest.TestCase):
    def test_workload_requires_unique_q1_q2(self):
        from supplement.sketchql_native_adapter import validate_records
        validate_records([{'query_id':'Q1'},{'query_id':'Q2'}])
        for records in [[],[{'query_id':'Q1'}],[{'query_id':'Q1'},{'query_id':'Q1'}]]:
            with self.assertRaises(ValueError):
                validate_records(records)

    def test_checkpoint_requires_readable_group_and_proposals(self):
        import tempfile
        import json
        from pathlib import Path
        from supplement.sketchql_native_adapter import native_group_valid,PROTOCOL,VARIANT
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.assertFalse(native_group_valid(path,'d','Q1',1))
            (path/'Q1.json').write_text('broken')
            self.assertFalse(native_group_valid(path,'d','Q1',1))
            group = dict(dataset='d',query_id='Q1',method_variant=VARIANT,protocol=PROTOCOL,
                raw=[{'elapsed_s':1}],audit={'candidate_frames':1,'spatial_predicate_evaluations':1},
                metrics={'precision':1,'recall':1,'f1':1},full_results=[])
            (path/'Q1.json').write_text(json.dumps(group))
            self.assertFalse(native_group_valid(path,'d','Q1',1))
            (path/'Q1.proposals.json').write_text(json.dumps({'masks':[],'proposals':[]}))
            self.assertTrue(native_group_valid(path,'d','Q1',1))

    def test_half_open_windows_and_grouped_roles_keep_binding_frame_association(self):
        from supplement.sketchql_native_adapter import proposal_masks
        # Native output groups repeated classes: query roles A,B,A -> native A,A,B.
        records = [[{'video_obj_id':11,'frame_start_end':(0,2)},
                    {'video_obj_id':33,'frame_start_end':(0,2)},
                    {'video_obj_id':22,'frame_start_end':(0,2)}],
                   [{'video_obj_id':44,'frame_start_end':(2,4)},
                    {'video_obj_id':66,'frame_start_end':(2,4)},
                    {'video_obj_id':55,'frame_start_end':(2,4)}]]
        masks = proposal_masks(records,[1,2,1])
        self.assertEqual(masks,{(11,22,33):{1,2},(44,55,66):{3,4}})

    def test_invalid_mixed_window_rejected(self):
        from supplement.sketchql_native_adapter import proposal_masks
        with self.assertRaises(ValueError):
            proposal_masks([[{'video_obj_id':1,'frame_start_end':(0,2)},
                             {'video_obj_id':2,'frame_start_end':(1,3)}]],[1,2])

    def test_empty_candidates_remain_empty(self):
        from supplement.sketchql_native_adapter import proposal_masks
        self.assertEqual(proposal_masks([],[1,2]),{})


if __name__ == '__main__':
    unittest.main()
