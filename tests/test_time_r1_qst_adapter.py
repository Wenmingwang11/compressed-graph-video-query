import unittest

from supplement.corrected_query_model import QuerySpec, RelationConstraint


class TimeR1AdapterTests(unittest.TestCase):
    def test_prompt_encodes_direction_distance_and_fixed_id_duration_without_answers(self):
        from supplement.time_r1_qst_adapter import query_text
        query = QuerySpec('Q1', (0, 2), {(0, 1): RelationConstraint(2, 2, frozenset({5}))}, 10, 10)
        prompt = query_text(query, {0: 'person', 2: 'car'}, 960, 540, 6.)
        for text in ('person', 'car', '375', '562.5', '90', '108', 'same individual', '10 consecutive'):
            self.assertIn(text, prompt)

    def test_parse_only_answer_and_explicit_sampled_frame_edges(self):
        from supplement.time_r1_qst_adapter import parse_prediction
        text = '<think>Maybe 50 to 70</think><answer>1.00 to 3.00</answer>'
        result = parse_prediction(text, first_frame=101, frame_count=100, sampled_fps=10.)
        self.assertEqual(result['start_frame'], 111)
        self.assertEqual(result['end_frame'], 130)
        self.assertNotIn('binding', result)
        for invalid in ('1 to 2', '<answer>4 to 2</answer>', '<answer>1 to 11</answer>'):
            with self.assertRaises(ValueError):
                parse_prediction(invalid, first_frame=101, frame_count=100, sampled_fps=10.)

    def test_missing_class_names_is_an_error_not_a_guessed_label(self):
        from supplement.time_r1_qst_adapter import query_text
        with self.assertRaises(ValueError):
            query_text(QuerySpec('Q', (81,), {}, 2, 1), {}, 100, 100, 2.)

    def test_request_uses_sampled_duration_and_drops_reference_answers(self):
        from supplement.time_r1_qst_adapter import build_requests
        query = QuerySpec('Q', (0,), {}, 2, 1)
        clips = [{'video': 'fixed_clip.mp4', 'first_frame': 101, 'frame_count': 120,
                  'sampled_fps': 6., 'source_event': {'binding': [9876], 'start_frame': 111},
                  'timestamp': [1., 4.]}]
        request = build_requests(query, clips, {0: 'person'}, 960, 540)[0]
        self.assertEqual(request['duration'], 20.)
        self.assertEqual(request['first_frame'], 101)
        self.assertNotIn('source_event', request)
        self.assertNotIn('timestamp', request)
        self.assertNotIn('9876', request['sentence'])


if __name__ == '__main__':
    unittest.main()
