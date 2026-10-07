import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from supplement.llm_query_translation_benchmark import (
    build_ollama_payload,
    chunk_cases,
    compile_semantic_query,
    resolve_codex_executable,
    run_ollama_model,
    safe_model_slug,
    score_prediction,
    summarize_scores,
)


def _query(case_id: str = "q1", object_count: int = 2) -> dict:
    objects = [
        {
            "role": f"r{index + 1}",
            "category": "car",
            "attributes": [],
        }
        for index in range(object_count)
    ]
    relations = [
        {
            "source": "r1",
            "target": "r2",
            "distance": "near",
            "direction": "left_of",
        }
    ]
    if object_count == 3:
        relations.append(
            {
                "source": "r3",
                "target": "r2",
                "distance": "any",
                "direction": "right_of",
            }
        )
    return {
        "case_id": case_id,
        "objects": objects,
        "relations": relations,
        "DF": 10,
        "k": 10,
    }


class LlmQueryTranslationBenchmarkTests(unittest.TestCase):
    @patch("supplement.llm_query_translation_benchmark.urllib.request.urlopen")
    def test_ollama_timeout_is_recorded_and_later_cases_continue(self, urlopen):
        def response_for(case_id):
            envelope = {
                "message": {
                    "content": json.dumps({"predictions": [_query(case_id)]})
                },
                "total_duration": 1_000_000_000,
            }
            response = MagicMock()
            response.__enter__.return_value.read.return_value = json.dumps(envelope).encode()
            return response

        urlopen.side_effect = [
            response_for("Q01"),
            TimeoutError("timed out"),
            response_for("Q03"),
        ]
        cases = [
            {"case_id": "Q01", "text": "first"},
            {"case_id": "Q02", "text": "second"},
            {"case_id": "Q03", "text": "third"},
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            payload, metrics = run_ollama_model(
                "model-a",
                cases,
                Path(temp_dir),
                run_number=1,
                timeout_seconds=1,
                base_url="http://127.0.0.1:11434",
            )
            responses = json.loads(
                (Path(temp_dir) / "ollama_model-a_run1_ollama_response.json").read_text()
            )

        self.assertEqual([item["case_id"] for item in payload["predictions"]], ["Q01", "Q03"])
        self.assertEqual(responses[1]["case_ids"], ["Q02"])
        self.assertEqual(responses[1]["error_type"], "TimeoutError")
        self.assertEqual(metrics["failed_batches"], 1)

    def test_cases_are_chunked_without_dropping_the_tail(self):
        chunks = chunk_cases(list(range(5)), 2)

        self.assertEqual(chunks, [[0, 1], [2, 3], [4]])

    def test_ollama_payload_uses_schema_and_deterministic_generation(self):
        payload = build_ollama_payload(
            "qwen3:4b",
            [{"case_id": "Q01", "text": "Find a car left of a truck."}],
        )

        self.assertEqual(payload["model"], "qwen3:4b")
        self.assertFalse(payload["stream"])
        self.assertFalse(payload["think"])
        self.assertEqual(payload["options"]["temperature"], 0)
        self.assertEqual(payload["options"]["seed"], 0)
        self.assertEqual(payload["options"]["num_predict"], 2048)
        self.assertEqual(payload["format"]["required"], ["predictions"])

    def test_model_name_is_safe_for_windows_result_filenames(self):
        self.assertEqual(safe_model_slug("ollama/qwen3:4b"), "ollama_qwen3_4b")

    @patch("supplement.llm_query_translation_benchmark.shutil.which")
    def test_windows_prefers_cmd_wrapper_for_codex(self, which):
        which.side_effect = lambda name: (
            r"C:\Users\tester\AppData\Roaming\npm\codex.cmd"
            if name == "codex.cmd"
            else None
        )

        self.assertEqual(
            resolve_codex_executable(),
            r"C:\Users\tester\AppData\Roaming\npm\codex.cmd",
        )
        which.assert_called_once_with("codex.cmd")

    def test_score_ignores_attribute_and_relation_list_order(self):
        gold = _query()
        gold["objects"][0]["attributes"] = [
            {"name": "color", "value": "black"},
            {"name": "kind", "value": "sedan"},
        ]
        prediction = _query()
        prediction["objects"][0]["attributes"] = list(
            reversed(gold["objects"][0]["attributes"])
        )

        score = score_prediction(gold, prediction)

        self.assertTrue(score["attributes_correct"])
        self.assertTrue(score["exact_match"])

    def test_wrong_relation_or_df_prevents_exact_match(self):
        gold = _query()
        prediction = _query()
        prediction["relations"][0]["direction"] = "right_of"
        prediction["DF"] = 20

        score = score_prediction(gold, prediction)

        self.assertFalse(score["relations_correct"])
        self.assertFalse(score["df_correct"])
        self.assertFalse(score["exact_match"])
        self.assertTrue(score["objects_correct"])
        self.assertTrue(score["k_correct"])

    def test_summary_reports_pair_and_triple_exact_accuracy(self):
        pair = score_prediction(_query("q1", 2), _query("q1", 2))
        pair["model"] = "model-a"
        pair["run"] = 1
        triple_gold = _query("q2", 3)
        triple_prediction = _query("q2", 3)
        triple_prediction["k"] = 5
        triple = score_prediction(triple_gold, triple_prediction)
        triple["model"] = "model-a"
        triple["run"] = 1

        summary = summarize_scores([pair, triple])

        self.assertEqual(summary[0]["n_cases"], 2)
        self.assertEqual(summary[0]["exact_match_pct"], 50.0)
        self.assertEqual(summary[0]["m2_exact_match_pct"], 100.0)
        self.assertEqual(summary[0]["m3_exact_match_pct"], 0.0)

    def test_semantic_relation_compiles_to_deterministic_bins(self):
        query = _query()
        query["objects"][0]["attributes"] = [{"name": "color", "value": "black"}]
        compiled = compile_semantic_query(
            query,
            category_ids={"car": 2},
            theta_parts=10,
            distance_parts=8,
        )

        self.assertEqual(compiled["role_types"], [2, 2])
        self.assertEqual(compiled["role_attributes"], [{"color": "black"}, {}])
        self.assertEqual(compiled["relations"][0]["distance_bins"], [0, 2])
        self.assertEqual(compiled["relations"][0]["theta_bins"], [-2, -1, 0, 1, 2])
        self.assertEqual(compiled["min_consecutive_frames"], 10)
        self.assertEqual(compiled["topk"], 10)


if __name__ == "__main__":
    unittest.main()
