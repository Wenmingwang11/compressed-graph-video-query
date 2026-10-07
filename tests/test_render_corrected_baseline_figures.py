import csv
import json
import tempfile
import unittest
from pathlib import Path

from supplement.render_corrected_baseline_figures import METHODS, render_figures


class RenderCorrectedBaselineFiguresTests(unittest.TestCase):
    def _write_summary(self, path: Path, source: str = "measured") -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "dataset",
                    "query_id",
                    "method",
                    "median_s",
                    "q1_s",
                    "q3_s",
                    "source",
                    "signature",
                    "signature_verified",
                ],
            )
            writer.writeheader()
            for query_id in ("Q1", "Q2"):
                for dataset_index, dataset in enumerate(
                    ("drtest", "drtrain", "bdd100kA", "bdd100kB", "I-24")
                ):
                    for method_index, method in enumerate(METHODS):
                        value = 0.1 + dataset_index + method_index / 10.0
                        writer.writerow(
                            {
                                "dataset": dataset,
                                "query_id": query_id,
                                "method": method,
                                "median_s": value,
                                "q1_s": value * 0.95,
                                "q3_s": value * 1.05,
                                "source": source,
                                "signature": f"{dataset}-{query_id}",
                                "signature_verified": "true",
                            }
                        )
        with (path.parent / "raw_measurements.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "dataset",
                    "query_id",
                    "method",
                    "repetition",
                    "elapsed_s",
                    "signature",
                    "signature_verified",
                ],
            )
            writer.writeheader()
            for query_id in ("Q1", "Q2"):
                for dataset_index, dataset in enumerate(
                    ("drtest", "drtrain", "bdd100kA", "bdd100kB", "I-24")
                ):
                    for method_index, method in enumerate(METHODS):
                        value = 0.1 + dataset_index + method_index / 10.0
                        for repetition, factor in enumerate((0.9, 0.95, 1.0, 1.05, 1.1), 1):
                            writer.writerow(
                                {
                                    "dataset": dataset,
                                    "query_id": query_id,
                                    "method": method,
                                    "repetition": repetition,
                                    "elapsed_s": value * factor,
                                    "signature": f"{dataset}-{query_id}",
                                    "signature_verified": "true",
                                }
                            )
        (path.parent / "run_manifest.json").write_text(
            json.dumps(
                {
                    "complete": True,
                    "repetitions": 5,
                    "oracle_signatures": {
                        dataset: {query_id: f"{dataset}-{query_id}" for query_id in ("Q1", "Q2")}
                        for dataset in ("drtest", "drtrain", "bdd100kA", "bdd100kB", "I-24")
                    },
                }
            ),
            encoding="utf-8",
        )

    def test_renders_q1_and_q2_pdf_and_png(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = root / "summary.csv"
            self._write_summary(summary)

            outputs = render_figures(summary, root)

            self.assertEqual(len(outputs), 4)
            self.assertTrue(all(path.exists() and path.stat().st_size > 0 for path in outputs))
            self.assertTrue((root / "baseline_query_time_q1_corrected.pdf").exists())
            self.assertTrue((root / "baseline_query_time_q2_corrected.pdf").exists())

    def test_rejects_estimated_or_unverified_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = root / "summary.csv"
            self._write_summary(summary, source="estimated")

            with self.assertRaisesRegex(ValueError, "measured"):
                render_figures(summary, root)


if __name__ == "__main__":
    unittest.main()
