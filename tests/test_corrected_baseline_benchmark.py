import unittest

from supplement.corrected_query_model import QueryResult, result_signature
from supplement.run_corrected_baseline_benchmark import measure_query, summarize_measurements


class _FakeStrategy:
    def __init__(self, results):
        self.results = results
        self.calls = 0

    def query(self, query):
        self.calls += 1
        return list(self.results)


class _Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        current = self.value
        self.value += 0.25
        return current


class CorrectedBaselineBenchmarkTests(unittest.TestCase):
    def test_excludes_warmup_and_records_five_measured_repetitions(self):
        results = [QueryResult((10, 20), 1, 10)]
        strategy = _FakeStrategy(results)

        rows = measure_query(
            strategy,
            query=object(),
            expected_signature=result_signature(results),
            repetitions=5,
            warmups=1,
            clock=_Clock(),
        )

        self.assertEqual(strategy.calls, 6)
        self.assertEqual([row["repetition"] for row in rows], [1, 2, 3, 4, 5])
        self.assertEqual([row["elapsed_s"] for row in rows], [0.25] * 5)
        self.assertTrue(all(row["signature_verified"] for row in rows))

    def test_rejects_signature_mismatch(self):
        strategy = _FakeStrategy([QueryResult((99, 100), 1, 10)])

        with self.assertRaisesRegex(RuntimeError, "signature mismatch"):
            measure_query(
                strategy,
                query=object(),
                expected_signature=result_signature([QueryResult((10, 20), 1, 10)]),
                repetitions=1,
                warmups=0,
                clock=_Clock(),
            )

    def test_summary_uses_median_and_quartiles(self):
        rows = [
            {
                "dataset": "drtest",
                "query_id": "Q1",
                "method": "Ours",
                "elapsed_s": value,
                "signature": "abc",
                "signature_verified": True,
            }
            for value in (1.0, 2.0, 3.0, 4.0, 100.0)
        ]

        summary = summarize_measurements(rows)

        self.assertEqual(summary[0]["median_s"], 3.0)
        self.assertEqual(summary[0]["q1_s"], 2.0)
        self.assertEqual(summary[0]["q3_s"], 4.0)
        self.assertEqual(summary[0]["source"], "measured")


if __name__ == "__main__":
    unittest.main()
