import unittest

from supplement import discretization_sensitivity


class DiscretizationSensitivityTests(unittest.TestCase):
    def test_parse_granularities_returns_expected_pairs(self):
        self.assertEqual(
            discretization_sensitivity.parse_granularities("6:4,8:6,10:8,12:10"),
            [(6, 4), (8, 6), (10, 8), (12, 10)],
        )

    def test_parse_granularities_rejects_invalid_tokens(self):
        with self.assertRaises(ValueError):
            discretization_sensitivity.parse_granularities("6-4")


if __name__ == "__main__":
    unittest.main()
