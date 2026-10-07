import unittest


from supplement.baseline_batch_benchmark import _spatial_constraint_scope


class BaselineBatchBenchmarkTests(unittest.TestCase):
    def test_spatial_constraint_scope_marks_full_coverage_constraints(self):
        self.assertEqual(
            _spatial_constraint_scope({(0, 1): (10.0, 8.0)}, 10, 8),
            "full_relation",
        )

    def test_spatial_constraint_scope_marks_restricted_constraints(self):
        self.assertEqual(
            _spatial_constraint_scope({(0, 1): (4.0, 3.0)}, 10, 8),
            "restricted_relation",
        )
