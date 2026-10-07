import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from supplement.corrected_query_model import (
    QuerySpec,
    RelationConstraint,
    exact_query,
    result_signature,
)
from supplement.corrected_baseline_strategies import STRATEGY_NAMES, build_strategy


def _two_role_data():
    rows = []
    for frame in (1, 2, 4, 5, 6):
        rows.extend(
            [
                (frame, 10, 1, 10.0, 50.0),
                (frame, 20, 2, 60.0, 50.0),
                (frame, 21, 2, 10.0, 90.0),
            ]
        )
    return pd.DataFrame(rows, columns=["frame", "track_id", "class_id", "cx", "cy"])


def _three_role_repeated_type_data():
    rows = []
    for frame in (10, 11, 12, 14, 15):
        rows.extend(
            [
                (frame, 30, 3, 10.0, 50.0),
                (frame, 31, 3, 40.0, 50.0),
                (frame, 40, 4, 70.0, 50.0),
            ]
        )
    # A single class-3 track must not satisfy the repeated-class count bound.
    rows.extend([(13, 30, 3, 10.0, 50.0), (13, 40, 4, 70.0, 50.0)])
    return pd.DataFrame(rows, columns=["frame", "track_id", "class_id", "cx", "cy"])


class CorrectedBaselineStrategyTests(unittest.TestCase):
    def test_strategies_share_the_input_table_without_copying_it(self):
        data = _two_role_data()

        for name in STRATEGY_NAMES:
            with self.subTest(strategy=name):
                strategy = build_strategy(name, data, 100, 100)
                self.assertIs(strategy._data, data)
                strategy.release()

    def test_vocal_registry_is_directly_partitioned_by_class_and_frame(self):
        strategy = build_strategy("vocal-udf", _two_role_data(), 100, 100)

        self.assertEqual(
            strategy.artifact.predicates[2],
            {
                1: (20, 21),
                2: (20, 21),
                4: (20, 21),
                5: (20, 21),
                6: (20, 21),
            },
        )

    def test_star_uses_compact_quantized_arrays_without_coordinate_objects(self):
        strategy = build_strategy("star", _two_role_data(), 100, 100)

        self.assertTrue(strategy.artifact.edge_postings)
        for posting in strategy.artifact.edge_postings.values():
            self.assertIsInstance(posting, np.ndarray)
            self.assertEqual(posting.dtype.names, ("frame", "sid", "tid", "theta", "d"))
        self.assertFalse(posting.dtype.hasobject)

    def test_star_can_limit_offline_postings_to_workload_type_pairs(self):
        strategy = build_strategy(
            "star",
            _two_role_data(),
            100,
            100,
            allowed_type_pairs={(1, 2)},
        )

        self.assertEqual(set(strategy.artifact.edge_postings), {(1, 2)})

        query = QuerySpec(
            query_id="default-quantization",
            role_types=(1, 2),
            relations={(0, 1): RelationConstraint(2, 2, frozenset({0}))},
            min_consecutive_frames=3,
            topk=10,
        )
        with patch(
            "supplement.corrected_baseline_strategies.quantize_relation",
            side_effect=AssertionError("STAR query must use its prequantized edge arrays"),
        ):
            strategy.query(query)

    def test_star_nondefault_quantization_conservatively_uses_type_frames(self):
        data = _two_role_data()
        strategy = build_strategy("star", data, 100, 100)
        query = QuerySpec(
            query_id="nondefault-quantization",
            role_types=(1, 2),
            relations={(0, 1): RelationConstraint(0, 8, frozenset({11}))},
            min_consecutive_frames=3,
            topk=10,
            theta_parts=12,
            distance_parts=9,
        )

        with patch(
            "supplement.corrected_baseline_strategies.exact_query",
            wraps=exact_query,
        ) as verifier:
            actual = strategy.query(query)

        self.assertEqual(result_signature(actual), result_signature(exact_query(data, query, 100, 100)))
        self.assertEqual(verifier.call_args.kwargs["candidate_frames"], {1, 2, 4, 5, 6})

    def test_star_default_prequantization_matches_signed_oracle_bins(self):
        data = _two_role_data()
        query = QuerySpec(
            query_id="signed-default-bin",
            role_types=(1, 2),
            relations={(1, 0): RelationConstraint(1, 1, frozenset({-5}))},
            min_consecutive_frames=3,
            topk=10,
        )

        oracle = exact_query(data, query, 100, 100)
        actual = build_strategy("star", data, 100, 100).query(query)

        self.assertEqual([(item.binding, item.start_frame, item.end_frame) for item in oracle], [((10, 21), 4, 6)])
        self.assertEqual(result_signature(actual), result_signature(oracle))

    def test_release_drops_large_references_and_is_idempotent(self):
        strategy = build_strategy("star", _two_role_data(), 100, 100)

        strategy.release()
        strategy.release()

        self.assertIsNone(strategy.artifact)
        self.assertIsNone(strategy._data)
        query = QuerySpec("released", (1,), {}, 1, 1)
        with self.assertRaisesRegex(RuntimeError, "released"):
            strategy.query(query)

    def test_builds_seven_distinct_query_independent_artifact_kinds(self):
        data = _two_role_data()

        strategies = [build_strategy(name, data, 100, 100) for name in STRATEGY_NAMES]

        self.assertEqual(
            tuple(STRATEGY_NAMES),
            ("vqpy", "eva", "vocal-udf", "video-colbert", "seiden", "lava", "star"),
        )
        self.assertEqual(len({strategy.artifact_kind for strategy in strategies}), 7)
        self.assertTrue(all(strategy.artifact is not None for strategy in strategies))

    def test_every_strategy_matches_oracle_for_two_roles_and_calls_exact_query(self):
        data = _two_role_data()
        query = QuerySpec(
            query_id="two-role",
            role_types=(1, 2),
            relations={(0, 1): RelationConstraint(2, 2, frozenset({0}))},
            min_consecutive_frames=3,
            topk=10,
        )
        oracle_signature = result_signature(exact_query(data, query, 100, 100))

        for name in STRATEGY_NAMES:
            with self.subTest(strategy=name):
                strategy = build_strategy(name, data, 100, 100)
                with patch(
                    "supplement.corrected_baseline_strategies.exact_query",
                    wraps=exact_query,
                ) as verifier:
                    results = strategy.query(query)
                self.assertEqual(result_signature(results), oracle_signature)
                verifier.assert_called_once()

    def test_every_strategy_matches_oracle_for_three_roles_with_repeated_type(self):
        data = _three_role_repeated_type_data()
        query = QuerySpec(
            query_id="three-role-repeated",
            role_types=(3, 3, 4),
            relations={
                (0, 1): RelationConstraint(1, 1, frozenset({0})),
                (1, 2): RelationConstraint(1, 1, frozenset({0})),
            },
            min_consecutive_frames=3,
            topk=10,
        )
        oracle = exact_query(data, query, 100, 100)

        self.assertEqual([(item.start_frame, item.end_frame) for item in oracle], [(10, 12)])
        for name in STRATEGY_NAMES:
            with self.subTest(strategy=name):
                actual = build_strategy(name, data, 100, 100).query(query)
                self.assertEqual(result_signature(actual), result_signature(oracle))

    def test_unknown_strategy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown corrected baseline strategy"):
            build_strategy("not-a-baseline", _two_role_data(), 100, 100)


if __name__ == "__main__":
    unittest.main()
