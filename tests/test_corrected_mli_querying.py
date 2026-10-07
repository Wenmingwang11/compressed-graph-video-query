import unittest

import igraph as ig
import pandas as pd

from supplement.corrected_query_model import QuerySpec, RelationConstraint, exact_query, result_signature
from vsimsearch.corrected_mli_querying import query_corrected_mli


def _graph(frames):
    graph = ig.Graph(directed=True)
    graph.add_vertices(1)
    graph.vs[0]["name"] = (10, 20)
    graph.vs[0]["combinations"] = {
        (10, 20): {(0, 2): set(frames)},
        (20, 10): {(10, 2): set(frames)},
    }
    return graph


class CorrectedMLIQueryingTests(unittest.TestCase):
    def setUp(self):
        rows = []
        for frame in (1, 2, 3, 5):
            rows.extend([(frame, 10, 0, 10.0, 50.0), (frame, 20, 1, 60.0, 50.0)])
        self.data = pd.DataFrame(rows, columns=["frame", "track_id", "class_id", "cx", "cy"])
        self.index = {
            1: {0: {10}, 1: {20}},
            2: {0: {10}, 1: {20}},
        }
        self.cp_graphs = {1: _graph({1, 2}), 2: _graph({3, 5})}
        self.query = QuerySpec(
            query_id="q1",
            role_types=(0, 1),
            relations={
                (0, 1): RelationConstraint(2, 2, frozenset({0})),
                (1, 0): RelationConstraint(2, 2, frozenset({10})),
            },
            min_consecutive_frames=3,
            topk=10,
        )

    def test_compressed_graph_candidates_match_oracle_across_windows(self):
        actual = query_corrected_mli(
            self.index,
            self.cp_graphs,
            self.data,
            self.query,
            frame_width=100,
            frame_height=100,
        )
        expected = exact_query(self.data, self.query, 100, 100)

        self.assertEqual(result_signature(actual), result_signature(expected))
        self.assertEqual((actual[0].start_frame, actual[0].end_frame), (1, 3))

    def test_missing_directed_relation_does_not_use_reverse_as_substitute(self):
        graph = _graph({1, 2, 3})
        graph.vs[0]["combinations"] = {(20, 10): {(10, 2): {1, 2, 3}}}

        actual = query_corrected_mli(
            {1: {0: {10}, 1: {20}}},
            {1: graph},
            self.data,
            self.query,
            100,
            100,
        )

        self.assertEqual(actual, [])

    def test_legacy_single_direction_index_transforms_reverse_angle(self):
        graph = _graph({1, 2, 3})
        graph.vs[0]["combinations"] = {(20, 10): {(10, 0): {1, 2, 3}}}

        actual = query_corrected_mli(
            {1: {0: {10}, 1: {20}}},
            {1: graph},
            self.data,
            self.query,
            100,
            100,
            legacy_single_direction=True,
            indexed_frame_width=1920,
            indexed_frame_height=1080,
        )

        self.assertEqual(result_signature(actual), result_signature(exact_query(self.data, self.query, 100, 100)))

    def test_lossless_window_edge_fallback_recovers_missing_compressed_relation(self):
        graph = _graph({1, 2, 3})
        graph.vs[0]["combinations"] = {}
        window_edges = {
            1: {
                (10, 20): {(0, 2): {1, 2, 3}},
                (20, 10): {(10, 2): {1, 2, 3}},
            }
        }

        actual = query_corrected_mli(
            {1: {0: {10}, 1: {20}}},
            {1: graph},
            self.data,
            self.query,
            100,
            100,
            window_edge_indexes=window_edges,
        )

        self.assertEqual(result_signature(actual), result_signature(exact_query(self.data, self.query, 100, 100)))


if __name__ == "__main__":
    unittest.main()
