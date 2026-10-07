import unittest

import igraph as ig

from vsimsearch.paper_querying import _filter_window_ids_by_types, match_bindings_in_window


class PaperQueryingTests(unittest.TestCase):
    def test_type_filter_respects_repeated_role_counts(self):
        index = {
            1: {0: {1, 2}, 1: {3}},
            2: {0: {4}, 1: {5}},
        }

        self.assertEqual(_filter_window_ids_by_types(index, [0, 1, 0]), [1])

    def _build_graph_with_predecessor_pair(self):
        graph = ig.Graph(directed=True)
        graph.add_vertices(2)
        graph.vs[0]["name"] = (1, 2)
        graph.vs[0]["combinations"] = {
            (1, 2): {(0.2, 0.3): {10, 11}},
        }
        graph.vs[1]["name"] = (1, 2, 3)
        graph.vs[1]["combinations"] = {
            (1, 3): {(0.2, 0.3): {10, 11}},
            (2, 3): {(0.2, 0.3): {10, 11}},
        }
        graph.add_edge(0, 1)
        return graph

    def test_matches_relations_from_current_vertex_and_predecessor_chain(self):
        graph = self._build_graph_with_predecessor_pair()
        window_data = {
            0: {1},
            1: {2},
            2: {3},
        }

        results = match_bindings_in_window(
            graph=graph,
            window_data=window_data,
            query_object_types=[0, 1, 2],
            theta_d_condition=(1.0, 1.0),
        )

        self.assertEqual(results, {(1, 2, 3): {10, 11}})

    def test_does_not_fallback_to_window_edge_index_when_graph_lacks_pair(self):
        graph = ig.Graph(directed=True)
        graph.add_vertices(1)
        graph.vs[0]["name"] = (1, 2)
        graph.vs[0]["combinations"] = {}
        window_data = {
            0: {1},
            1: {2},
        }

        results = match_bindings_in_window(
            graph=graph,
            window_data=window_data,
            query_object_types=[0, 1],
            theta_d_condition=(1.0, 1.0),
        )

        self.assertEqual(results, {})


if __name__ == "__main__":
    unittest.main()
