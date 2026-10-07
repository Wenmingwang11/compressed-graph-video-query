import unittest

import igraph as ig

from vsimsearch.paper_querying_pairs import (
    match_bindings_in_window_with_pair_constraints,
    parse_pair_constraints,
)


class PaperQueryingPairsTests(unittest.TestCase):
    def test_parse_pair_constraints(self):
        constraints = parse_pair_constraints("0-1:1.0,5;0-2:0.5,2;1-2:2.0,8", role_count=3)
        self.assertEqual(
            constraints,
            {
                (0, 1): (1.0, 5.0),
                (0, 2): (0.5, 2.0),
                (1, 2): (2.0, 8.0),
            },
        )

    def test_parse_pair_constraints_requires_all_pairs(self):
        with self.assertRaises(ValueError):
            parse_pair_constraints("0-1:1.0,5;0-2:0.5,2", role_count=3)

    def test_matches_binding_when_each_role_pair_uses_its_own_threshold(self):
        graph = ig.Graph(directed=True)
        graph.add_vertices(2)
        graph.vs[0]["name"] = (1, 2)
        graph.vs[0]["combinations"] = {
            (1, 2): {(0.2, 0.3): {10, 11}},
        }
        graph.vs[1]["name"] = (1, 2, 3)
        graph.vs[1]["combinations"] = {
            (1, 3): {(0.4, 1.5): {10, 11}},
            (2, 3): {(1.5, 6.0): {10, 11}},
        }
        graph.add_edge(0, 1)

        window_data = {
            0: {1},
            1: {2},
            2: {3},
        }
        pair_constraints = {
            (0, 1): (1.0, 1.0),
            (0, 2): (0.5, 2.0),
            (1, 2): (2.0, 8.0),
        }

        results = match_bindings_in_window_with_pair_constraints(
            graph=graph,
            window_data=window_data,
            query_object_types=[0, 1, 2],
            pair_constraints=pair_constraints,
        )

        self.assertEqual(results, {(1, 2, 3): {10, 11}})

    def test_rejects_binding_when_one_role_pair_threshold_is_not_satisfied(self):
        graph = ig.Graph(directed=True)
        graph.add_vertices(2)
        graph.vs[0]["name"] = (1, 2)
        graph.vs[0]["combinations"] = {
            (1, 2): {(0.2, 0.3): {10, 11}},
        }
        graph.vs[1]["name"] = (1, 2, 3)
        graph.vs[1]["combinations"] = {
            (1, 3): {(0.4, 1.5): {10, 11}},
            (2, 3): {(1.5, 6.0): {10, 11}},
        }
        graph.add_edge(0, 1)

        window_data = {
            0: {1},
            1: {2},
            2: {3},
        }
        pair_constraints = {
            (0, 1): (1.0, 1.0),
            (0, 2): (0.3, 2.0),
            (1, 2): (2.0, 8.0),
        }

        results = match_bindings_in_window_with_pair_constraints(
            graph=graph,
            window_data=window_data,
            query_object_types=[0, 1, 2],
            pair_constraints=pair_constraints,
        )

        self.assertEqual(results, {})


if __name__ == "__main__":
    unittest.main()
