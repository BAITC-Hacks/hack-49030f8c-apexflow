"""Behavioural tests for UI helpers that do not need a Streamlit server."""

from __future__ import annotations

import unittest

from ui.data import ViewDataError, find_node, parse_gid, synthetic_view_data
from ui.graph import build_subgraph, render_svg


class UiDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = synthetic_view_data()

    def test_large_gid_is_not_rounded_or_converted_from_scientific_notation(self) -> None:
        gid = "9007199254740993"
        self.assertEqual(parse_gid(f"  {gid}  "), gid)
        self.assertEqual(find_node(self.data, gid)["gid"], gid)
        with self.assertRaises(ViewDataError):
            parse_gid("9.007199254740993e15")

    def test_search_uses_full_node_table_not_the_priority_queue(self) -> None:
        isolated_gid = "10000000000000007"
        self.assertNotIn(isolated_gid, set(self.data.top_nodes.head(3)["_gid_key"]))
        found = find_node(self.data, isolated_gid)
        self.assertIsNotNone(found)
        self.assertEqual(found["_gid_key"], isolated_gid)

    def test_isolate_is_a_valid_one_node_subgraph(self) -> None:
        isolated_gid = "10000000000000007"
        subset = build_subgraph(self.data.nodes_roles, self.data.edges, isolated_gid)
        self.assertEqual(len(subset.nodes), 1)
        self.assertTrue(subset.edges.empty)
        self.assertIn(isolated_gid, render_svg(subset, isolated_gid))

    def test_graph_edges_keep_source_to_destination_direction(self) -> None:
        selected_gid = "10000000000000002"
        subset = build_subgraph(self.data.nodes_roles, self.data.edges, selected_gid)
        self.assertTrue(((subset.edges["_src_key"] == selected_gid) | (subset.edges["_dst_key"] == selected_gid)).all())
        rendered = render_svg(subset, selected_gid)
        self.assertIn('marker-end="url(#arrow)"', rendered)
        self.assertIn("9007199254740993 → 10000000000000002", rendered)


if __name__ == "__main__":
    unittest.main()
