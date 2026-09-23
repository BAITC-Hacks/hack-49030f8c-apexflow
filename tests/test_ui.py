"""Behavioural tests for UI helpers that do not need a Streamlit server."""

from __future__ import annotations

import unittest
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from ui.data import ViewDataError, find_node, parse_gid, synthetic_view_data, load_view_data, file_signature
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


@pytest.fixture
def disk_sample(tmp_path):
    data = synthetic_view_data()
    data_dir, output = tmp_path / "data", tmp_path / "output"
    data_dir.mkdir()
    output.mkdir()
    nodes = data.nodes[["gid", "depth", "is_seed"]].copy()
    nodes["gid"] = nodes["gid"].map(int).astype("int64")
    edges = data.edges[["src", "dst", "sum_kzt", "n_tx", "depth"]].copy()
    for column in ("src", "dst"):
        edges[column] = edges[column].map(int).astype("int64")
    nodes.to_parquet(data_dir / "nodes.parquet", index=False)
    edges.to_parquet(data_dir / "edges.parquet", index=False)
    for filename, payload in data.raw_csv.items():
        (output / filename).write_bytes(payload)
    return data_dir, output


def test_real_parquet_csv_path_preserves_large_ids_and_original_downloads(disk_sample):
    data_dir, output = disk_sample
    loaded = load_view_data(data_dir, output)
    assert find_node(loaded, "9007199254740993")["gid"] == "9007199254740993"
    assert loaded.edges.iloc[0]["src"] == "9007199254740993"
    for filename, payload in loaded.raw_csv.items():
        assert payload == (output / filename).read_bytes()
    top_gids = json.loads(pd.read_csv(output / "clusters.csv").iloc[0]["top_gids"])
    assert all(type(gid) is int for gid in top_gids)


@pytest.mark.parametrize("value", [True, 1.0, "1e3", str(2**63), str(-(2**63)-1), [], "9"*5000, "１２"])
def test_search_rejects_non_int64(value):
    with pytest.raises(ViewDataError):
        parse_gid(value)


@pytest.mark.parametrize("table,column,value", [
    ("nodes_roles", "evidence", ""),
    ("nodes_roles", "evidence", "я"*201),
    ("nodes_roles", "role_score", float("inf")),
    ("nodes_roles", "cluster_id", "1.0"),
    ("top_nodes", "priority_score", 0),
    ("top_nodes", "why", ""),
    ("clusters", "sum_kzt_internal", 1),
    ("clusters", "top_gids", '["9007199254740993"]'),
    ("clusters", "top_gids", '[true]'),
    ("clusters", "top_gids", '[{}]'),
    ("clusters", "hypothesis", ""),
])
def test_ui_rejects_corrupt_results(disk_sample, table, column, value):
    data_dir, output = disk_sample
    path = output / f"{table}.csv"
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame.loc[0, column] = str(value)
    frame.to_csv(path, index=False)
    with pytest.raises(ViewDataError):
        load_view_data(data_dir, output)


def test_strings_are_not_treated_as_true_seed(disk_sample):
    data_dir, output = disk_sample
    path = data_dir / "nodes.parquet"
    frame = pd.read_parquet(path)
    frame["is_seed"] = "False"
    frame.to_parquet(path, index=False)
    with pytest.raises(ViewDataError, match="bool"):
        load_view_data(data_dir, output)


def test_changed_files_abort_snapshot(disk_sample):
    data_dir, output = disk_sample
    before = file_signature(data_dir, output)
    with patch("ui.data.file_signature", side_effect=[before, before + (("changed", 0, 0),)]):
        with pytest.raises(ViewDataError, match="изменились"):
            load_view_data(data_dir, output)


def test_refresh_signature_and_content_change(disk_sample):
    data_dir, output = disk_sample
    before = file_signature(data_dir, output)
    path = output / "nodes_roles.csv"
    frame = pd.read_csv(path, dtype=str)
    frame.loc[0, "evidence"] = "Обновлено: 1 наблюдаемый перевод."
    frame.to_csv(path, index=False)
    assert before != file_signature(data_dir, output)
    assert load_view_data(data_dir, output).nodes_roles.iloc[0]["evidence"] == frame.loc[0, "evidence"]


def test_graph_keeps_cluster_isolates_and_reports_limits():
    data = synthetic_view_data()
    roles = data.nodes_roles.copy()
    roles["_cluster_key"] = "1"
    graph = build_subgraph(roles, data.edges, "10000000000000002", mode="cluster")
    assert "10000000000000007" in set(graph.nodes["_gid_key"])
    limited = build_subgraph(roles, data.edges, "10000000000000007", mode="cluster", max_edges=1, max_nodes=3)
    assert "10000000000000007" in set(limited.nodes["_gid_key"])
    assert limited.hidden_nodes == 7 - len(limited.nodes)
    assert limited.hidden_edges == 6 - len(limited.edges)


def test_reciprocal_edges_and_self_loop_have_visible_paths():
    data = synthetic_view_data()
    edges = data.edges.copy()
    loop = edges.iloc[[0]].copy()
    loop["src"] = loop["dst"] = loop["_src_key"] = loop["_dst_key"] = "10000000000000002"
    edges = pd.concat([edges, loop], ignore_index=True)
    svg = render_svg(build_subgraph(data.nodes_roles, edges, "10000000000000002"), "10000000000000002")
    assert ' Q ' in svg  # reciprocal curves
    assert ' C ' in svg  # nondegenerate self loop
    assert svg.count('marker-end=') == 6
    assert "Признаки консолидации" in svg


def test_streamlit_search_survives_filter_and_missing_files(tmp_path, monkeypatch):
    monkeypatch.setenv("APEXFLOW_DATA_DIR", str(tmp_path / "absent"))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=20)
    assert not app.exception
    assert len(app.error) == 1
    app.toggle[0].set_value(True).run(timeout=20)
    assert not app.exception
    role = next(element for element in app.selectbox if element.label == "Роль в таблице топа")
    role.set_value("consolidator").run()
    app.text_input[0].set_value("10000000000000007")
    next(button for button in app.button if button.label == "Найти клиента").click().run()
    assert not app.exception
    assert app.session_state["selected_gid"] == "10000000000000007"
    assert any("в показанной части" in item.value for item in app.info)
    assert any("Наблюдаемых связей нет" in item.value for item in app.info)
    app.text_input[0].set_value("1e6")
    next(button for button in app.button if button.label == "Найти клиента").click().run()
    assert not app.exception
    assert len(app.error) == 1
