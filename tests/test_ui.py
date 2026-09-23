"""Behavioural tests for UI helpers that do not need a Streamlit server."""

from __future__ import annotations

import unittest
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from apexflow.io import INPUT_FILES, OUTPUT_COLUMNS, file_hashes, write_run_manifest

from ui.data import ViewDataError, find_node, parse_gid, synthetic_view_data, load_view_data, file_signature
from ui.graph import build_subgraph, render_svg, render_legend


def _complete_manifest(data_dir, output):
    write_run_manifest(output, {
        "schema_version": 1, "status": "complete",
        "inputs": file_hashes(data_dir, INPUT_FILES.values()),
        "outputs": file_hashes(output, [f"{name}.csv" for name in OUTPUT_COLUMNS]),
    })


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
    pd.DataFrame([
        {"src": int(row.src), "dst": int(row.dst), "sum_kzt": row.sum_kzt / row.n_tx,
         "date": pd.Timestamp("2026-07-01")}
        for row in edges.itertuples() for _ in range(row.n_tx)
    ]).to_parquet(data_dir / "transactions.parquet", index=False)
    for filename, payload in data.raw_csv.items():
        (output / filename).write_bytes(payload)
    _complete_manifest(data_dir, output)
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
    _complete_manifest(data_dir, output)
    with pytest.raises(ViewDataError):
        load_view_data(data_dir, output)


def test_strings_are_not_treated_as_true_seed(disk_sample):
    data_dir, output = disk_sample
    path = data_dir / "nodes.parquet"
    frame = pd.read_parquet(path)
    frame["is_seed"] = "False"
    frame.to_parquet(path, index=False)
    _complete_manifest(data_dir, output)
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
    _complete_manifest(data_dir, output)
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


def test_limited_neighborhood_does_not_make_hidden_neighbors_look_isolated():
    data = synthetic_view_data()
    gid = "10000000000000002"
    subset = build_subgraph(data.nodes_roles, data.edges, gid, max_edges=1)
    endpoints = set(subset.edges["_src_key"]) | set(subset.edges["_dst_key"])
    assert set(subset.nodes["_gid_key"]) == endpoints | {gid}
    all_neighbors = data.edges.loc[(data.edges["_src_key"] == gid) | (data.edges["_dst_key"] == gid)]
    all_keys = set(all_neighbors["_src_key"]) | set(all_neighbors["_dst_key"])
    assert subset.hidden_nodes == len(all_keys - endpoints)
    assert subset.hidden_nodes > 0


def test_dense_neighborhood_has_centered_selection_separate_nodes_and_short_labels():
    data = synthetic_view_data()
    template = data.nodes_roles.iloc[0].to_dict()
    gids = [str(100000000000000000 + index) for index in range(40)]
    nodes = pd.DataFrame([{**template, "gid": gid, "_gid_key": gid, "is_seed": True} for gid in gids])
    edges = pd.DataFrame([
        {"src": gids[0], "dst": gid, "_src_key": gids[0], "_dst_key": gid, "sum_kzt": 1, "n_tx": 1}
        for gid in gids[1:]
    ])
    subset = build_subgraph(nodes, edges, gids[0], max_edges=40)
    root = ET.fromstring(render_svg(subset, gids[0]))
    ns = {"svg": "http://www.w3.org/2000/svg"}
    groups = root.findall("svg:g", ns)
    assert len(groups) == 40
    geometry = []
    for index, group in enumerate(groups, start=1):
        circle = group.find("svg:circle", ns)
        x, y, radius = (float(circle.attrib[key]) for key in ("cx", "cy", "r"))
        geometry.append((x, y, radius))
        assert group.find("svg:text", ns).text == f"№{index}"
        assert f"gid: {group.attrib['data-gid']}" in group.find("svg:title", ns).text
        if group.attrib["data-gid"] == gids[0]:
            assert (x, y) == (450, 260)
            assert circle.attrib["stroke-width"] == "5"
        assert radius < 15  # The former radius29 overlaps in the dense graph.
    for index, (x, y, radius) in enumerate(geometry):
        for other_x, other_y, other_radius in geometry[index + 1:]:
            assert math.hypot(x - other_x, y - other_y) > radius + other_radius + 16
    assert "не gid" in render_legend(subset, "role")
    assert root.findall("svg:path", ns)[0].attrib["data-src"] == gids[0]


def test_streamlit_search_survives_filter_and_missing_files(tmp_path, monkeypatch):
    monkeypatch.setenv("APEXFLOW_DATA_DIR", str(tmp_path / "absent"))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=20)
    assert not app.exception
    assert len(app.error) == 1
    assert any("исходные Parquet" in item.value for item in app.info)
    assert not any("интегрированный модуль аналитики" in item.value for item in app.info)
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


@pytest.mark.parametrize("filename", ["transactions.parquet", "nodes_roles.csv"])
def test_ui_rejects_modified_snapshot_even_if_schema_is_valid(disk_sample, filename):
    data_dir, output = disk_sample
    if filename.endswith("parquet"):
        path = data_dir / filename
        frame = pd.read_parquet(path)
        frame["date"] += pd.Timedelta(days=1)
        frame.to_parquet(path, index=False)
    else:
        path = output / filename
        frame = pd.read_csv(path, dtype=str)
        frame.loc[0, "evidence"] = "Правдоподобное, но чужое объяснение."
        frame.to_csv(path, index=False)
    with pytest.raises(ViewDataError, match="изменились"):
        load_view_data(data_dir, output)


def test_streamlit_rejects_failed_run_after_successful_cached_load(disk_sample, monkeypatch):
    data_dir, output = disk_sample
    monkeypatch.setenv("APEXFLOW_DATA_DIR", str(data_dir))
    monkeypatch.setenv("APEXFLOW_OUTPUT_DIR", str(output))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=20)
    assert not app.exception and not app.error
    assert any("01.07.2026 — 01.07.2026" in item.value for item in app.caption)
    write_run_manifest(output, {"schema_version": 1, "status": "failed"})
    app.run()
    assert not app.exception
    assert len(app.error) == 1
    assert not app.dataframe  # Previous successful results are no longer shown.


def test_streamlit_cluster_card_follows_search_and_boundary_warning(tmp_path, monkeypatch):
    monkeypatch.setenv("APEXFLOW_DATA_DIR", str(tmp_path / "absent"))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=20)
    app.toggle[0].set_value(True).run()
    app.text_input[0].set_value("10000000000000007")
    next(button for button in app.button if button.label == "Найти клиента").click().run()
    assert not app.exception
    assert any("Кластер выбранного клиента: 2" in item.value for item in app.markdown)
    assert next(item for item in app.metric if item.label == "Узлов в выбранном кластере").value == "1"
    boundary = synthetic_view_data().nodes_roles.query("depth == 4").iloc[0]["gid"]
    app.text_input[0].set_value(boundary)
    next(button for button in app.button if button.label == "Найти клиента").click().run()
    assert not app.exception
    assert any("четвёртом колене" in item.value for item in app.warning)
    assert any("Кластер выбранного клиента: 1" in item.value for item in app.markdown)


def test_card_large_int64_money_does_not_wrap_negative():
    data = synthetic_view_data()
    gid = "10000000000000002"
    data.edges["sum_kzt"] = 0
    inbound = data.edges.index[data.edges["_dst_key"] == gid][:2]
    assert len(inbound) == 2
    data.edges.loc[inbound, "sum_kzt"] = 2**62
    from app import _render_node_card
    with patch("app.st") as display:
        display.columns.return_value = [display, display, display]
        _render_node_card(data, find_node(data, gid))
    observed = next(call.args[0] for call in display.markdown.call_args_list if "Наблюдаемые связи" in call.args[0])
    assert "9\u202f223\u202f372\u202f036\u202f854\u202f775\u202f808.00 KZT" in observed
    assert "-9" not in observed


def test_demo_examples_are_selected_from_current_data_and_keep_exact_edges():
    from ui.demo_examples import select_examples
    data = synthetic_view_data()
    examples = select_examples(data)
    assert len(examples) == 3
    assert not examples[0]["is_seed"]
    assert examples[0]["role"] != examples[1]["role"]
    assert examples[2]["depth"] == 4
    for example in examples:
        assert example["evidence"] == find_node(data, example["gid"])["evidence"]
        edge = example["edge_to_verify"]
        if edge is not None:
            observed = data.edges.loc[(data.edges["src"] == edge["src"]) & (data.edges["dst"] == edge["dst"])].iloc[0]
            assert edge["sum_kzt"] == observed["sum_kzt"]
            assert edge["n_tx"] == observed["n_tx"]
