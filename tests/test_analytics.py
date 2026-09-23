from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from apexflow.analytics import (  # noqa: E402
    CLUSTERS_COLUMNS,
    NODES_ROLES_COLUMNS,
    TOP_NODES_COLUMNS,
    analyze,
    assign_clusters,
)
from apexflow.features import add_cluster_context, build_graph, compute_features  # noqa: E402
from apexflow.rules import assign_roles, derive_thresholds, make_evidence, make_priority_why  # noqa: E402


def _transactions() -> pd.DataFrame:
    return pd.DataFrame(columns=["src", "dst", "date", "sum_kzt"])


def _frames(node_rows, edge_rows):
    nodes = pd.DataFrame(node_rows, columns=["gid", "depth", "is_seed"]).astype(
        {"gid": "int64", "depth": "int64", "is_seed": "bool"}
    )
    edges = pd.DataFrame(edge_rows, columns=["src", "dst", "sum_kzt", "n_tx"]).astype(
        {"src": "int64", "dst": "int64", "sum_kzt": "float64", "n_tx": "int64"}
    )
    return nodes, edges


def _assert_contract(result, nodes, edges):
    roles, clusters, top = (result[key] for key in ("nodes_roles", "clusters", "top_nodes"))
    assert set(result) == {"nodes_roles", "clusters", "top_nodes"}
    assert roles.columns.tolist() == NODES_ROLES_COLUMNS
    assert clusters.columns.tolist() == CLUSTERS_COLUMNS
    assert top.columns.tolist() == TOP_NODES_COLUMNS
    assert roles.gid.tolist() == sorted(nodes.gid.tolist())
    assert not roles.gid.duplicated().any()
    assert roles.gid.dtype == top.gid.dtype == "int64"
    assert roles.cluster_id.dtype == clusters.cluster_id.dtype == "int64"
    assert clusters.cluster_id.is_unique
    assert clusters.cluster_id.tolist() == sorted(clusters.cluster_id.tolist())
    for frame in result.values():
        assert not frame.isna().any().any()
    for column in ("role_score", "priority_score"):
        assert roles[column].map(math.isfinite).all()
        assert roles[column].between(0, 1).all()
    assert roles.evidence.str.len().between(1, 200).all()
    assert top.why.str.len().gt(0).all()
    assert len(top) == min(20, len(nodes))
    assert top["rank"].tolist() == list(range(1, len(top) + 1))
    expected_top = roles.sort_values(["priority_score", "gid"], ascending=[False, True]).head(20)
    assert_frame_equal(
        top[["gid", "role", "priority_score"]],
        expected_top[["gid", "role", "priority_score"]].reset_index(drop=True),
    )
    mapping = roles.set_index("gid").cluster_id
    assert set(mapping) == set(clusters.cluster_id)
    for cluster in clusters.itertuples(index=False):
        group = roles.loc[roles.cluster_id.eq(cluster.cluster_id)]
        assert cluster.n_nodes == len(group)
        assert cluster.n_seed == nodes.loc[nodes.gid.isin(group.gid), "is_seed"].sum()
        internal = edges.src.isin(group.gid) & edges.dst.isin(group.gid)
        assert cluster.sum_kzt_internal == pytest.approx(math.fsum(edges.loc[internal, "sum_kzt"]))
        expected = group.sort_values(["priority_score", "gid"], ascending=[False, True]).head(5).gid.tolist()
        assert json.loads(cluster.top_gids) == expected


def test_analyze_contract_is_deterministic_and_complete() -> None:
    nodes = pd.DataFrame(
        {
            "gid": list(range(1, 23)),
            "depth": [0] + [1] * 21,
            "is_seed": [True] + [False] * 21,
        }
    )
    edges = pd.DataFrame(
        {
            "src": list(range(1, 22)),
            "dst": list(range(2, 23)),
            "sum_kzt": [10_000 + value * 1_000 for value in range(21)],
            "n_tx": [1] * 21,
        }
    )

    first = analyze(nodes, edges, _transactions())
    second = analyze(nodes.sample(frac=1, random_state=7), edges.sample(frac=1, random_state=9), _transactions())
    _assert_contract(first, nodes, edges)

    assert list(first["nodes_roles"].columns) == NODES_ROLES_COLUMNS
    assert list(first["clusters"].columns) == CLUSTERS_COLUMNS
    assert list(first["top_nodes"].columns) == TOP_NODES_COLUMNS
    assert first["nodes_roles"]["gid"].tolist() == list(range(1, 23))
    assert len(first["top_nodes"]) == 20
    assert first["top_nodes"]["rank"].tolist() == list(range(1, 21))
    assert first["nodes_roles"]["role"].isin(
        {"consolidator", "transit", "distributor", "terminal", "coordinator", "peripheral"}
    ).all()
    assert first["nodes_roles"]["role_score"].between(0, 1).all()
    assert first["nodes_roles"]["priority_score"].between(0, 1).all()
    assert first["nodes_roles"]["evidence"].str.len().between(1, 200).all()
    assert first["clusters"]["top_gids"].map(json.loads).map(len).le(5).all()

    for name in first:
        assert_frame_equal(first[name], second[name])


def test_depth_four_never_receives_terminal_role() -> None:
    nodes = pd.DataFrame(
        {
            "gid": [1, 2, 3, 4, 5, 99, 100],
            "depth": [0, 0, 1, 2, 3, 4, 3],
            "is_seed": [True, True, False, False, False, False, False],
        }
    )
    edges = pd.DataFrame(
        {
            "src": [1, 99, 2, 3, 4],
            "dst": [99, 3, 100, 4, 5],
            "sum_kzt": [100_000, 100_000, 100_000, 100, 100],
            "n_tx": [1, 1, 1, 1, 1],
        }
    )

    roles = analyze(nodes, edges, _transactions())["nodes_roles"].set_index("gid")

    assert roles.loc[99, "role"] != "terminal"
    assert roles.loc[99, "role"] != "transit"
    assert roles.loc[100, "role"] == "terminal"
    assert "Depth=4" in roles.loc[99, "evidence"]


@pytest.mark.parametrize("count", [0, 1, 4])
def test_empty_or_edgeless_graph_keeps_every_node(count):
    nodes, edges = _frames([(i, 0, True) for i in range(count)], [])
    result = analyze(nodes, edges, _transactions())
    _assert_contract(result, nodes, edges)
    assert result["nodes_roles"].role.eq("peripheral").all()
    assert result["nodes_roles"].priority_score.eq(0).all()
    assert len(result["clusters"]) == count


@pytest.mark.parametrize("zero_money", [False, True])
def test_self_transfers_do_not_prove_transit_and_are_not_hidden(zero_money):
    nodes, edges = _frames([(1, 1, False)], [(1, 1, 0 if zero_money else 1000, 2)])
    features = compute_features(nodes, edges).iloc[0]
    assert not features.is_isolated
    assert features.in_degree == features.out_degree == 0
    result = analyze(nodes, edges, _transactions())
    _assert_contract(result, nodes, edges)
    row = result["nodes_roles"].iloc[0]
    assert row.role == "peripheral"
    assert "Самопереводы" in row.evidence
    assert "нет входящих" not in row.evidence
    assert "Изолированный" not in result["clusters"].iloc[0].hypothesis


def test_zero_weight_projection_does_not_crash_or_invent_monetary_communities():
    nodes, edges = _frames([(1, 0, True), (2, 1, False), (3, 2, False)], [(1, 2, 0, 1), (2, 3, 0, 1)])
    result = analyze(nodes, edges, _transactions())
    _assert_contract(result, nodes, edges)
    assert len(result["clusters"]) == 3
    assert result["nodes_roles"].role.eq("peripheral").all()
    features = compute_features(nodes, edges)
    assert features.in_sum.eq(0).all()
    assert features.flow_ratio.isna().all()
    assert features.n_seed_reachable.tolist() == [0, 1, 1]


def test_features_preserve_direction_cycles_seed_counts_and_node_metadata():
    nodes, edges = _frames(
        [(1, 0, True), (2, 0, True), (3, 1, False), (4, 0, True)],
        [(1, 3, 100, 2), (2, 3, 200, 3), (3, 1, 80, 1)],
    )
    graph = build_graph(nodes, edges)
    assert graph.nodes[3] == {"depth": 1, "is_seed": False}
    assert graph[1][3] == {"sum_kzt": 100, "n_tx": 2}
    assert not graph.has_edge(3, 2)
    features = compute_features(nodes, edges).set_index("gid")
    assert features.n_seed_reachable.to_dict() == {1: 1, 2: 0, 3: 2, 4: 0}
    assert features.loc[3, ["in_degree", "out_degree", "unique_neighbor_count"]].tolist() == [2, 1, 2]
    assert features.loc[3, ["in_sum", "out_sum", "in_n_tx", "out_n_tx", "observed_net"]].tolist() == [300, 80, 5, 1, 220]
    assert features.loc[3, "flow_ratio"] == pytest.approx(80 / 300)
    assert math.isnan(features.loc[2, "flow_ratio"])
    assert features.loc[4, "is_isolated"]


@pytest.mark.parametrize("direction,role", [("in", "consolidator"), ("out", "distributor")])
def test_fan_direction_selects_the_explained_role(direction, role):
    rows = [(i, 9, 100, 1) if direction == "in" else (9, i, 100, 1) for i in range(1, 7)]
    nodes, edges = _frames([(i, 1, False) for i in [*range(1, 7), 9]], rows)
    result = analyze(nodes, edges, _transactions())
    _assert_contract(result, nodes, edges)
    assert result["nodes_roles"].set_index("gid").loc[9, "role"] == role


@pytest.mark.parametrize("direction", ["in", "out"])
def test_zero_value_fan_and_large_self_loop_do_not_prove_a_money_flow_role(direction):
    rows = [(i, 9, 0, 1) if direction == "in" else (9, i, 0, 1) for i in range(1, 7)]
    nodes, edges = _frames(
        [(i, 1, False) for i in [*range(1, 7), 9]],
        rows + [(9, 9, 1_000_000, 1)],
    )
    result = analyze(nodes, edges, _transactions())
    _assert_contract(result, nodes, edges)
    row = result["nodes_roles"].set_index("gid").loc[9]
    assert row["role"] == "peripheral"
    assert row["role_score"] == 0


@pytest.mark.parametrize("direction,role", [("in", "consolidator"), ("out", "distributor")])
def test_large_self_loop_does_not_inflate_a_real_external_flow_role(direction, role):
    rows = [(i, 9, 100, 1) if direction == "in" else (9, i, 100, 1) for i in range(1, 7)]
    nodes, edges = _frames([(i, 1, False) for i in [*range(1, 7), 9]], rows)
    expected = analyze(nodes, edges, _transactions())["nodes_roles"].set_index("gid").loc[9]
    _, with_self = _frames([], rows + [(9, 9, 1e20, 1)])
    result = analyze(nodes, with_self, _transactions())
    _assert_contract(result, nodes, with_self)
    actual = result["nodes_roles"].set_index("gid").loc[9]
    features = compute_features(nodes, with_self).set_index("gid").loc[9]
    assert actual["role"] == expected["role"] == role
    assert actual["role_score"] == expected["role_score"]
    assert features[f"external_{direction}_sum"] == 600
    assert features[f"funded_{direction}_degree"] == 6
    assert "600 KZT" in actual["evidence"]
    assert "Самопереводы" in actual["evidence"]


@pytest.mark.parametrize("outflow,depth,seed,expected", [
    (0, 3, False, "terminal"), (10, 3, False, "terminal"),
    (11, 3, False, "peripheral"), (79, 3, False, "peripheral"),
    (80, 3, False, "transit"), (100, 3, False, "transit"),
    (120, 3, False, "transit"), (121, 3, False, "peripheral"),
    (0, 4, False, "peripheral"), (100, 4, False, "peripheral"),
    (0, 0, True, "peripheral"), (100, 0, True, "peripheral"),
])
def test_flow_rules_at_boundaries_and_with_incomplete_observation(outflow, depth, seed, expected):
    nodes, edges = _frames([(1, 0, True), (2, depth, seed), (3, 4, False)], [(1, 2, 100, 1)] + ([(2, 3, outflow, 1)] if outflow else []))
    row = analyze(nodes, edges, _transactions())["nodes_roles"].set_index("gid").loc[2]
    assert row.role == expected
    if seed:
        assert "входящие неполны" in row.evidence
    if depth == 4:
        assert "исходящие могут быть обрезаны" in row.evidence


def test_coordinator_score_uses_positive_q95_and_role_precedence():
    nodes, edges = _frames(
        [(1, 0, True), (2, 0, True), (3, 1, False)] + [(i, 2, False) for i in range(4, 10)],
        [(1, 3, 300, 1), (2, 3, 300, 1)] + [(3, i, 100, 1) for i in range(4, 10)],
    )
    # Hand-labelled communities isolate rule behaviour from community detection.
    clusters = pd.DataFrame({"gid": nodes.gid, "cluster_id": nodes.gid})
    features = add_cluster_context(compute_features(nodes, edges), edges, clusters)
    thresholds = derive_thresholds(features)
    assert 0 < thresholds["betweenness_scale"] < 1
    row = assign_roles(features).set_index("gid").loc[3]
    assert row.role == "coordinator"  # Also satisfies distributor and transit.
    assert row.role_score == pytest.approx(1.0)
    # Without two reachable seeds, the same node becomes distributor, not transit.
    features["n_seed_reachable"] = 1
    assert assign_roles(features).set_index("gid").loc[3, "role"] == "distributor"


@pytest.mark.parametrize("count", [8, 510])
def test_large_int64_ids_shuffling_and_input_immutability(count):
    base = 10**17
    nodes, edges = _frames(
        [(base + i, min(i, 4), i == 0) for i in range(count)],
        [(base + i, base + i + 1, 100, 1) for i in range(count - 1)],
    )
    transactions = edges[["src", "dst", "sum_kzt"]].assign(date=pd.Timestamp("2026-01-01"))
    originals = [frame.copy(deep=True) for frame in (nodes, edges, transactions)]
    expected = analyze(nodes, edges, transactions)
    _assert_contract(expected, nodes, edges)
    for seed in (7, 19):
        actual = analyze(nodes.sample(frac=1, random_state=seed), edges.sample(frac=1, random_state=seed), transactions)
        for key in expected:
            assert_frame_equal(actual[key], expected[key], check_exact=True)
    for frame, original in zip((nodes, edges, transactions), originals):
        assert_frame_equal(frame, original)


def test_reciprocal_projection_sums_weights_without_losing_direction(monkeypatch):
    nodes, edges = _frames([(1, 0, True), (2, 1, False), (3, 1, False)], [(1, 2, 30, 1), (2, 1, 70, 1), (2, 3, 0, 1)])
    def check_projection(graph, weight, seed):
        assert list(graph.edges(data=True)) == [(1, 2, {"weight": 1.0})]
        return [{1, 2}]
    monkeypatch.setattr("apexflow.analytics.nx.algorithms.community.louvain_communities", check_projection)
    result = analyze(nodes, edges, _transactions())
    _assert_contract(result, nodes, edges)
    assert result["clusters"].sum_kzt_internal.tolist() == [100, 0]


@pytest.mark.parametrize("scale", [1e-200, 1.0, 1e200])
def test_clustering_is_invariant_to_finite_monetary_scale(scale):
    nodes, edges = _frames(
        [(1, 0, True), (2, 1, False), (3, 1, False), (4, 2, False), (5, 0, True)],
        [(1, 2, 100, 1), (2, 3, 1, 1), (3, 4, 100, 1)],
    )
    expected = assign_clusters(nodes, edges)
    assert expected["cluster_id"].tolist() == [1, 1, 2, 2, 3]
    scaled = edges.assign(sum_kzt=edges["sum_kzt"] * scale)
    actual = assign_clusters(nodes, scaled)
    assert_frame_equal(actual, expected, check_exact=True)
    # Only the community weights are normalised; CSV turnover retains KZT.
    result = analyze(nodes, scaled, _transactions())
    _assert_contract(result, nodes, scaled)
    assert result["clusters"]["sum_kzt_internal"].tolist() == pytest.approx(
        [100 * scale, 100 * scale, 0], rel=1e-12, abs=0
    )


@pytest.mark.parametrize("transaction_count", [2**53 + 1, 2**63 - 1])
def test_transaction_counts_remain_exact_above_float_and_int64_sum_limits(transaction_count):
    nodes, edges = _frames(
        [(1, 0, True), (2, 0, True), (3, 1, False)],
        [(1, 3, 100, transaction_count), (2, 3, 100, transaction_count)],
    )
    features = compute_features(nodes, edges).set_index("gid")
    assert features["in_n_tx"].to_dict() == {1: 0, 2: 0, 3: 2 * transaction_count}
    assert features["out_n_tx"].to_dict() == {1: transaction_count, 2: transaction_count, 3: 0}


@pytest.mark.parametrize("integer_dtype,boolean_dtype,money_dtype", [
    ("Int64", "boolean", "Float64"),
    ("int64[pyarrow]", "bool[pyarrow]", "double[pyarrow]"),
])
def test_nullable_and_arrow_inputs_preserve_large_identifiers_and_results(
    integer_dtype, boolean_dtype, money_dtype
):
    if "pyarrow" in integer_dtype:
        pytest.importorskip("pyarrow")
    base = 2**63 - 5
    nodes, edges = _frames(
        [(base + i, min(i, 4), i == 0) for i in range(5)],
        [(base, base + 1, 100, 1), (base + 1, base + 2, 90, 1)],
    )
    expected = analyze(nodes, edges, _transactions())
    typed_nodes = nodes.astype({"gid": integer_dtype, "depth": integer_dtype, "is_seed": boolean_dtype})
    typed_edges = edges.astype({
        "src": integer_dtype, "dst": integer_dtype,
        "n_tx": integer_dtype, "sum_kzt": money_dtype,
    })
    actual = analyze(typed_nodes, typed_edges, _transactions())
    _assert_contract(actual, typed_nodes, typed_edges)
    for name in expected:
        assert_frame_equal(actual[name], expected[name], check_exact=True)


@pytest.mark.parametrize("table,column,values", [
    ("nodes", "gid", [1.0, 2.0]), ("nodes", "gid", [1, 1]),
    ("nodes", "is_seed", ["True", "False"]), ("nodes", "depth", [0, 5]),
    ("edges", "src", [99]), ("edges", "n_tx", [0]), ("edges", "n_tx", [1.5]),
    ("edges", "sum_kzt", [-1.0]), ("edges", "sum_kzt", [float("inf")]),
    ("edges", "sum_kzt", [float("nan")]), ("edges", "sum_kzt", ["100"]),
])
def test_invalid_inputs_fail_explicitly(table, column, values):
    nodes, edges = _frames([(1, 0, True), (2, 1, False)], [(1, 2, 100, 1)])
    {"nodes": nodes, "edges": edges}[table][column] = values
    with pytest.raises(ValueError):
        analyze(nodes, edges, _transactions())


def test_duplicate_pairs_fail_instead_of_overwriting_payments():
    nodes, edges = _frames([(1, 0, True), (2, 1, False)], [(1, 2, 100, 1), (1, 2, 200, 2)])
    with pytest.raises(ValueError, match="directed pair"):
        analyze(nodes, edges, _transactions())


def test_integer_amounts_do_not_overflow_before_aggregation():
    nodes, edges = _frames([(1, 0, True), (2, 0, True), (3, 1, False)], [(1, 3, 6e18, 1), (2, 3, 6e18, 1)])
    edges["sum_kzt"] = edges.sum_kzt.astype("int64")
    features = compute_features(nodes, edges).set_index("gid")
    assert features.loc[3, "in_sum"] == pytest.approx(1.2e19)
    _assert_contract(analyze(nodes, edges, _transactions()), nodes, edges)


def test_evidence_reserves_space_for_observation_limitations():
    row = pd.Series({"role": "consolidator", "in_degree": 1000000,
        "funded_in_degree": 1000000, "external_in_sum": 1e30,
        "in_sum": 1e30, "flow_ratio": 1e20, "is_seed": True,
        "is_boundary": True, "has_self_loop": True, "self_sum": 1e10})
    evidence = make_evidence(row)
    assert len(evidence) <= 200
    assert "Seed: входящие неполны." in evidence
    assert "Depth=4: исходящие могут быть обрезаны." in evidence
    assert "Самопереводы" in evidence


def test_priority_explanation_uses_largest_actual_contributions():
    row = pd.Series({"in_sum": 1000, "out_sum": 1000,
        "unique_neighbor_count": 4, "betweenness": 0.00001,
        "n_seed_reachable": 9, "priority_volume": .01,
        "priority_degree": .02, "priority_centrality": .25, "priority_reach": .15})
    text = make_priority_why(row)
    assert "оборот" not in text
    assert text.index("центральность") < text.index("9 seed") < text.index("4 контрагентов")
    assert "1e-05" in text  # A positive metric must not be displayed as zero.


def test_money_summation_is_independent_of_input_order():
    nodes, edges = _frames(
        [(i, 1, False) for i in range(25)],
        [(i, 24, 1e16 if i == 0 else 1.0, 1) for i in range(24)],
    )
    first = analyze(nodes, edges, _transactions())
    second = analyze(nodes, edges.sample(frac=1, random_state=7), _transactions())
    for key in first:
        assert_frame_equal(first[key], second[key], check_exact=True)


def test_self_loop_cannot_turn_a_real_terminal_pattern_into_transit():
    nodes, edges = _frames([(1, 0, True), (2, 1, False)], [(1, 2, 100, 1), (2, 2, 10000, 1)])
    result = analyze(nodes, edges, _transactions())
    row = result["nodes_roles"].set_index("gid").loc[2]
    assert row.role == "peripheral"
    assert "Самопереводы" in row.evidence
    _assert_contract(result, nodes, edges)
