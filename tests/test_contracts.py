from __future__ import annotations

import json

import pandas as pd
import pytest

from apexflow.io import validate_inputs, validate_outputs


def sample_inputs(n: int = 21):
    nodes = pd.DataFrame({"gid": pd.Series(range(1, n + 1), dtype="int64"), "depth": pd.Series([0] + [1] * (n - 1), dtype="int64"), "is_seed": pd.Series([True] + [False] * (n - 1), dtype="bool")})
    transactions = pd.DataFrame({"src": pd.Series(range(1, n), dtype="int64"), "dst": pd.Series(range(2, n + 1), dtype="int64"), "date": ["2026-07-01"] * (n - 1), "sum_kzt": [100.0] * (n - 1)})
    edges = pd.DataFrame({"src": pd.Series(range(1, n), dtype="int64"), "dst": pd.Series(range(2, n + 1), dtype="int64"), "sum_kzt": [100.0] * (n - 1), "n_tx": pd.Series([1] * (n - 1), dtype="int64"), "depth": pd.Series([1] * (n - 1), dtype="int64")})
    return nodes, edges, transactions


def sample_results(nodes):
    roles = pd.DataFrame({"gid": nodes.gid, "role": ["peripheral"] * len(nodes), "role_score": [0.1] * len(nodes), "cluster_id": [1] * len(nodes), "priority_score": [round(1 - i / 1000, 3) for i in range(len(nodes))], "evidence": ["Наблюдаемые связи: 0."] * len(nodes)})
    clusters = pd.DataFrame({"cluster_id": [1], "n_nodes": [len(nodes)], "n_seed": [int(nodes.is_seed.sum())], "sum_kzt_internal": [(len(nodes) - 1) * 100.0], "top_gids": [json.dumps(list(nodes.gid.head(3)))], "hypothesis": ["Единая тестовая компонента"]})
    top = roles.sort_values(["priority_score", "gid"], ascending=[False, True]).head(20).copy()
    top.insert(0, "rank", range(1, len(top) + 1))
    top["why"] = "Тестовое объяснение"
    return {"nodes_roles": roles, "clusters": clusters, "top_nodes": top[["rank", "gid", "role", "priority_score", "why"]]}


def test_valid_input_and_output_contracts():
    nodes, edges, transactions = sample_inputs()
    validate_inputs(nodes, edges, transactions)
    validate_outputs(sample_results(nodes), nodes, edges)


def test_unknown_endpoint_is_rejected():
    nodes, edges, transactions = sample_inputs()
    edges.loc[0, "src"] = 999
    with pytest.raises(ValueError, match="отсутствующие"):
        validate_inputs(nodes, edges, transactions)


def test_output_cannot_lose_an_isolate():
    nodes, edges, _ = sample_inputs()
    results = sample_results(nodes)
    results["nodes_roles"] = results["nodes_roles"].iloc[:-1]
    with pytest.raises(ValueError, match="каждый gid"):
        validate_outputs(results, nodes, edges)


@pytest.mark.parametrize("n", [1, 7, 21])
def test_small_samples_use_minimum_of_twenty_and_node_count(n):
    nodes, edges, transactions = sample_inputs(n)
    validate_inputs(nodes, edges, transactions)
    validate_outputs(sample_results(nodes), nodes, edges)


@pytest.mark.parametrize("table,column,value", [
    ("nodes_roles", "gid", 1.5),
    ("nodes_roles", "priority_score", True),
    ("nodes_roles", "evidence", None),
    ("clusters", "sum_kzt_internal", float("nan")),
    ("clusters", "sum_kzt_internal", 999),
    ("clusters", "n_seed", 0),
    ("clusters", "top_gids", "[true]"),
    ("clusters", "top_gids", "[[1]]"),
    ("clusters", "top_gids", "[2,1]"),
    ("clusters", "top_gids", "[]"),
    ("clusters", "top_gids", "[1,2,3,4,5,6]"),
    ("top_nodes", "why", None),
])
def test_invalid_outputs_fail_closed(table, column, value):
    nodes, edges, _ = sample_inputs()
    results = sample_results(nodes)
    results[table][column] = results[table][column].astype(object)
    results[table].loc[0, column] = value
    with pytest.raises(ValueError):
        validate_outputs(results, nodes, edges)


def test_numeric_strings_and_unsigned_overflow_are_rejected():
    nodes, edges, transactions = sample_inputs()
    edges["sum_kzt"] = edges["sum_kzt"].astype(str)
    with pytest.raises(ValueError, match="числовые"):
        validate_inputs(nodes, edges, transactions)
    nodes["gid"] = pd.Series([2**63 + i for i in range(len(nodes))], dtype="uint64")
    with pytest.raises(ValueError, match="int64"):
        validate_inputs(nodes, edges, transactions)


def test_unsorted_roles_are_rejected_before_publication():
    nodes, edges, _ = sample_inputs()
    results = sample_results(nodes)
    results["nodes_roles"] = results["nodes_roles"].iloc[::-1]
    with pytest.raises(ValueError, match="отсортированы"):
        validate_outputs(results, nodes, edges)


@pytest.mark.parametrize("value", [-1.0, complex(1, 1), float("inf"), float("nan")])
@pytest.mark.parametrize("table", ["edges", "transactions"])
def test_invalid_money_is_rejected(table, value):
    nodes, edges, transactions = sample_inputs()
    frame = edges if table == "edges" else transactions
    frame["sum_kzt"] = value
    with pytest.raises(ValueError):
        validate_inputs(nodes, edges, transactions)


def test_integer_turnover_cannot_overflow_during_reconciliation():
    nodes, edges, transactions = sample_inputs(2)
    transactions = pd.concat([transactions, transactions], ignore_index=True)
    transactions["sum_kzt"] = pd.Series([6 * 10**18, 6 * 10**18], dtype="int64")
    edges["sum_kzt"] = 1.2e19
    edges["n_tx"] = 2
    validate_inputs(nodes, edges, transactions)
    results = sample_results(nodes)
    results["clusters"]["sum_kzt_internal"] = 1.2e19
    validate_outputs(results, nodes, edges)


@pytest.mark.parametrize("offset", [1, -1])
def test_adjacent_large_integer_amounts_cannot_reconcile_as_equal(offset):
    nodes, edges, transactions = sample_inputs(2)
    edges["sum_kzt"] = pd.Series([2**53], dtype="int64")
    transactions["sum_kzt"] = pd.Series([2**53 + offset], dtype="int64")
    with pytest.raises(ValueError, match="агрегированными"):
        validate_inputs(nodes, edges, transactions)


def test_large_integer_reconciliation_keeps_small_transactions():
    nodes, edges, transactions = sample_inputs(2)
    transactions = pd.concat([transactions, transactions], ignore_index=True)
    transactions["sum_kzt"] = pd.Series([2**53, 1], dtype="int64")
    edges["sum_kzt"] = pd.Series([2**53 + 1], dtype="int64")
    edges["n_tx"] = 2
    validate_inputs(nodes, edges, transactions)
    edges["sum_kzt"] = 2**53
    with pytest.raises(ValueError, match="агрегированными"):
        validate_inputs(nodes, edges, transactions)


def test_large_turnover_tolerance_is_roundoff_not_percentage():
    nodes, edges, transactions = sample_inputs(2)
    transactions["sum_kzt"] = 1e15
    edges["sum_kzt"] = 1e15 + 1000
    with pytest.raises(ValueError, match="агрегированными"):
        validate_inputs(nodes, edges, transactions)
    edges["sum_kzt"] = 1e15
    results = sample_results(nodes)
    results["clusters"]["sum_kzt_internal"] = 1e15 + 1000
    with pytest.raises(ValueError, match="внутренними"):
        validate_outputs(results, nodes, edges)


def test_cluster_reconciliation_sums_int64_edges_without_overflow():
    nodes, edges, transactions = sample_inputs(3)
    edges["sum_kzt"] = pd.Series([6 * 10**18, 6 * 10**18], dtype="int64")
    transactions["sum_kzt"] = edges["sum_kzt"]
    validate_inputs(nodes, edges, transactions)
    results = sample_results(nodes)
    results["clusters"]["sum_kzt_internal"] = 1.2e19
    validate_outputs(results, nodes, edges)


def test_total_money_overflow_is_rejected():
    nodes, edges, transactions = sample_inputs(3)
    edges["sum_kzt"] = 1e308
    with pytest.raises(ValueError, match="Суммарный оборот"):
        validate_inputs(nodes, edges, transactions)
