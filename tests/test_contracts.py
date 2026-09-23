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
    clusters = pd.DataFrame({"cluster_id": [1], "n_nodes": [len(nodes)], "n_seed": [1], "sum_kzt_internal": [2000.0], "top_gids": [json.dumps(list(nodes.gid.head(3)))], "hypothesis": ["Единая тестовая компонента"]})
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
