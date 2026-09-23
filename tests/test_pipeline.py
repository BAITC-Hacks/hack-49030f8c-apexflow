from __future__ import annotations

import sys
import types

from apexflow.pipeline import run_pipeline
from test_contracts import sample_inputs, sample_results


def test_pipeline_writes_three_csvs(tmp_path, monkeypatch):
    nodes, edges, transactions = sample_inputs()
    data = tmp_path / "data"
    data.mkdir()
    nodes.to_parquet(data / "nodes.parquet", index=False)
    edges.to_parquet(data / "edges.parquet", index=False)
    transactions.to_parquet(data / "transactions.parquet", index=False)
    monkeypatch.setitem(sys.modules, "apexflow.analytics", types.SimpleNamespace(analyze=lambda *_: sample_results(nodes)))
    output = tmp_path / "output"
    report = run_pipeline(data, output)
    assert report["nodes"] == len(nodes)
    assert {path.name for path in output.glob("*.csv")} == {"nodes_roles.csv", "clusters.csv", "top_nodes.csv"}
