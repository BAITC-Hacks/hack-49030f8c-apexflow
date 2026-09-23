from __future__ import annotations

import sys
import types

from apexflow.pipeline import run_pipeline
from apexflow.io import write_outputs
import pandas as pd
import pytest
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


def test_failed_csv_roundtrip_removes_temporary_files(tmp_path, monkeypatch):
    nodes, _, _ = sample_inputs()
    def fail(*args, **kwargs):
        raise ValueError("read failed")
    monkeypatch.setattr(pd, "read_csv", fail)
    with pytest.raises(ValueError, match="read failed"):
        write_outputs(sample_results(nodes), tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_csv_roundtrip_keeps_quotes_newlines_and_large_gid(tmp_path):
    nodes, _, _ = sample_inputs()
    nodes["gid"] += 9007199254740992
    results = sample_results(nodes)
    results["nodes_roles"].loc[0, "evidence"] = '1 перевод, "цитата"\nновая строка'
    write_outputs(results, tmp_path)
    reread = pd.read_csv(tmp_path / "nodes_roles.csv", dtype={"gid":str})
    assert reread.loc[0, "gid"] == "9007199254740993"
    assert reread.loc[0, "evidence"] == results["nodes_roles"].loc[0, "evidence"]
