from __future__ import annotations

import sys
import types
import json

from apexflow.pipeline import run_pipeline
from apexflow.io import RUN_MANIFEST, file_hashes, validate_run_manifest, write_outputs
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
    manifest = validate_run_manifest(data, output)
    assert manifest["report"] == report
    assert manifest["status"] == "complete"
    assert manifest["inputs"] == file_hashes(data, ("nodes.parquet", "edges.parquet", "transactions.parquet"))


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


def write_sample_inputs(directory):
    directory.mkdir()
    for name, frame in zip(("nodes", "edges", "transactions"), sample_inputs()):
        frame.to_parquet(directory / f"{name}.parquet", index=False)


@pytest.mark.parametrize("filename", ["edges.parquet", "transactions.parquet", "nodes_roles.csv"])
def test_manifest_detects_input_or_output_changes(tmp_path, filename):
    data, output = tmp_path / "data", tmp_path / "output"
    write_sample_inputs(data)
    run_pipeline(data, output)
    target = (data if filename.endswith("parquet") else output) / filename
    target.write_bytes(target.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="изменились"):
        validate_run_manifest(data, output)


def test_failed_rerun_invalidates_previous_success(tmp_path, monkeypatch):
    data, output = tmp_path / "data", tmp_path / "output"
    write_sample_inputs(data)
    run_pipeline(data, output)
    old_csvs = file_hashes(output, ("nodes_roles.csv", "clusters.csv", "top_nodes.csv"))

    def fail(*args):
        raise ValueError("broken analysis")

    monkeypatch.setattr("apexflow.analytics.analyze", fail)
    with pytest.raises(ValueError, match="broken analysis"):
        run_pipeline(data, output)
    assert json.loads((output / RUN_MANIFEST).read_text())["status"] == "failed"
    assert file_hashes(output, old_csvs) == old_csvs
    with pytest.raises(ValueError, match="не завершён"):
        validate_run_manifest(data, output)


def test_pipeline_rejects_inputs_changed_during_calculation(tmp_path, monkeypatch):
    data, output = tmp_path / "data", tmp_path / "output"
    write_sample_inputs(data)

    def change_inputs(nodes, *args):
        path = data / "nodes.parquet"
        path.write_bytes(path.read_bytes() + b"changed")
        return sample_results(nodes)

    monkeypatch.setattr("apexflow.analytics.analyze", change_inputs)
    with pytest.raises(ValueError, match="во время расчёта"):
        run_pipeline(data, output)
    assert not list(output.glob("*.csv"))
    assert json.loads((output / RUN_MANIFEST).read_text())["status"] == "failed"


@pytest.mark.parametrize("content", [None, b"not JSON", b"\xff", b"[]", b'{"schema_version":1,"status":"running"}'])
def test_missing_or_broken_manifest_never_looks_fresh(tmp_path, content):
    if content is not None:
        (tmp_path / RUN_MANIFEST).write_bytes(content)
    with pytest.raises(ValueError):
        validate_run_manifest(tmp_path, tmp_path)


def test_cli_missing_data_returns_failure_and_invalidates_output(tmp_path, monkeypatch, capsys):
    from apexflow.__main__ import main

    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["apexflow", "--data", str(tmp_path / "missing"), "--output", str(output)])
    assert main() == 1
    assert "Ошибка pipeline" in capsys.readouterr().err
    assert json.loads((output / RUN_MANIFEST).read_text())["status"] == "failed"
