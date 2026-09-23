from __future__ import annotations

import sys
import types
import json
from pathlib import Path
import subprocess

from apexflow.pipeline import run_pipeline
from apexflow.io import write_outputs
import pandas as pd
import pytest
from test_contracts import sample_inputs, sample_results
from make_fixture import make_fixture
from check_smoke import check
from apexflow.provenance import verify_run, input_fingerprints


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


@pytest.fixture
def real_inputs(tmp_path):
    data = tmp_path / "data"
    make_fixture(data)
    return data, tmp_path / "output"


def test_real_pipeline_roundtrip_ui_and_repeatability(real_inputs):
    data, output = real_inputs
    before = input_fingerprints(data)
    first = run_pipeline(data, output)
    check(data, output)
    first_manifest = verify_run(data, output)
    second = run_pipeline(data, output)
    second_manifest = verify_run(data, output)
    assert first["nodes"] == 24
    assert first["top_nodes"] == 20
    assert first["run_id"] != second["run_id"]
    assert first_manifest["outputs"] == second_manifest["outputs"]
    assert before == input_fingerprints(data)
    assert first_manifest["profile"]["isolates"] == 2
    assert len(first_manifest["runtime"]["source_sha256"]) == 64
    assert first_manifest["profile"]["tables"]["transactions"]["rows"] == 21
    assert not (output / ".pipeline.lock").exists()


@pytest.mark.parametrize("n", [1, 7, 21])
def test_real_analysis_small_and_edgeless_datasets(tmp_path, n):
    data = tmp_path / "data"
    data.mkdir()
    for name, frame in zip(("nodes", "edges", "transactions"), sample_inputs(n)):
        frame.to_parquet(data / f"{name}.parquet", index=False)
    report = run_pipeline(data, tmp_path / "output")
    assert report["top_nodes"] == min(20, n)
    check(data, tmp_path / "output")


@pytest.mark.parametrize("corruption", ["missing", "column", "endpoint", "gid", "date", "aggregate"])
def test_invalid_inputs_fail_cli_and_invalidate_old_success(real_inputs, corruption):
    data, output = real_inputs
    run_pipeline(data, output)
    original_csvs = {p.name: p.read_bytes() for p in output.glob("*.csv")}
    if corruption == "missing":
        (data / "nodes.parquet").unlink()
    else:
        name = "nodes" if corruption in ("column", "gid") else "transactions"
        frame = pd.read_parquet(data / f"{name}.parquet")
        if corruption == "column":
            frame = frame.drop(columns="depth")
        elif corruption == "gid":
            frame.loc[1, "gid"] = frame.loc[0, "gid"]
        elif corruption == "endpoint":
            frame.loc[0, "src"] = -1
        elif corruption == "date":
            frame.loc[0, "date"] = "not-a-date"
        else:
            frame.loc[0, "sum_kzt"] += 1
        frame.to_parquet(data / f"{name}.parquet", index=False)
    result = subprocess.run([sys.executable, "-m", "apexflow", "--data", str(data), "--output", str(output)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 1
    assert "load_validate" in result.stderr
    assert {p.name: p.read_bytes() for p in output.glob("*.csv")} == original_csvs
    assert json.loads((output / "run_status.json").read_text())["status"] == "failed"
    with pytest.raises(ValueError, match="Последний запуск"):
        verify_run(data, output)


@pytest.mark.parametrize("target", ["data", "output"])
def test_verify_detects_tampering(real_inputs, target):
    data, output = real_inputs
    run_pipeline(data, output)
    path = data / "edges.parquet" if target == "data" else output / "nodes_roles.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        verify_run(data, output)


def test_lock_does_not_overwrite_another_runs_status(real_inputs):
    data, output = real_inputs
    run_pipeline(data, output)
    before = (output / "run_status.json").read_bytes()
    (output / ".pipeline.lock").write_text("another-run")
    with pytest.raises(RuntimeError, match="Другой pipeline"):
        run_pipeline(data, output)
    with pytest.raises(ValueError, match="lock"):
        verify_run(data, output)
    assert (output / "run_status.json").read_bytes() == before


def test_changed_input_during_analysis_is_not_published(real_inputs, monkeypatch):
    import apexflow.analytics as analytics
    original = analytics.analyze
    data, output = real_inputs
    def changing(*args):
        result = original(*args)
        path = data / "nodes.parquet"
        path.write_bytes(path.read_bytes() + b"\n")
        return result
    monkeypatch.setattr(analytics, "analyze", changing)
    with pytest.raises(RuntimeError, match="изменились"):
        run_pipeline(data, output)
    assert not list(output.glob("*.csv"))


@pytest.mark.parametrize("bad_result", [None, {}, {"nodes_roles": None}])
def test_invalid_analysis_result_is_not_published(real_inputs, monkeypatch, bad_result):
    import apexflow.analytics as analytics
    data, output = real_inputs
    monkeypatch.setattr(analytics, "analyze", lambda *args: bad_result)
    with pytest.raises(RuntimeError, match="validate_outputs"):
        run_pipeline(data, output)
    assert not list(output.glob("*.csv"))


@pytest.mark.parametrize("existing", [True, False])
def test_partial_publication_rolls_back(tmp_path, monkeypatch, existing):
    nodes, _, _ = sample_inputs()
    results = sample_results(nodes)
    if existing:
        write_outputs(results, tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.glob("*.csv")}
    original = Path.replace
    def fail_second(self, target):
        if self.suffix == ".tmp" and Path(target).name == "clusters.csv":
            raise OSError("simulated disk failure")
        return original(self, target)
    monkeypatch.setattr(Path, "replace", fail_second)
    results["nodes_roles"]["evidence"] = "Новый расчёт"
    with pytest.raises(OSError, match="simulated"):
        write_outputs(results, tmp_path)
    assert {p.name: p.read_bytes() for p in tmp_path.glob("*.csv")} == before
    assert not list(tmp_path.glob(".*"))


def test_acceptance_rejects_synthetic_dataset(real_inputs):
    with pytest.raises(RuntimeError, match="Приёмка полного набора"):
        run_pipeline(*real_inputs, acceptance=True)


def test_cli_json_and_verification(real_inputs):
    data, output = real_inputs
    command = [sys.executable, "-m", "apexflow", "--data", str(data), "--output", str(output), "--json"]
    run = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stderr
    report = json.loads(run.stdout)
    verify = subprocess.run(command + ["--verify-output"], capture_output=True, text=True, timeout=30)
    assert verify.returncode == 0, verify.stderr
    assert json.loads(verify.stdout)["run_id"] == report["run_id"]


def test_output_cannot_overlap_input(real_inputs):
    data, _ = real_inputs
    for output in (data, data / "output", data.parent):
        with pytest.raises(ValueError, match="раздельными"):
            run_pipeline(data, output)


def test_verification_rejects_different_source_code(real_inputs, monkeypatch):
    import apexflow.provenance as provenance
    run_pipeline(*real_inputs)
    monkeypatch.setattr(provenance, "runtime_identity", lambda: {"source_sha256": "changed"})
    with pytest.raises(ValueError, match="Код изменился"):
        verify_run(*real_inputs)
