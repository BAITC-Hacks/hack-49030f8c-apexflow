"""Orchestration for a single, reproducible ApexFlow analysis."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from datetime import datetime, timezone
from importlib.metadata import version
import os
import platform

from .io import INPUT_FILES, OUTPUT_COLUMNS, file_hashes, load_inputs, validate_outputs, write_outputs, write_run_manifest


def run_pipeline(data_dir: str | Path, output_dir: str | Path) -> dict[str, int | float | str]:
    started = perf_counter()
    manifest = {
        "schema_version": 1, "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "revision": os.environ.get("APEXFLOW_REVISION", "unknown"),
        "python": platform.python_version(),
        "dependencies": {name: version(name) for name in ("pandas", "pyarrow", "networkx")},
    }
    # Invalidate a previous successful run before reading or replacing any data.
    write_run_manifest(output_dir, manifest)
    try:
        manifest["inputs"] = file_hashes(data_dir, INPUT_FILES.values())
        nodes, edges, transactions = load_inputs(data_dir)
        from .analytics import analyze
        results = analyze(nodes, edges, transactions)
        validate_outputs(results, nodes, edges)
        if manifest["inputs"] != file_hashes(data_dir, INPUT_FILES.values()):
            raise ValueError("Входные данные изменились во время расчёта")
        write_outputs(results, output_dir)
        report = {
            "nodes": len(nodes), "clusters": len(results["clusters"]), "top_nodes": len(results["top_nodes"]),
            "seconds": perf_counter() - started, "output": str(Path(output_dir).resolve()),
        }
        manifest.update(status="complete", completed_at=datetime.now(timezone.utc).isoformat(), report=report,
                        outputs=file_hashes(output_dir, [f"{name}.csv" for name in OUTPUT_COLUMNS]))
        write_run_manifest(output_dir, manifest)
    except Exception:
        manifest.update(status="failed", failed_at=datetime.now(timezone.utc).isoformat())
        write_run_manifest(output_dir, manifest)
        raise
    return report
