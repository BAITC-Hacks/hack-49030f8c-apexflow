"""Orchestration for a single, reproducible ApexFlow analysis."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

from .io import load_inputs, validate_outputs, write_outputs


def run_pipeline(data_dir: str | Path, output_dir: str | Path) -> dict[str, int | float | str]:
    started = perf_counter()
    nodes, edges, transactions = load_inputs(data_dir)
    try:
        from .analytics import analyze
    except ImportError as exc:
        raise RuntimeError("Модуль analytics.py с функцией analyze ещё не интегрирован") from exc
    results = analyze(nodes, edges, transactions)
    validate_outputs(results, nodes, edges)
    write_outputs(results, output_dir)
    return {
        "nodes": len(nodes), "clusters": len(results["clusters"]), "top_nodes": len(results["top_nodes"]),
        "seconds": perf_counter() - started, "output": str(Path(output_dir).resolve()),
    }
