"""Verify the supplied dataset end to end inside the runtime image.

Uses real analytics and the UI loader. Only aggregate evidence is printed;
the three required CSV files and provenance stay in the chosen output directory.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter

from pandas.testing import assert_frame_equal

from apexflow.analytics import analyze
from apexflow.io import INPUT_FILES, OUTPUT_COLUMNS, file_hashes, load_inputs, validate_outputs, validate_run_manifest
from apexflow.pipeline import run_pipeline
from ui.data import find_node, load_view_data
from ui.graph import build_subgraph, render_svg


def verify(data_dir: Path, output_dir: Path) -> dict:
    original_hashes = file_hashes(data_dir, INPUT_FILES.values())
    nodes, edges, transactions = load_inputs(data_dir)
    snapshots = [frame.copy(deep=True) for frame in (nodes, edges, transactions)]
    started = perf_counter()
    run_pipeline(data_dir, output_dir)
    first_seconds = perf_counter() - started
    output_names = [f"{name}.csv" for name in OUTPUT_COLUMNS]
    first_outputs = file_hashes(output_dir, output_names)
    started = perf_counter()
    run_pipeline(data_dir, output_dir)
    second_seconds = perf_counter() - started
    assert first_outputs == file_hashes(output_dir, output_names), "CSV differ on repeated calculation"
    assert max(first_seconds, second_seconds) < 300, "Pipeline exceeds 5-minute limit"

    expected = analyze(nodes, edges, transactions)
    permuted = analyze(nodes.sample(frac=1, random_state=31), edges.sample(frac=1, random_state=31), transactions)
    validate_outputs(expected, nodes, edges)
    for name, frame in expected.items():
        assert_frame_equal(frame, permuted[name], check_exact=True)
        assert (output_dir / f"{name}.csv").read_bytes() == frame.to_csv(index=False).encode("utf-8")
    for original, snapshot in zip((nodes, edges, transactions), snapshots):
        assert_frame_equal(original, snapshot, check_exact=True)
    roles = expected["nodes_roles"].set_index("gid")
    incomplete = nodes.loc[nodes.is_seed | nodes.depth.eq(4), "gid"]
    assert not roles.loc[incomplete, "role"].isin(["terminal", "transit"]).any()

    view = load_view_data(data_dir, output_dir)
    for gid in nodes.gid:
        assert find_node(view, str(gid)) is not None
    linked = set(edges.src) | set(edges.dst)
    isolates = nodes.loc[~nodes.gid.isin(linked), "gid"]
    for gid in isolates:
        graph = build_subgraph(view.nodes_roles, view.edges, str(gid))
        assert len(graph.nodes) == 1 and graph.edges.empty
        assert str(gid) in render_svg(graph, str(gid))
    assert set(view.raw_csv) == set(output_names)
    for name, content in view.raw_csv.items():
        assert content == (output_dir / name).read_bytes()
    assert original_hashes == file_hashes(data_dir, INPUT_FILES.values()), "Source data changed"
    manifest = validate_run_manifest(data_dir, output_dir)
    return {
        "status": "passed", "revision": manifest["revision"], "python": platform.python_version(),
        "dependencies": manifest["dependencies"], "input_sha256": original_hashes,
        "output_sha256": first_outputs, "nodes": len(nodes), "edges": len(edges),
        "transactions": len(transactions), "isolates": len(isolates),
        "clusters": len(expected["clusters"]), "top_nodes": len(expected["top_nodes"]),
        "pipeline_seconds": [first_seconds, second_seconds],
        "checks": ["all gids searchable", "CSV downloads identical", "isolates visible",
                   "seed/boundary constraints", "deterministic CSV and shuffled inputs",
                   "source immutability", "manifest hashes", "pipeline under 300 seconds"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("/app/data"))
    parser.add_argument("--output", type=Path, default=Path("/app/output"))
    args = parser.parse_args()
    print(json.dumps(verify(args.data, args.output), ensure_ascii=False, indent=2))
