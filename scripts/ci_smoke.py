"""Create an explicitly synthetic CI dataset and verify the runtime container."""

from __future__ import annotations

import argparse
from pathlib import Path
import urllib.request

import pandas as pd

from apexflow.io import validate_run_manifest
from ui.data import find_node, load_view_data


def create_fixture(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    # A chain with two seeds, a boundary, an isolate and exact large identifiers.
    gids = list(range(2**53 + 1, 2**53 + 25))
    nodes = pd.DataFrame({
        "gid": pd.Series(gids, dtype="int64"),
        "depth": pd.Series([min(index // 5, 4) for index in range(24)], dtype="int64"),
        "is_seed": [index in (0, 5) for index in range(24)],
    })
    transactions = pd.DataFrame({
        "src": pd.Series(gids[:-2], dtype="int64"),
        "dst": pd.Series(gids[1:-1], dtype="int64"),
        "date": pd.to_datetime(["2026-07-01"] * 22),
        "sum_kzt": [5000.0 + index * 100 for index in range(22)],
    })
    edges = transactions.drop(columns="date").assign(n_tx=1, depth=1)
    for name, frame in (("nodes", nodes), ("edges", edges), ("transactions", transactions)):
        frame.to_parquet(directory / f"{name}.parquet", index=False)
    print("Created synthetic CI data: 24 nodes, 22 edges, 22 transactions.")


def verify(data_dir: Path, output_dir: Path) -> None:
    manifest = validate_run_manifest(data_dir, output_dir)
    data = load_view_data(data_dir, output_dir)
    for gid in data.nodes["gid"]:
        if find_node(data, str(gid)) is None:
            raise ValueError(f"UI cannot find gid {gid}")
    if manifest["report"]["seconds"] >= 300:
        raise ValueError("Pipeline exceeded the 5-minute limit")
    with urllib.request.urlopen("http://localhost:8501/_stcore/health", timeout=5) as response:
        if response.status != 200:
            raise ValueError("UI is not healthy")
    print(f"Runtime smoke passed: {len(data.nodes)} searchable nodes; complete manifest; UI HTTP 200.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("create", "verify"))
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path, nargs="?", default=Path("/app/output"))
    args = parser.parse_args()
    if args.mode == "create":
        create_fixture(args.data)
    else:
        verify(args.data, args.output)
