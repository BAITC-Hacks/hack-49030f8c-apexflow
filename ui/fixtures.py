"""One shared, explicitly synthetic fixture for Streamlit and GitHub Pages."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ui.data import ViewData, validate_view_data


def build_synthetic_view_data() -> ViewData:
    sample = json.loads((Path(__file__).resolve().parents[1] / "site" / "demo-data.json").read_text(encoding="utf-8"))
    if sample.get("synthetic") is not True:
        raise ValueError("The bundled UI fixture must be explicitly synthetic.")
    node_rows = sample["nodes"]
    nodes = pd.DataFrame([
        {"gid": int(row["gid"]), "depth": row["depth"], "is_seed": row["isSeed"]}
        for row in node_rows
    ])
    edges = pd.DataFrame([
        {"src": int(row["src"]), "dst": int(row["dst"]), "sum_kzt": row["sum"], "n_tx": row["nTx"], "depth": row["depth"]}
        for row in sample["edges"]
    ])
    roles = pd.DataFrame([
        {"gid": int(row["gid"]), "role": row["role"], "role_score": row["roleScore"],
         "cluster_id": row["cluster"], "priority_score": row["priority"], "evidence": row["evidence"]}
        for row in node_rows
    ]).sort_values("gid").reset_index(drop=True)
    clusters = pd.DataFrame([
        {"cluster_id": row["id"], "n_nodes": row["nNodes"], "n_seed": row["nSeed"],
         "sum_kzt_internal": row["sum"], "top_gids": json.dumps([int(gid) for gid in row["topGids"]]),
         "hypothesis": row["hypothesis"]}
        for row in sample["clusters"]
    ])
    top = roles.sort_values(["priority_score", "gid"], ascending=[False, True]).copy()
    top.insert(0, "rank", range(1, len(top) + 1))
    top["why"] = top["evidence"]
    return validate_view_data(
        roles, clusters, top[["rank", "gid", "role", "priority_score", "why"]], nodes, edges,
        source="Синтетический пример: 7 узлов; роли и scores заданы для проверки UI, не рассчитаны аналитикой",
    )
