"""Select current, verifiable demonstration examples without changing analysis."""

from __future__ import annotations

import argparse
import json
import math

from apexflow.io import validate_run_manifest
from ui.data import ViewData, find_node, load_view_data


def select_examples(data: ViewData) -> list[dict]:
    ranked = data.nodes_roles.assign(_gid_order=data.nodes_roles["gid"].map(int)).sort_values(
        ["priority_score", "_gid_order"], ascending=[False, True], kind="stable"
    )
    first = ranked.loc[~ranked["is_seed"]]
    first = (first if not first.empty else ranked).iloc[0]
    chosen = [("Приоритетный non-seed" if not first["is_seed"] else "Приоритетный seed", first["gid"])]
    different = ranked.loc[(ranked["role"] != first["role"]) & (ranked["gid"] != first["gid"])]
    if not different.empty:
        chosen.append(("Другой профиль потоков", different.iloc[0]["gid"]))
    boundary = ranked.loc[(ranked["depth"] == 4) & ~ranked["gid"].isin([gid for _, gid in chosen])]
    if not boundary.empty:
        chosen.append(("Граница наблюдения depth=4", boundary.iloc[0]["gid"]))
    else:
        linked = set(data.edges["_src_key"]) | set(data.edges["_dst_key"])
        isolates = ranked.loc[~ranked["gid"].isin(linked | {gid for _, gid in chosen})]
        if not isolates.empty:
            chosen.append(("Изолированный узел", isolates.iloc[0]["gid"]))
    examples = []
    for purpose, gid in chosen:
        node = find_node(data, gid)
        incoming = data.edges.loc[data.edges["_dst_key"] == gid]
        outgoing = data.edges.loc[data.edges["_src_key"] == gid]
        incident = data.edges.loc[(data.edges["_src_key"] == gid) | (data.edges["_dst_key"] == gid)]
        strongest = incident.sort_values("sum_kzt", ascending=False, kind="stable").head(1)
        examples.append({
            "purpose": purpose, "gid": gid, "role": str(node["role"]),
            "role_score": float(node["role_score"]), "priority_score": float(node["priority_score"]),
            "cluster_id": str(node["cluster_id"]), "depth": int(node["depth"]), "is_seed": bool(node["is_seed"]),
            "evidence": str(node["evidence"]),
            "incoming_kzt": math.fsum(float(x) for x in incoming["sum_kzt"]),
            "outgoing_kzt": math.fsum(float(x) for x in outgoing["sum_kzt"]),
            "external_payers": int(incoming.loc[incoming["_src_key"] != gid, "_src_key"].nunique()),
            "external_recipients": int(outgoing.loc[outgoing["_dst_key"] != gid, "_dst_key"].nunique()),
            "edge_to_verify": None if strongest.empty else {
                "src": str(strongest.iloc[0]["src"]), "dst": str(strongest.iloc[0]["dst"]),
                "sum_kzt": float(strongest.iloc[0]["sum_kzt"]), "n_tx": int(strongest.iloc[0]["n_tx"]),
            },
        })
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="/app/data")
    parser.add_argument("--output", default="/app/output")
    args = parser.parse_args()
    data = load_view_data(args.data, args.output)
    manifest = validate_run_manifest(args.data, args.output)
    print(json.dumps({"period": data.period, "run": manifest.get("report", {}),
                      "examples": select_examples(data)}, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
