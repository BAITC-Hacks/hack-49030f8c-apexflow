"""Public, deterministic ApexFlow analytics API."""

from __future__ import annotations

import json
from typing import Any

import networkx as nx
import pandas as pd

from .features import _gid_key, add_cluster_context, build_graph, compute_features
from .rules import assign_roles, make_evidence, make_priority_why


NODES_ROLES_COLUMNS = [
    "gid",
    "role",
    "role_score",
    "cluster_id",
    "priority_score",
    "evidence",
]
CLUSTERS_COLUMNS = [
    "cluster_id",
    "n_nodes",
    "n_seed",
    "sum_kzt_internal",
    "top_gids",
    "hypothesis",
]
TOP_NODES_COLUMNS = ["rank", "gid", "role", "priority_score", "why"]
LOUVAIN_SEED = 42


def _scalar(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def assign_clusters(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Find Louvain communities and give them deterministic, presentation-safe IDs."""
    build_graph(nodes, edges)  # Reuse the input-contract checks used by features.
    graph = nx.Graph()
    gids = sorted(nodes["gid"].tolist(), key=_gid_key)
    graph.add_nodes_from(gids)

    edge_rows = edges[["src", "dst", "sum_kzt"]].to_dict("records")
    for edge in sorted(
        edge_rows, key=lambda row: (_gid_key(row["src"]), _gid_key(row["dst"]))
    ):
        if edge["src"] == edge["dst"]:
            continue
        weight = float(edge["sum_kzt"])
        if graph.has_edge(edge["src"], edge["dst"]):
            graph[edge["src"]][edge["dst"]]["weight"] += weight
        else:
            graph.add_edge(edge["src"], edge["dst"], weight=weight)

    isolates = sorted(nx.isolates(graph), key=_gid_key)
    isolate_set = set(isolates)
    connected = graph.subgraph([gid for gid in graph if gid not in isolate_set]).copy()
    communities: list[set[Any]] = []
    if connected.number_of_nodes():
        communities.extend(
            set(community)
            for community in nx.algorithms.community.louvain_communities(
                connected, weight="weight", seed=LOUVAIN_SEED
            )
        )
    communities.extend({gid} for gid in isolates)
    communities.sort(key=lambda community: _gid_key(min(community, key=_gid_key)))

    records = []
    for index, community in enumerate(communities, start=1):
        cluster_id = f"cluster-{index:03d}"
        for gid in sorted(community, key=_gid_key):
            records.append({"gid": gid, "cluster_id": cluster_id})
    return pd.DataFrame(records, columns=["gid", "cluster_id"])


def _hypothesis(group: pd.DataFrame) -> str:
    seed_count = int(group["is_seed"].sum())
    roles = group["role"].value_counts()
    if len(group) == 1 and bool(group.iloc[0]["is_isolated"]):
        return "Изолированный узел: наблюдаемых связей в графе нет."
    if seed_count >= 2:
        return f"Связанная группа охватывает {seed_count} seed-направления; проверьте общий контур."
    if roles.get("coordinator", 0):
        return "В группе есть структурный координатор; проверьте связи между сообществами."
    if roles.get("consolidator", 0):
        return "В группе есть признаки консолидации поступлений у приоритетных узлов."
    if roles.get("distributor", 0):
        return "В группе есть признаки распределения средств нескольким получателям."
    if roles.get("transit", 0):
        return "В группе есть признаки транзитного пропуска средств."
    return "Связанная группа без достаточных признаков отдельной роли; проверьте лидирующие связи."


def summarize_clusters(
    features: pd.DataFrame, edges: pd.DataFrame
) -> pd.DataFrame:
    """Create the prescribed cluster-level review table."""
    if features.empty:
        return pd.DataFrame(columns=CLUSTERS_COLUMNS)

    cluster_by_gid = features.set_index("gid")["cluster_id"].to_dict()
    edge_clusters = edges[["src", "dst", "sum_kzt"]].copy()
    edge_clusters["src_cluster"] = edge_clusters["src"].map(cluster_by_gid)
    edge_clusters["dst_cluster"] = edge_clusters["dst"].map(cluster_by_gid)
    internal_sums = (
        edge_clusters.loc[edge_clusters["src_cluster"] == edge_clusters["dst_cluster"]]
        .groupby("src_cluster", sort=False)["sum_kzt"]
        .sum()
        .to_dict()
    )

    records = []
    for cluster_id, group in features.groupby("cluster_id", sort=True):
        ranked = group.sort_values(
            ["priority_score", "gid"], ascending=[False, True], kind="mergesort"
        ).head(5)
        records.append(
            {
                "cluster_id": cluster_id,
                "n_nodes": int(len(group)),
                "n_seed": int(group["is_seed"].sum()),
                "sum_kzt_internal": float(internal_sums.get(cluster_id, 0.0)),
                "top_gids": json.dumps(
                    [_scalar(gid) for gid in ranked["gid"].tolist()], ensure_ascii=False
                ),
                "hypothesis": _hypothesis(group),
            }
        )
    return pd.DataFrame(records, columns=CLUSTERS_COLUMNS).sort_values(
        "cluster_id", kind="mergesort"
    ).reset_index(drop=True)


def _top_nodes(features: pd.DataFrame) -> pd.DataFrame:
    ranked = features.sort_values(
        ["priority_score", "gid"], ascending=[False, True], kind="mergesort"
    ).head(min(20, len(features)))
    result = ranked[["gid", "role", "priority_score"]].copy()
    result.insert(0, "rank", range(1, len(result) + 1))
    result["why"] = ranked.apply(make_priority_why, axis=1).to_list()
    return result[TOP_NODES_COLUMNS].reset_index(drop=True)


def analyze(
    nodes: pd.DataFrame, edges: pd.DataFrame, transactions: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Run the complete structural baseline without files, network calls or mutation.

    ``transactions`` is accepted for the stable application interface.  This
    baseline intentionally uses the complete aggregated graph first; temporal
    indicators are a later, additive iteration and do not change this schema.
    """
    _ = transactions
    clusters = assign_clusters(nodes, edges)
    features = add_cluster_context(compute_features(nodes, edges), edges, clusters)
    scored = assign_roles(features)
    scored["evidence"] = scored.apply(make_evidence, axis=1)

    nodes_roles = scored[NODES_ROLES_COLUMNS].sort_values(
        "gid", kind="mergesort"
    ).reset_index(drop=True)
    return {
        "nodes_roles": nodes_roles,
        "clusters": summarize_clusters(scored, edges),
        "top_nodes": _top_nodes(scored),
    }
