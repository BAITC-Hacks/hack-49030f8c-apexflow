"""Deterministic graph features used by the ApexFlow analytics baseline."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any

import networkx as nx
import pandas as pd


BETWEENNESS_EXACT_MAX_NODES = 500
BETWEENNESS_SAMPLES = 128
BETWEENNESS_SEED = 42


def _require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {', '.join(sorted(missing))}")


def _validate_inputs(nodes: pd.DataFrame, edges: pd.DataFrame) -> None:
    _require_columns(nodes, {"gid", "depth", "is_seed"}, "nodes")
    _require_columns(edges, {"src", "dst", "sum_kzt", "n_tx"}, "edges")

    # The pipeline owns full validation; these guards protect graph calculations
    # from lossy IDs, truthy strings and invalid monetary weights at direct calls.
    for frame, name, columns in (
        (nodes, "nodes", ("gid", "depth")),
        (edges, "edges", ("src", "dst", "n_tx")),
    ):
        for column in columns:
            values = frame[column]
            if len(values) and (
                not pd.api.types.is_integer_dtype(values.dtype)
                or values.isna().any()
                or not values.between(-(2**63), 2**63 - 1).all()
            ):
                raise ValueError(f"{name}.{column} must contain int64-compatible integers")
    if len(nodes) and (
        not pd.api.types.is_bool_dtype(nodes["is_seed"].dtype)
        or nodes["is_seed"].isna().any()
    ):
        raise ValueError("nodes.is_seed must contain booleans without missing values")
    if not nodes["depth"].between(0, 4).all():
        raise ValueError("nodes.depth must be between 0 and 4")
    if not edges["n_tx"].gt(0).all():
        raise ValueError("edges.n_tx must be positive")
    amounts = edges["sum_kzt"]
    if len(amounts) and (
        not pd.api.types.is_numeric_dtype(amounts.dtype)
        or pd.api.types.is_bool_dtype(amounts.dtype)
        or pd.api.types.is_complex_dtype(amounts.dtype)
        or amounts.isna().any()
        or not amounts.map(math.isfinite).all()
        or not amounts.ge(0).all()
    ):
        raise ValueError("edges.sum_kzt must contain finite nonnegative numbers")
    try:
        math.fsum(amounts)
    except OverflowError as exc:
        raise ValueError("Total observed amount exceeds the supported numeric range") from exc

    if nodes["gid"].isna().any() or nodes["gid"].duplicated().any():
        raise ValueError("nodes.gid must be present and unique")

    known_gids = set(nodes["gid"])
    unknown = set(edges["src"]).union(edges["dst"]).difference(known_gids)
    if unknown:
        raise ValueError("edges contain identifiers absent from nodes")

    if edges.duplicated(["src", "dst"]).any():
        raise ValueError("edges must have at most one row per directed pair")


def build_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.DiGraph:
    """Build a directed graph while keeping all nodes and a stable insertion order."""
    _validate_inputs(nodes, edges)
    graph = nx.DiGraph()

    for row in nodes.sort_values("gid").itertuples(index=False):
        graph.add_node(row.gid, depth=int(row.depth), is_seed=bool(row.is_seed))

    rows = edges[["src", "dst", "sum_kzt", "n_tx"]].to_dict("records")
    for edge in sorted(rows, key=lambda row: (row["src"], row["dst"])):
        graph.add_edge(
            edge["src"],
            edge["dst"],
            sum_kzt=float(edge["sum_kzt"]),
            n_tx=int(edge["n_tx"]),
        )
    return graph


def _series_from_group(
    frame: pd.DataFrame, group: str, value: str, gids: pd.Series
) -> pd.Series:
    # Sort before accumulation: floating-point summation must not depend on input
    # row order. Cast amounts before summing to avoid signed integer overflow.
    ordered = frame.sort_values(["src", "dst"]).copy()
    ordered[value] = ordered[value].astype(float)
    grouped = ordered.groupby(group, sort=False)[value].sum()
    return gids.map(grouped).fillna(0.0).astype(float)


def _reachable_seed_counts(graph: nx.DiGraph, seed_gids: list[Any]) -> dict[Any, int]:
    counts = dict.fromkeys(graph.nodes, 0)
    for seed_gid in sorted(seed_gids):
        for gid in nx.descendants(graph, seed_gid):
            counts[gid] += 1
    return counts


def _betweenness(graph: nx.DiGraph) -> tuple[dict[Any, float], str]:
    node_count = graph.number_of_nodes()
    if node_count <= BETWEENNESS_EXACT_MAX_NODES:
        return nx.betweenness_centrality(graph, normalized=True, weight=None), "exact"
    return (
        nx.betweenness_centrality(
            graph,
            k=min(BETWEENNESS_SAMPLES, node_count),
            normalized=True,
            weight=None,
            seed=BETWEENNESS_SEED,
        ),
        f"sampled:{min(BETWEENNESS_SAMPLES, node_count)}",
    )


def compute_features(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Calculate structural features from the directed, aggregated payments graph.

    The function does not mutate the input frames.  Cluster-dependent features are
    added later by :func:`add_cluster_context` after community detection.
    """
    graph = build_graph(nodes, edges)
    result = nodes[["gid", "depth", "is_seed"]].copy()
    result = result.sort_values("gid", kind="mergesort").reset_index(drop=True)
    result["is_seed"] = result["is_seed"].astype(bool)

    external_edges = edges.loc[edges["src"] != edges["dst"]].copy()
    result["in_degree"] = result["gid"].map(
        external_edges.groupby("dst", sort=False).size()
    ).fillna(0).astype(int)
    result["out_degree"] = result["gid"].map(
        external_edges.groupby("src", sort=False).size()
    ).fillna(0).astype(int)
    result["in_sum"] = _series_from_group(edges, "dst", "sum_kzt", result["gid"])
    result["out_sum"] = _series_from_group(edges, "src", "sum_kzt", result["gid"])
    result["in_n_tx"] = _series_from_group(edges, "dst", "n_tx", result["gid"])
    result["out_n_tx"] = _series_from_group(edges, "src", "n_tx", result["gid"])

    neighbors: dict[Any, set[Any]] = defaultdict(set)
    for edge in external_edges[["src", "dst"]].itertuples(index=False):
        neighbors[edge.src].add(edge.dst)
        neighbors[edge.dst].add(edge.src)
    result["unique_neighbor_count"] = result["gid"].map(
        lambda gid: len(neighbors[gid])
    ).astype(int)

    self_edges = edges.loc[edges["src"] == edges["dst"]]
    result["self_sum"] = _series_from_group(self_edges, "src", "sum_kzt", result["gid"])
    result["has_self_loop"] = result["gid"].isin(self_edges["src"])
    result["flow_ratio"] = float("nan")
    positive_inflow = result["in_sum"] > 0
    result.loc[positive_inflow, "flow_ratio"] = (
        result.loc[positive_inflow, "out_sum"] / result.loc[positive_inflow, "in_sum"]
    )
    result["observed_net"] = result["in_sum"] - result["out_sum"]
    result["is_boundary"] = result["depth"].eq(4)
    result["is_isolated"] = (
        result["in_degree"].eq(0) & result["out_degree"].eq(0) & ~result["has_self_loop"]
    )

    reachable = _reachable_seed_counts(
        graph, result.loc[result["is_seed"], "gid"].tolist()
    )
    result["n_seed_reachable"] = result["gid"].map(reachable).fillna(0).astype(int)
    centrality, centrality_mode = _betweenness(graph)
    result["betweenness"] = result["gid"].map(centrality).fillna(0.0).astype(float)
    result["betweenness_mode"] = centrality_mode
    return result


def add_cluster_context(
    features: pd.DataFrame, edges: pd.DataFrame, clusters: pd.DataFrame
) -> pd.DataFrame:
    """Attach stable community IDs and the number of adjacent communities."""
    _require_columns(clusters, {"gid", "cluster_id"}, "clusters")
    result = features.copy()
    cluster_by_gid = clusters.set_index("gid")["cluster_id"].to_dict()
    result["cluster_id"] = result["gid"].map(cluster_by_gid)
    if result["cluster_id"].isna().any():
        raise ValueError("clusters must assign every node")
    result["cluster_id"] = result["cluster_id"].astype("int64")

    adjacent_clusters: dict[Any, set[int]] = defaultdict(set)
    for edge in edges.loc[edges["src"] != edges["dst"], ["src", "dst"]].itertuples(index=False):
        source_cluster = cluster_by_gid[edge.src]
        destination_cluster = cluster_by_gid[edge.dst]
        if source_cluster != destination_cluster:
            adjacent_clusters[edge.src].add(destination_cluster)
            adjacent_clusters[edge.dst].add(source_cluster)
    result["neighbor_cluster_count"] = result["gid"].map(
        lambda gid: len(adjacent_clusters[gid])
    ).astype(int)
    return result
