"""Offline SVG rendering for a selected directed neighbourhood."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import html
import math

import pandas as pd


ROLE_COLORS = {
    "consolidator": "#2563eb",
    "transit": "#7c3aed",
    "distributor": "#ea580c",
    "terminal": "#059669",
    "coordinator": "#be123c",
    "peripheral": "#64748b",
}
CLUSTER_COLORS = ("#2563eb", "#7c3aed", "#059669", "#ea580c", "#be123c", "#0f766e", "#a16207")


@dataclass(frozen=True)
class GraphSubset:
    nodes: pd.DataFrame
    edges: pd.DataFrame
    hidden_edges: int
    mode: str


def build_subgraph(
    nodes_roles: pd.DataFrame,
    edges: pd.DataFrame,
    selected_gid: str,
    *,
    mode: str = "neighborhood",
    max_edges: int = 24,
) -> GraphSubset:
    """Select observed edges only, preserving the original ``src → dst`` direction."""

    selected_rows = nodes_roles.loc[nodes_roles["_gid_key"] == selected_gid]
    if selected_rows.empty:
        raise ValueError("Selected gid is absent from the graph data.")
    selected = selected_rows.iloc[0]

    if mode == "cluster":
        cluster_nodes = nodes_roles.loc[nodes_roles["_cluster_key"] == selected["_cluster_key"]]
        node_keys = set(cluster_nodes["_gid_key"])
        candidates = edges.loc[
            edges["_src_key"].isin(node_keys) & edges["_dst_key"].isin(node_keys)
        ]
    else:
        candidates = edges.loc[
            (edges["_src_key"] == selected_gid) | (edges["_dst_key"] == selected_gid)
        ]

    ordered = candidates.assign(_amount=pd.to_numeric(candidates["sum_kzt"], errors="coerce").fillna(0))
    ordered = ordered.sort_values(
        ["_amount", "_src_key", "_dst_key"], ascending=[False, True, True], kind="stable"
    )
    visible_edges = ordered.head(max_edges).drop(columns="_amount")
    visible_keys = {selected_gid} | set(visible_edges["_src_key"]) | set(visible_edges["_dst_key"])
    visible_nodes = nodes_roles.loc[nodes_roles["_gid_key"].isin(visible_keys)].copy()
    visible_nodes = visible_nodes.sort_values("_gid_key", kind="stable")
    return GraphSubset(
        nodes=visible_nodes,
        edges=visible_edges,
        hidden_edges=max(len(candidates) - len(visible_edges), 0),
        mode=mode,
    )


def _cluster_color(cluster_id: object) -> str:
    digest = hashlib.sha256(str(cluster_id).encode("utf-8")).digest()[0]
    return CLUSTER_COLORS[digest % len(CLUSTER_COLORS)]


def _format_amount(value: object) -> str:
    try:
        return f"{float(value):,.2f}".replace(",", "\u202f") + " KZT"
    except (TypeError, ValueError):
        return "сумма не указана"


def _short_gid(gid: object, max_length: int = 16) -> str:
    value = str(gid)
    return value if len(value) <= max_length else f"{value[:7]}…{value[-6:]}"


def render_svg(subgraph: GraphSubset, selected_gid: str, *, color_by: str = "role") -> str:
    """Render a self-contained SVG: no CDN, scripts, or unescaped file content."""

    width, height, radius = 900, 520, 29
    nodes = subgraph.nodes.reset_index(drop=True)
    count = len(nodes)
    positions: dict[str, tuple[float, float]] = {}
    if count == 1:
        positions[str(nodes.iloc[0]["_gid_key"])] = (width / 2, height / 2)
    else:
        orbit_x, orbit_y = width * 0.39, height * 0.35
        for index, row in nodes.iterrows():
            angle = -math.pi / 2 + 2 * math.pi * index / count
            positions[str(row["_gid_key"])] = (
                width / 2 + orbit_x * math.cos(angle),
                height / 2 + orbit_y * math.sin(angle),
            )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="Направленный граф окружения выбранного узла">',
        "<defs><marker id=\"arrow\" viewBox=\"0 0 10 10\" refX=\"9\" refY=\"5\" markerWidth=\"7\" markerHeight=\"7\" orient=\"auto-start-reverse\"><path d=\"M 0 0 L 10 5 L 0 10 z\" fill=\"#475569\"/></marker></defs>",
        '<rect width="100%" height="100%" rx="12" fill="#f8fafc"/>',
    ]

    for _, edge in subgraph.edges.iterrows():
        source, target = str(edge["_src_key"]), str(edge["_dst_key"])
        if source not in positions or target not in positions:
            continue
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        distance = math.hypot(x2 - x1, y2 - y1) or 1
        start_x, start_y = x1 + radius * (x2 - x1) / distance, y1 + radius * (y2 - y1) / distance
        end_x, end_y = x2 - (radius + 6) * (x2 - x1) / distance, y2 - (radius + 6) * (y2 - y1) / distance
        title = html.escape(f"{source} → {target}; {_format_amount(edge['sum_kzt'])}; операций: {edge['n_tx']}")
        parts.append(
            f'<line x1="{start_x:.1f}" y1="{start_y:.1f}" x2="{end_x:.1f}" y2="{end_y:.1f}" '
            f'stroke="#475569" stroke-width="2" marker-end="url(#arrow)"><title>{title}</title></line>'
        )

    for _, row in nodes.iterrows():
        gid = str(row["_gid_key"])
        x, y = positions[gid]
        color = _cluster_color(row["cluster_id"]) if color_by == "cluster" else ROLE_COLORS.get(row["role"], "#64748b")
        selected = gid == selected_gid
        seed = bool(row["is_seed"])
        stroke = "#0f172a" if selected else ("#facc15" if seed else "#ffffff")
        stroke_width = 5 if selected else 3
        dash = " stroke-dasharray=\"5 3\"" if seed and not selected else ""
        title = html.escape(
            f"gid: {gid}\nРоль: {row['role']}\nКластер: {row['cluster_id']}\n"
            f"depth: {row['depth']}; seed: {'да' if seed else 'нет'}"
        )
        label = html.escape(_short_gid(gid))
        parts.extend(
            [
                f'<g><title>{title}</title><circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}" stroke="{stroke}" stroke-width="{stroke_width}"{dash}/>',
                f'<text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle" font-family="system-ui, sans-serif" font-size="10" fill="white">{label}</text></g>',
            ]
        )

    parts.append("</svg>")
    return "".join(parts)
