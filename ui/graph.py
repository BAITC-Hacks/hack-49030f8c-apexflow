"""Offline SVG rendering for a selected directed neighbourhood."""

from __future__ import annotations

from dataclasses import dataclass
import html
import math

import pandas as pd

from ui.data import ROLE_LABELS


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
    hidden_nodes: int = 0


def build_subgraph(
    nodes_roles: pd.DataFrame,
    edges: pd.DataFrame,
    selected_gid: str,
    *,
    mode: str = "neighborhood",
    max_edges: int = 24,
    max_nodes: int = 40,
) -> GraphSubset:
    """Select observed edges only, preserving the original ``src → dst`` direction."""

    if mode not in {"neighborhood", "cluster"} or max_edges < 1 or max_nodes < 1:
        raise ValueError("Invalid graph mode or limit.")
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
        node_keys = {selected_gid} | set(candidates["_src_key"]) | set(candidates["_dst_key"])

    ordered = candidates.assign(_amount=candidates["sum_kzt"], _src_order=candidates["_src_key"].map(int), _dst_order=candidates["_dst_key"].map(int))
    ordered = ordered.sort_values(
        ["_amount", "_src_order", "_dst_order"], ascending=[False, True, True], kind="stable"
    )
    # Always preserve selection. Prefer largest observed edges, respecting both
    # limits; then add remaining cluster members, including isolated vertices.
    visible_keys = {selected_gid}
    edge_indices = []
    for index, row in ordered.iterrows():
        keys = {row["_src_key"], row["_dst_key"]}
        if len(edge_indices) < max_edges and len(visible_keys | keys) <= max_nodes:
            visible_keys |= keys
            edge_indices.append(index)
    if mode == "cluster":
        for key in sorted(node_keys, key=int):
            if len(visible_keys) < max_nodes:
                visible_keys.add(key)
    visible_edges = candidates.loc[edge_indices].copy()
    visible_nodes = nodes_roles.loc[nodes_roles["_gid_key"].isin(visible_keys)].copy()
    visible_nodes = visible_nodes.sort_values("_gid_key", kind="stable")
    return GraphSubset(
        nodes=visible_nodes,
        edges=visible_edges,
        hidden_edges=max(len(candidates) - len(visible_edges), 0),
        mode=mode,
        hidden_nodes=len(node_keys - visible_keys),
    )


def cluster_color(cluster_id: object) -> str:
    # Stable for integer IDs and shared with the website. A palette can repeat;
    # the legend always includes cluster IDs, never implies globally unique hues.
    return CLUSTER_COLORS[int(cluster_id) % len(CLUSTER_COLORS)]


def _format_amount(value: object) -> str:
    try:
        return f"{float(value):,.2f}".replace(",", "\u202f") + " KZT"
    except (TypeError, ValueError):
        return "сумма не указана"


def render_svg(subgraph: GraphSubset, selected_gid: str, *, color_by: str = "role") -> str:
    """Render a self-contained SVG: no CDN, scripts, or unescaped file content."""

    width, height = 900, 520
    nodes = subgraph.nodes.reset_index(drop=True)
    count = len(nodes)
    positions: dict[str, tuple[float, float]] = {}
    ring = nodes.loc[nodes["_gid_key"] != selected_gid] if subgraph.mode == "neighborhood" else nodes
    orbit_x, orbit_y = width * 0.39, height * 0.35
    # Adjacent points on the narrow axis set a conservative size bound. Include
    # the seed ring in that bound so dense neighbourhoods remain distinguishable.
    spacing = 2 * orbit_y * math.sin(math.pi / max(len(ring), 2))
    radius = max(5, min(29, spacing / 2 - 9))
    label_positions: dict[str, tuple[float, float, str]] = {}
    if count == 1:
        positions[str(nodes.iloc[0]["_gid_key"])] = (width / 2, height / 2)
        label_positions[str(nodes.iloc[0]["_gid_key"])] = (width / 2, height / 2 + radius + 23, "middle")
    else:
        if subgraph.mode == "neighborhood":
            positions[selected_gid] = (width / 2, height / 2)
            label_positions[selected_gid] = (width / 2, height / 2 + radius + 23, "middle")
        for index, (_, row) in enumerate(ring.iterrows()):
            angle = -math.pi / 2 + 2 * math.pi * index / len(ring)
            gid = str(row["_gid_key"])
            positions[gid] = (
                width / 2 + orbit_x * math.cos(angle),
                height / 2 + orbit_y * math.sin(angle),
            )
            x, y = positions[gid]
            cosine = math.cos(angle)
            anchor = "start" if cosine > 0.25 else "end" if cosine < -0.25 else "middle"
            label_positions[gid] = (x + (radius + 13) * cosine, y + (radius + 18) * math.sin(angle) + 4, anchor)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="Направленный граф окружения выбранного узла">',
        "<defs><marker id=\"arrow\" viewBox=\"0 0 10 10\" refX=\"9\" refY=\"5\" markerWidth=\"7\" markerHeight=\"7\" orient=\"auto-start-reverse\"><path d=\"M 0 0 L 10 5 L 0 10 z\" fill=\"#475569\"/></marker></defs>",
        '<rect width="100%" height="100%" rx="12" fill="#f8fafc"/>',
    ]

    pairs = set(zip(subgraph.edges["_src_key"], subgraph.edges["_dst_key"]))
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
        if source == target:
            loop_size = radius + 30
            path = f"M {x1-radius*.6:.1f} {y1-radius*.8:.1f} C {x1-loop_size:.1f} {y1-loop_size:.1f}, {x1+loop_size:.1f} {y1-loop_size:.1f}, {x1+radius*.7:.1f} {y1-radius-6:.1f}"
        elif (target, source) in pairs:
            cx = (x1+x2)/2 - 38*(y2-y1)/distance
            cy = (y1+y2)/2 + 38*(x2-x1)/distance
            path = f"M {start_x:.1f} {start_y:.1f} Q {cx:.1f} {cy:.1f}, {end_x:.1f} {end_y:.1f}"
        else:
            path = f"M {start_x:.1f} {start_y:.1f} L {end_x:.1f} {end_y:.1f}"
        parts.append(f'<path d="{path}" data-src="{html.escape(source)}" data-dst="{html.escape(target)}" fill="none" stroke="#475569" stroke-width="2" marker-end="url(#arrow)"><title>{title}</title></path>')

    for index, row in nodes.iterrows():
        gid = str(row["_gid_key"])
        x, y = positions[gid]
        color = cluster_color(row["cluster_id"]) if color_by == "cluster" else ROLE_COLORS.get(row["role"], "#64748b")
        selected = gid == selected_gid
        seed = bool(row["is_seed"])
        stroke = "#0f172a" if selected else ("#facc15" if seed else "#ffffff")
        stroke_width = 5 if selected else 3
        dash = " stroke-dasharray=\"5 3\"" if seed and not selected else ""
        title = html.escape(
            f"№{index + 1}; gid: {gid}\nРоль: {ROLE_LABELS[row['role']]} ({row['role']})\nКластер: {row['cluster_id']}\n"
            f"depth: {row['depth']}; seed: {'да' if seed else 'нет'}"
        )
        label = f"№{index + 1}"
        label_x, label_y, label_anchor = label_positions[gid]
        if seed:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius+7}" fill="none" stroke="#b8860b" stroke-width="2" stroke-dasharray="5 3"/>')
        parts.extend(
            [
                f'<g data-gid="{html.escape(gid)}"><title>{title}</title><circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{color}" stroke="{stroke}" stroke-width="{stroke_width}"{dash}/>',
                f'<text x="{label_x:.1f}" y="{label_y:.1f}" text-anchor="{label_anchor}" font-family="system-ui, sans-serif" font-size="13" fill="#142229">{label}</text></g>',
            ]
        )

    parts.append("</svg>")
    return "".join(parts)


def render_legend(subgraph: GraphSubset, color_by: str) -> str:
    if color_by == "cluster":
        items = [(cluster_color(key), f"Кластер {key}") for key in sorted(set(subgraph.nodes["_cluster_key"]), key=int)]
    else:
        items = [(ROLE_COLORS[role], ROLE_LABELS[role]) for role in ROLE_COLORS if role in set(subgraph.nodes["role"])]
    return '<p>№ — локальный номер узла в текущем графе, не gid. Полный gid, роль и глубина — в подсказке при наведении.</p><div style="display:flex;flex-wrap:wrap;gap:12px">' + "".join(
        f'<span><i style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{color}"></i> {html.escape(label)}</span>'
        for color, label in items
    ) + "</div>"
