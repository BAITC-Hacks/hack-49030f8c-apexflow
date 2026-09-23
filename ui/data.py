"""Loading and validation for the read-only ApexFlow UI data model."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import math
import re
from typing import Mapping

import pandas as pd


ROLE_CODES = {
    "consolidator",
    "transit",
    "distributor",
    "terminal",
    "coordinator",
    "peripheral",
}

ROLE_LABELS = {
    "consolidator": "Признаки консолидации",
    "transit": "Признаки транзита",
    "distributor": "Признаки распределения",
    "terminal": "Предполагаемый конечный получатель",
    "coordinator": "Структурный кандидат на координацию",
    "peripheral": "Недостаточно выраженных признаков / периферия",
}

CSV_COLUMNS = {
    "nodes_roles.csv": {
        "gid",
        "role",
        "role_score",
        "cluster_id",
        "priority_score",
        "evidence",
    },
    "clusters.csv": {
        "cluster_id",
        "n_nodes",
        "n_seed",
        "sum_kzt_internal",
        "top_gids",
        "hypothesis",
    },
    "top_nodes.csv": {"rank", "gid", "role", "priority_score", "why"},
}

NODE_COLUMNS = {"gid", "depth", "is_seed"}
EDGE_COLUMNS = {"src", "dst", "sum_kzt", "n_tx", "depth"}
INTEGER_TEXT = re.compile(r"[+-]?\d+")


class ViewDataError(ValueError):
    """A data problem that can be shown to an AML analyst without a traceback."""


@dataclass(frozen=True)
class ViewData:
    """Validated tables used by the UI.

    Identifier key columns (prefixed with an underscore) are canonical integer
    strings. They prevent a browser or CSV reader from rounding a large gid.
    """

    nodes_roles: pd.DataFrame
    clusters: pd.DataFrame
    top_nodes: pd.DataFrame
    nodes: pd.DataFrame
    edges: pd.DataFrame
    raw_csv: Mapping[str, bytes]
    source: str


def parse_gid(value: object) -> str:
    """Return a canonical gid without ever accepting float/scientific notation."""

    if value is None or pd.isna(value):
        raise ViewDataError("Введите целочисленный gid.")
    text = str(value).strip()
    if not INTEGER_TEXT.fullmatch(text):
        raise ViewDataError("gid должен быть целым числом без дробной части или экспоненты.")
    return str(int(text))


def _require_columns(
    frame: pd.DataFrame, required: set[str], label: str, *, allow_empty: bool = False
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ViewDataError(f"В файле {label} отсутствуют обязательные поля: {', '.join(missing)}.")
    if frame.empty and not allow_empty:
        raise ViewDataError(f"Файл {label} пустой.")


def _identifier_keys(frame: pd.DataFrame, column: str, label: str) -> pd.Series:
    try:
        return frame[column].map(parse_gid)
    except ViewDataError as error:
        raise ViewDataError(f"Некорректный идентификатор в {label}.{column}: {error}") from error


def _finite_scores(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    result = frame.copy()
    for column in ("role_score", "priority_score"):
        values = pd.to_numeric(result[column], errors="coerce")
        if values.isna().any() or not values.map(math.isfinite).all():
            raise ViewDataError(f"В {label}.{column} должны быть конечные числа.")
        if ((values < 0) | (values > 1)).any():
            raise ViewDataError(f"Значения {label}.{column} должны находиться в диапазоне от 0 до 1.")
        result[column] = values.astype(float)
    return result


def _raw_csv_bytes(tables: Mapping[str, pd.DataFrame]) -> dict[str, bytes]:
    return {
        filename: table.to_csv(index=False).encode("utf-8")
        for filename, table in tables.items()
    }


def validate_view_data(
    nodes_roles: pd.DataFrame,
    clusters: pd.DataFrame,
    top_nodes: pd.DataFrame,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    raw_csv: Mapping[str, bytes] | None = None,
    source: str = "Результаты pipeline",
) -> ViewData:
    """Validate the UI boundary and attach exact string keys for joins/search."""

    for name, required in CSV_COLUMNS.items():
        table = {
            "nodes_roles.csv": nodes_roles,
            "clusters.csv": clusters,
            "top_nodes.csv": top_nodes,
        }[name]
        _require_columns(table, required, name)
    _require_columns(nodes, NODE_COLUMNS, "nodes.parquet")
    # A selected node may be an isolate, and a valid small test data set can
    # therefore contain no observed edges at all.
    _require_columns(edges, EDGE_COLUMNS, "edges.parquet", allow_empty=True)

    roles = _finite_scores(nodes_roles, "nodes_roles.csv")
    roles["_gid_key"] = _identifier_keys(roles, "gid", "nodes_roles.csv")
    roles["_cluster_key"] = _identifier_keys(roles, "cluster_id", "nodes_roles.csv")
    if roles["_gid_key"].duplicated().any():
        raise ViewDataError("nodes_roles.csv содержит повторяющиеся gid.")
    unknown_roles = sorted(set(roles["role"].astype(str)) - ROLE_CODES)
    if unknown_roles:
        raise ViewDataError(f"В nodes_roles.csv есть неизвестные роли: {', '.join(unknown_roles)}.")
    if roles["evidence"].isna().any() or (roles["evidence"].astype(str).str.strip() == "").any():
        raise ViewDataError("В nodes_roles.csv у каждого узла должно быть непустое evidence.")

    source_nodes = nodes.copy()
    source_nodes["_gid_key"] = _identifier_keys(source_nodes, "gid", "nodes.parquet")
    if source_nodes["_gid_key"].duplicated().any():
        raise ViewDataError("nodes.parquet содержит повторяющиеся gid.")
    if set(source_nodes["_gid_key"]) != set(roles["_gid_key"]):
        missing = len(set(source_nodes["_gid_key"]) - set(roles["_gid_key"]))
        extra = len(set(roles["_gid_key"]) - set(source_nodes["_gid_key"]))
        raise ViewDataError(
            "Множества gid в nodes.parquet и nodes_roles.csv не совпадают "
            f"(нет результатов: {missing}; лишние результаты: {extra})."
        )

    cluster_table = clusters.copy()
    cluster_table["_cluster_key"] = _identifier_keys(cluster_table, "cluster_id", "clusters.csv")
    if cluster_table["_cluster_key"].duplicated().any():
        raise ViewDataError("clusters.csv содержит повторяющиеся cluster_id.")
    missing_clusters = set(roles["_cluster_key"]) - set(cluster_table["_cluster_key"])
    if missing_clusters:
        raise ViewDataError("Часть cluster_id из nodes_roles.csv отсутствует в clusters.csv.")

    top = top_nodes.copy()
    top["_gid_key"] = _identifier_keys(top, "gid", "top_nodes.csv")
    if top["_gid_key"].duplicated().any():
        raise ViewDataError("top_nodes.csv содержит повторяющиеся gid.")
    if not set(top["_gid_key"]).issubset(set(roles["_gid_key"])):
        raise ViewDataError("top_nodes.csv содержит gid, которого нет в nodes_roles.csv.")
    top["priority_score"] = pd.to_numeric(top["priority_score"], errors="coerce")
    if top["priority_score"].isna().any() or not top["priority_score"].map(math.isfinite).all():
        raise ViewDataError("В top_nodes.csv.priority_score должны быть конечные числа.")
    if top["rank"].duplicated().any() or set(pd.to_numeric(top["rank"], errors="coerce")) != set(range(1, len(top) + 1)):
        raise ViewDataError("rank в top_nodes.csv должен быть последовательностью от 1 без повторов.")
    role_index = roles.set_index("_gid_key")
    mismatched_roles = [
        gid
        for gid, role in zip(top["_gid_key"], top["role"], strict=True)
        if str(role) != str(role_index.at[gid, "role"])
    ]
    if mismatched_roles:
        raise ViewDataError("Роль в top_nodes.csv не совпадает с nodes_roles.csv.")
    mismatched_priority = [
        gid
        for gid, score in zip(top["_gid_key"], top["priority_score"], strict=True)
        if not math.isclose(float(score), float(role_index.at[gid, "priority_score"]), rel_tol=1e-9, abs_tol=1e-12)
    ]
    if mismatched_priority:
        raise ViewDataError("priority_score в top_nodes.csv не совпадает с nodes_roles.csv.")

    source_edges = edges.copy()
    source_edges["_src_key"] = _identifier_keys(source_edges, "src", "edges.parquet")
    source_edges["_dst_key"] = _identifier_keys(source_edges, "dst", "edges.parquet")
    edge_gids = set(source_edges["_src_key"]) | set(source_edges["_dst_key"])
    unknown_edge_gids = edge_gids - set(source_nodes["_gid_key"])
    if unknown_edge_gids:
        raise ViewDataError("edges.parquet ссылается на gid, которого нет в nodes.parquet.")

    # The UI only enriches the results with observed source attributes. It never
    # recalculates role, cluster or priority.
    enriched_roles = roles.merge(
        source_nodes[["_gid_key", "depth", "is_seed"]],
        on="_gid_key",
        how="left",
        validate="one_to_one",
    )
    enriched_roles = enriched_roles.sort_values("_gid_key", kind="stable").reset_index(drop=True)
    cluster_table = cluster_table.sort_values("_cluster_key", kind="stable").reset_index(drop=True)
    top = top.sort_values(["rank", "_gid_key"], kind="stable").reset_index(drop=True)

    generated_csv = _raw_csv_bytes(
        {
            "nodes_roles.csv": nodes_roles,
            "clusters.csv": clusters,
            "top_nodes.csv": top_nodes,
        }
    )
    return ViewData(
        nodes_roles=enriched_roles,
        clusters=cluster_table,
        top_nodes=top,
        nodes=source_nodes,
        edges=source_edges,
        raw_csv=dict(raw_csv or generated_csv),
        source=source,
    )


def expected_paths(data_dir: str | Path, output_dir: str | Path) -> dict[str, Path]:
    """Return every input expected by the UI, with no computer-specific paths."""

    data_path = Path(data_dir)
    output_path = Path(output_dir)
    return {
        "nodes.parquet": data_path / "nodes.parquet",
        "edges.parquet": data_path / "edges.parquet",
        "nodes_roles.csv": output_path / "nodes_roles.csv",
        "clusters.csv": output_path / "clusters.csv",
        "top_nodes.csv": output_path / "top_nodes.csv",
    }


def file_signature(data_dir: str | Path, output_dir: str | Path) -> tuple[tuple[str, int, int], ...]:
    """A cache key that changes when pipeline output or source inputs change."""

    paths = expected_paths(data_dir, output_dir)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ViewDataError("Не найдены обязательные файлы:\n" + "\n".join(missing))
    return tuple(
        (name, path.stat().st_mtime_ns, path.stat().st_size)
        for name, path in sorted(paths.items())
    )


def load_view_data(data_dir: str | Path, output_dir: str | Path) -> ViewData:
    """Load the pipeline's immutable output and matching source tables."""

    paths = expected_paths(data_dir, output_dir)
    # Check paths first so a missing file becomes a clear UI state instead of a
    # library-specific FileNotFoundError.
    file_signature(data_dir, output_dir)
    try:
        nodes_roles = pd.read_csv(paths["nodes_roles.csv"], dtype={"gid": "string", "cluster_id": "string"})
        clusters = pd.read_csv(paths["clusters.csv"], dtype={"cluster_id": "string"})
        top_nodes = pd.read_csv(paths["top_nodes.csv"], dtype={"gid": "string"})
        nodes = pd.read_parquet(paths["nodes.parquet"])
        edges = pd.read_parquet(paths["edges.parquet"])
    except Exception as error:  # pandas exposes several engine-specific errors
        raise ViewDataError(f"Не удалось прочитать входные файлы: {error}") from error

    raw_csv = {
        filename: paths[filename].read_bytes()
        for filename in ("nodes_roles.csv", "clusters.csv", "top_nodes.csv")
    }
    return validate_view_data(
        nodes_roles,
        clusters,
        top_nodes,
        nodes,
        edges,
        raw_csv=raw_csv,
        source="Результаты последнего расчёта pipeline",
    )


def find_node(data: ViewData, raw_gid: object) -> pd.Series | None:
    """Find a node in the complete result, independent of visible filters/top."""

    key = parse_gid(raw_gid)
    matches = data.nodes_roles.loc[data.nodes_roles["_gid_key"] == key]
    return None if matches.empty else matches.iloc[0]


def synthetic_view_data() -> ViewData:
    """Build the explicitly synthetic development sample lazily to avoid cycles."""

    from ui.fixtures import build_synthetic_view_data

    return build_synthetic_view_data()
