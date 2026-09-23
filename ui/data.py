"""Read-only UI boundary, sharing the pipeline's contract checks."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re
from typing import Mapping

import pandas as pd

from apexflow.io import OUTPUT_COLUMNS, validate_graph_inputs, validate_outputs


ROLE_LABELS = {
    "consolidator": "Признаки консолидации",
    "transit": "Признаки транзита",
    "distributor": "Признаки распределения",
    "terminal": "Предполагаемый конечный получатель",
    "coordinator": "Структурный кандидат на координацию",
    "peripheral": "Недостаточно выраженных признаков / периферия",
}
ROLE_CODES = set(ROLE_LABELS)
INTEGER_TEXT = re.compile(r"[+-]?[0-9]+")


class ViewDataError(ValueError):
    """A data problem that can be presented without a traceback."""


@dataclass(frozen=True)
class ViewData:
    nodes_roles: pd.DataFrame
    clusters: pd.DataFrame
    top_nodes: pd.DataFrame
    nodes: pd.DataFrame
    edges: pd.DataFrame
    raw_csv: Mapping[str, bytes]
    source: str


def parse_gid(value: object) -> str:
    """Canonical signed int64, with no float or browser-number round trip."""
    if not pd.api.types.is_scalar(value) or value is None or pd.isna(value):
        raise ViewDataError("Введите целочисленный gid.")
    text = str(value).strip()
    if len(text) > 64 or not INTEGER_TEXT.fullmatch(text):
        raise ViewDataError("gid должен быть целым числом без дробной части или экспоненты.")
    integer = int(text)
    if not -(2**63) <= integer < 2**63:
        raise ViewDataError("gid выходит за диапазон int64.")
    return str(integer)


def _decode_results(tables: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Decode CSV scalars exactly; the common validator checks their semantics."""
    result = {}
    integer_columns = {"gid", "cluster_id", "rank", "n_nodes", "n_seed"}
    number_columns = {"role_score", "priority_score", "sum_kzt_internal"}
    for name, columns in OUTPUT_COLUMNS.items():
        frame = tables[name].copy()
        if list(frame.columns) != columns:
            raise ViewDataError(f"Неверная схема {name}.csv: ожидаются {', '.join(columns)}")
        for column in integer_columns.intersection(columns):
            frame[column] = frame[column].map(lambda value: int(parse_gid(value))).astype("int64")
        for column in number_columns.intersection(columns):
            if frame[column].map(lambda value: isinstance(value, bool)).any():
                raise ViewDataError(f"{name}.{column}: bool не является суммой/score")
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        result[name] = frame
    return result


def validate_view_data(
    nodes_roles: pd.DataFrame, clusters: pd.DataFrame, top_nodes: pd.DataFrame,
    nodes: pd.DataFrame, edges: pd.DataFrame, *,
    raw_csv: Mapping[str, bytes] | None = None, source: str = "CSV с диска",
) -> ViewData:
    tables = {"nodes_roles": nodes_roles, "clusters": clusters, "top_nodes": top_nodes}
    try:
        results = _decode_results(tables)
        validate_graph_inputs(nodes, edges)
        validate_outputs(results, nodes, edges)
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        raise ViewDataError(str(error)) from error

    roles = results["nodes_roles"].merge(nodes[["gid", "depth", "is_seed"]], on="gid", validate="one_to_one")
    roles = roles.sort_values("gid", kind="stable").reset_index(drop=True)
    cluster_table = results["clusters"].sort_values("cluster_id", kind="stable").copy()
    top = results["top_nodes"].copy()
    source_nodes, source_edges = nodes.copy(), edges.copy()
    # Every identifier sent to the browser is text, including source endpoints.
    for frame in (roles, top, source_nodes):
        frame["gid"] = frame["gid"].map(str)
        frame["_gid_key"] = frame["gid"]
    for frame in (roles, cluster_table):
        frame["cluster_id"] = frame["cluster_id"].map(str)
        frame["_cluster_key"] = frame["cluster_id"]
    for column in ("src", "dst"):
        source_edges[column] = source_edges[column].map(str)
        source_edges[f"_{column}_key"] = source_edges[column]

    csv_bytes = dict(raw_csv) if raw_csv is not None else {
        f"{name}.csv": frame.to_csv(index=False).encode("utf-8") for name, frame in results.items()
    }
    return ViewData(roles, cluster_table, top, source_nodes, source_edges, csv_bytes, source)


def expected_paths(data_dir: str | Path, output_dir: str | Path) -> dict[str, Path]:
    return {
        "nodes.parquet": Path(data_dir) / "nodes.parquet",
        "edges.parquet": Path(data_dir) / "edges.parquet",
        **{f"{name}.csv": Path(output_dir) / f"{name}.csv" for name in OUTPUT_COLUMNS},
    }


def file_signature(data_dir: str | Path, output_dir: str | Path) -> tuple[tuple[str, int, int], ...]:
    signature = []
    for name, path in sorted(expected_paths(data_dir, output_dir).items()):
        try:
            stat = path.stat()
            if not path.is_file():
                raise OSError("ожидался файл")
        except OSError as error:
            raise ViewDataError(f"Не удалось открыть {path}: {error}") from error
        signature.append((name, stat.st_mtime_ns, stat.st_size))
    return tuple(signature)


def load_view_data(data_dir: str | Path, output_dir: str | Path) -> ViewData:
    paths = expected_paths(data_dir, output_dir)
    before = file_signature(data_dir, output_dir)
    try:
        # Parse and download the same snapshot, never read each CSV twice.
        raw_csv = {f"{name}.csv": paths[f"{name}.csv"].read_bytes() for name in OUTPUT_COLUMNS}
        tables = {
            name: pd.read_csv(BytesIO(raw_csv[f"{name}.csv"]), dtype=str, keep_default_na=False)
            for name in OUTPUT_COLUMNS
        }
        nodes = pd.read_parquet(paths["nodes.parquet"])
        edges = pd.read_parquet(paths["edges.parquet"])
    except Exception as error:  # engine-specific decode errors are user-facing
        raise ViewDataError(f"Не удалось прочитать входные файлы: {error}") from error
    if before != file_signature(data_dir, output_dir):
        raise ViewDataError("Файлы изменились во время чтения. Дождитесь завершения pipeline и обновите данные.")
    return validate_view_data(
        tables["nodes_roles"], tables["clusters"], tables["top_nodes"], nodes, edges,
        raw_csv=raw_csv, source="CSV с диска; свежесть расчёта проверяется отдельным запуском pipeline",
    )


def find_node(data: ViewData, raw_gid: object) -> pd.Series | None:
    key = parse_gid(raw_gid)
    matches = data.nodes_roles.loc[data.nodes_roles["_gid_key"] == key]
    return None if matches.empty else matches.iloc[0]


def synthetic_view_data() -> ViewData:
    from ui.fixtures import build_synthetic_view_data
    return build_synthetic_view_data()
