"""Validated Parquet input and CSV output for ApexFlow."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

import pandas as pd

INPUT_FILES = {"nodes": "nodes.parquet", "edges": "edges.parquet", "transactions": "transactions.parquet"}
OUTPUT_COLUMNS = {
    "nodes_roles": ["gid", "role", "role_score", "cluster_id", "priority_score", "evidence"],
    "clusters": ["cluster_id", "n_nodes", "n_seed", "sum_kzt_internal", "top_gids", "hypothesis"],
    "top_nodes": ["rank", "gid", "role", "priority_score", "why"],
}
ROLES = {"consolidator", "transit", "distributor", "terminal", "coordinator", "peripheral"}


def load_inputs(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    directory = Path(data_dir)
    missing = [name for name, filename in INPUT_FILES.items() if not (directory / filename).is_file()]
    if missing:
        raise ValueError(f"Не найдены входные файлы: {', '.join(missing)} в {directory}")
    frames = {name: pd.read_parquet(directory / filename) for name, filename in INPUT_FILES.items()}
    validate_inputs(frames["nodes"], frames["edges"], frames["transactions"])
    frames["transactions"] = frames["transactions"].copy()
    frames["transactions"]["date"] = pd.to_datetime(frames["transactions"]["date"], errors="raise")
    return frames["nodes"], frames["edges"], frames["transactions"]


def _require_columns(frame: pd.DataFrame, name: str, columns: set[str]) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"В {name} отсутствуют обязательные колонки: {', '.join(sorted(missing))}")


def _integer_column(frame: pd.DataFrame, column: str, table: str, *, positive: bool = False) -> None:
    values = frame[column]
    if values.isna().any() or not pd.api.types.is_integer_dtype(values):
        raise ValueError(f"{table}.{column} должен быть целочисленным без пропусков")
    if positive and (values <= 0).any():
        raise ValueError(f"{table}.{column} должен содержать только положительные значения")


def _finite_column(frame: pd.DataFrame, column: str, table: str) -> None:
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any() or not values.map(math.isfinite).all():
        raise ValueError(f"{table}.{column} должен содержать только конечные числовые значения")


def validate_inputs(nodes: pd.DataFrame, edges: pd.DataFrame, transactions: pd.DataFrame) -> None:
    _require_columns(nodes, "nodes", {"gid", "depth", "is_seed"})
    _require_columns(edges, "edges", {"src", "dst", "sum_kzt", "n_tx", "depth"})
    _require_columns(transactions, "transactions", {"src", "dst", "date", "sum_kzt"})
    for frame, name, columns in ((nodes, "nodes", ("gid", "depth")), (edges, "edges", ("src", "dst", "n_tx", "depth")), (transactions, "transactions", ("src", "dst"))):
        for column in columns:
            _integer_column(frame, column, name, positive=(name == "edges" and column == "n_tx"))
    if not pd.api.types.is_bool_dtype(nodes["is_seed"]) or nodes["is_seed"].isna().any():
        raise ValueError("nodes.is_seed должен быть настоящим bool без пропусков")
    if nodes["gid"].duplicated().any():
        raise ValueError("nodes.gid содержит дубликаты")
    for frame, name in ((nodes, "nodes"), (edges, "edges")):
        if not frame["depth"].between(0, 4).all():
            raise ValueError(f"{name}.depth содержит значения вне диапазона 0..4")
    for frame, name in ((edges, "edges"), (transactions, "transactions")):
        _finite_column(frame, "sum_kzt", name)
    dates = pd.to_datetime(transactions["date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("transactions.date содержит неразбираемые даты")
    gids = set(nodes["gid"].tolist())
    for frame, name in ((edges, "edges"), (transactions, "transactions")):
        endpoints = set(frame["src"].tolist()) | set(frame["dst"].tolist())
        unknown = endpoints - gids
        if unknown:
            raise ValueError(f"В {name} есть ссылки на отсутствующие nodes.gid: {sorted(unknown)[:5]}")
    if edges.duplicated(["src", "dst"]).any():
        raise ValueError("edges содержит повторные направленные пары; агрегируйте или выясните источник")
    _validate_aggregation(edges, transactions)


def _validate_aggregation(edges: pd.DataFrame, transactions: pd.DataFrame) -> None:
    actual = transactions.groupby(["src", "dst"], as_index=False).agg(sum_kzt=("sum_kzt", "sum"), n_tx=("sum_kzt", "size"))
    expected = edges[["src", "dst", "sum_kzt", "n_tx"]]
    merged = expected.merge(actual, on=["src", "dst"], how="outer", suffixes=("_edge", "_tx"), indicator=True)
    if not (merged["_merge"] == "both").all():
        raise ValueError("Пары edges и агрегированные transactions не совпадают")
    if not (merged["n_tx_edge"] == merged["n_tx_tx"]).all():
        raise ValueError("n_tx в edges не совпадает с количеством transactions")
    if not (merged["sum_kzt_edge"] - merged["sum_kzt_tx"]).abs().le(1e-9).all():
        raise ValueError("sum_kzt в edges не совпадает с агрегированными transactions")


def validate_outputs(results: Mapping[str, pd.DataFrame], nodes: pd.DataFrame, edges: pd.DataFrame) -> None:
    if set(results) != set(OUTPUT_COLUMNS):
        raise ValueError(f"analyze должен вернуть ровно ключи: {', '.join(OUTPUT_COLUMNS)}")
    for name, columns in OUTPUT_COLUMNS.items():
        frame = results[name]
        if not isinstance(frame, pd.DataFrame) or list(frame.columns) != columns:
            raise ValueError(f"Неверная схема {name}: ожидаются колонки {', '.join(columns)}")
    roles, clusters, top = results["nodes_roles"], results["clusters"], results["top_nodes"]
    if roles["gid"].duplicated().any() or set(roles["gid"]) != set(nodes["gid"]):
        raise ValueError("nodes_roles должен содержать каждый gid из nodes ровно один раз")
    if not roles["role"].isin(ROLES).all():
        raise ValueError("nodes_roles содержит недопустимую роль")
    for column in ("role_score", "priority_score"):
        values = pd.to_numeric(roles[column], errors="coerce")
        if values.isna().any() or not values.map(math.isfinite).all() or not values.between(0, 1).all():
            raise ValueError(f"nodes_roles.{column} должен быть конечным числом от 0 до 1")
    evidence = roles["evidence"].astype("string")
    if evidence.isna().any() or evidence.str.strip().eq("").any() or evidence.str.len().gt(200).any():
        raise ValueError("nodes_roles.evidence должен быть непустым текстом не длиннее 200 символов")
    if clusters["cluster_id"].duplicated().any() or clusters.empty:
        raise ValueError("clusters должен содержать уникальные непустые cluster_id")
    if not set(roles["cluster_id"]).issubset(set(clusters["cluster_id"])):
        raise ValueError("nodes_roles ссылается на cluster_id, отсутствующий в clusters")
    joined = roles[["gid", "cluster_id"]].merge(nodes[["gid", "is_seed"]], on="gid")
    expected = joined.groupby("cluster_id").agg(n_nodes=("gid", "size"), n_seed=("is_seed", "sum"))
    actual = clusters.set_index("cluster_id")
    if not expected.index.equals(actual.index) or not expected["n_nodes"].equals(actual["n_nodes"]) or not expected["n_seed"].equals(actual["n_seed"]):
        raise ValueError("n_nodes/n_seed в clusters не совпадают с nodes_roles и nodes")
    for _, row in clusters.iterrows():
        try:
            gids = json.loads(row["top_gids"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("clusters.top_gids должен быть JSON-массивом") from exc
        if not isinstance(gids, list) or not set(gids).issubset(set(roles.loc[roles["cluster_id"] == row["cluster_id"], "gid"])):
            raise ValueError("clusters.top_gids содержит gid не из своего кластера")
        if not isinstance(row["hypothesis"], str) or not row["hypothesis"].strip():
            raise ValueError("clusters.hypothesis должен быть непустым текстом")
    if top["gid"].duplicated().any() or len(top) < 20:
        raise ValueError("top_nodes должен содержать не менее 20 уникальных узлов")
    if top["rank"].tolist() != list(range(1, len(top) + 1)):
        raise ValueError("top_nodes.rank должен быть непрерывным от 1")
    lookup = roles.set_index("gid")
    if not set(top["gid"]).issubset(set(lookup.index)):
        raise ValueError("top_nodes содержит неизвестный gid")
    matched = top.join(lookup[["role", "priority_score"]], on="gid", rsuffix="_expected")
    if not (matched["role"] == matched["role_expected"]).all() or not (matched["priority_score"] == matched["priority_score_expected"]).all():
        raise ValueError("role или priority_score top_nodes не совпадает с nodes_roles")
    expected_top = roles.sort_values(["priority_score", "gid"], ascending=[False, True]).head(len(top))["gid"].tolist()
    if top["gid"].tolist() != expected_top or not top["why"].astype("string").str.strip().ne("").all():
        raise ValueError("top_nodes нарушает порядок priority_score/gid или содержит пустое why")


def write_outputs(results: Mapping[str, pd.DataFrame], output_dir: str | Path) -> None:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    temp_paths: list[tuple[Path, Path]] = []
    try:
        for name, columns in OUTPUT_COLUMNS.items():
            target = directory / f"{name}.csv"
            temporary = directory / f".{name}.csv.tmp"
            results[name].to_csv(temporary, columns=columns, index=False, encoding="utf-8")
            reread = pd.read_csv(temporary)
            if list(reread.columns) != columns or len(reread) != len(results[name]):
                raise ValueError(f"Не удалось повторно прочитать {name}.csv")
            temp_paths.append((temporary, target))
        for temporary, target in temp_paths:
            temporary.replace(target)
    finally:
        for temporary, _ in temp_paths:
            temporary.unlink(missing_ok=True)
