"""Validated Parquet input and CSV output for ApexFlow."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Mapping

import pandas as pd

INPUT_FILES = {"nodes": "nodes.parquet", "edges": "edges.parquet", "transactions": "transactions.parquet"}
OUTPUT_COLUMNS = {
    "nodes_roles": ["gid", "role", "role_score", "cluster_id", "priority_score", "evidence"],
    "clusters": ["cluster_id", "n_nodes", "n_seed", "sum_kzt_internal", "top_gids", "hypothesis"],
    "top_nodes": ["rank", "gid", "role", "priority_score", "why"],
}
ROLES = {"consolidator", "transit", "distributor", "terminal", "coordinator", "peripheral"}
RUN_MANIFEST = "run_manifest.json"


def file_hashes(directory: str | Path, filenames) -> dict[str, str]:
    """Hash the actual files without interpreting or publishing their contents."""
    hashes = {}
    for filename in filenames:
        with (Path(directory) / filename).open("rb") as source:
            hashes[filename] = hashlib.file_digest(source, "sha256").hexdigest()
    return hashes


def write_run_manifest(output_dir: str | Path, manifest: dict) -> None:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, prefix=".run.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(directory / RUN_MANIFEST)
    finally:
        temporary.unlink(missing_ok=True)


def validate_run_manifest(data_dir: str | Path, output_dir: str | Path) -> dict:
    """Reject unfinished runs and CSVs calculated from a different input snapshot."""
    try:
        manifest = json.loads((Path(output_dir) / RUN_MANIFEST).read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
            raise ValueError("Последний расчёт не завершён успешно. Выполните pipeline заново.")
        if manifest.get("inputs") != file_hashes(data_dir, INPUT_FILES.values()):
            raise ValueError("Исходные данные изменились после расчёта. Выполните pipeline заново.")
        if manifest.get("outputs") != file_hashes(output_dir, [f"{name}.csv" for name in OUTPUT_COLUMNS]):
            raise ValueError("CSV изменились после расчёта. Выполните pipeline заново.")
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Не удалось подтвердить свежесть результатов: {exc}. Выполните pipeline заново.") from exc
    return manifest


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
    if frame.columns.duplicated().any():
        raise ValueError(f"{name} содержит повторные колонки")
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"В {name} отсутствуют обязательные колонки: {', '.join(sorted(missing))}")


def _integer_column(frame: pd.DataFrame, column: str, table: str, *, positive: bool = False) -> None:
    values = frame[column]
    if values.isna().any() or not pd.api.types.is_integer_dtype(values):
        raise ValueError(f"{table}.{column} должен быть целочисленным без пропусков")
    if any(not -(2**63) <= int(value) < 2**63 for value in values):
        raise ValueError(f"{table}.{column} выходит за диапазон int64")
    if positive and (values <= 0).any():
        raise ValueError(f"{table}.{column} должен содержать только положительные значения")


def _finite_column(frame: pd.DataFrame, column: str, table: str) -> None:
    values = frame[column]
    if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values) or pd.api.types.is_complex_dtype(values):
        raise ValueError(f"{table}.{column} должен содержать числовые значения, не строки/bool")
    if values.isna().any() or not values.map(math.isfinite).all():
        raise ValueError(f"{table}.{column} должен содержать только конечные числовые значения")
    if (values < 0).any():
        raise ValueError(f"{table}.{column} должен содержать неотрицательные значения")


def _money_sum(values) -> float:
    try:
        return math.fsum(float(value) for value in values)
    except OverflowError as exc:
        raise ValueError("Суммарный оборот выходит за конечный диапазон чисел") from exc


def _same_money(left: float, right: float) -> bool:
    # Allow only floating-point accumulation noise, not a percentage of turnover.
    tolerance = max(1e-9, 4 * math.ulp(float(left)), 4 * math.ulp(float(right)))
    return abs(float(left) - float(right)) <= tolerance


def validate_graph_inputs(nodes: pd.DataFrame, edges: pd.DataFrame) -> None:
    """Shared read-only boundary for the pipeline and UI (including isolates)."""
    _require_columns(nodes, "nodes", {"gid", "depth", "is_seed"})
    _require_columns(edges, "edges", {"src", "dst", "sum_kzt", "n_tx", "depth"})
    if nodes.empty:
        raise ValueError("nodes не содержит узлов")
    for frame, name, columns in ((nodes, "nodes", ("gid", "depth")), (edges, "edges", ("src", "dst", "n_tx", "depth"))):
        for column in columns:
            _integer_column(frame, column, name, positive=(name == "edges" and column == "n_tx"))
    if not pd.api.types.is_bool_dtype(nodes["is_seed"]) or nodes["is_seed"].isna().any():
        raise ValueError("nodes.is_seed должен быть настоящим bool без пропусков")
    if nodes["gid"].duplicated().any():
        raise ValueError("nodes.gid содержит дубликаты")
    for frame, name in ((nodes, "nodes"), (edges, "edges")):
        if not frame["depth"].between(0, 4).all():
            raise ValueError(f"{name}.depth содержит значения вне диапазона 0..4")
    _finite_column(edges, "sum_kzt", "edges")
    _money_sum(edges["sum_kzt"])
    _validate_endpoints(nodes, edges, "edges")
    if edges.duplicated(["src", "dst"]).any():
        raise ValueError("edges содержит повторные направленные пары; выясните источник")


def _validate_endpoints(nodes: pd.DataFrame, frame: pd.DataFrame, name: str) -> None:
    unknown = (set(frame["src"]) | set(frame["dst"])) - set(nodes["gid"])
    if unknown:
        raise ValueError(f"В {name} есть ссылки на отсутствующие nodes.gid: {sorted(unknown)[:5]}")


def validate_inputs(nodes: pd.DataFrame, edges: pd.DataFrame, transactions: pd.DataFrame) -> None:
    validate_graph_inputs(nodes, edges)
    _require_columns(transactions, "transactions", {"src", "dst", "date", "sum_kzt"})
    for column in ("src", "dst"):
        _integer_column(transactions, column, "transactions")
    _finite_column(transactions, "sum_kzt", "transactions")
    _money_sum(transactions["sum_kzt"])
    if not transactions.empty and pd.api.types.is_numeric_dtype(transactions["date"]):
        raise ValueError("transactions.date должен содержать даты, а не числовые метки без единиц")
    dates = pd.to_datetime(transactions["date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("transactions.date содержит неразбираемые даты")
    _validate_endpoints(nodes, transactions, "transactions")
    _validate_aggregation(edges, transactions)


def _validate_aggregation(edges: pd.DataFrame, transactions: pd.DataFrame) -> None:
    actual = transactions.groupby(["src", "dst"], as_index=False).agg(sum_kzt=("sum_kzt", _money_sum), n_tx=("sum_kzt", "size"))
    expected = edges[["src", "dst", "sum_kzt", "n_tx"]]
    merged = expected.merge(actual, on=["src", "dst"], how="outer", suffixes=("_edge", "_tx"), indicator=True)
    if not (merged["_merge"] == "both").all():
        raise ValueError("Пары edges и агрегированные transactions не совпадают")
    if not (merged["n_tx_edge"] == merged["n_tx_tx"]).all():
        raise ValueError("n_tx в edges не совпадает с количеством transactions")
    if pd.api.types.is_integer_dtype(edges["sum_kzt"]) and pd.api.types.is_integer_dtype(transactions["sum_kzt"]):
        # Keep Python integers: float64 cannot distinguish adjacent sums above 2**53.
        exact_totals = {
            pair: sum(int(value) for value in amounts)
            for pair, amounts in transactions.groupby(["src", "dst"])["sum_kzt"]
        }
        amounts_match = all(
            amount == exact_totals[(src, dst)]
            for src, dst, amount in edges[["src", "dst", "sum_kzt"]].itertuples(index=False, name=None)
        )
    else:
        amounts_match = all(_same_money(edge, tx) for edge, tx in zip(merged["sum_kzt_edge"], merged["sum_kzt_tx"]))
    if not amounts_match:
        raise ValueError("sum_kzt в edges не совпадает с агрегированными transactions")


def validate_outputs(results: Mapping[str, pd.DataFrame], nodes: pd.DataFrame, edges: pd.DataFrame) -> None:
    if set(results) != set(OUTPUT_COLUMNS):
        raise ValueError(f"analyze должен вернуть ровно ключи: {', '.join(OUTPUT_COLUMNS)}")
    for name, columns in OUTPUT_COLUMNS.items():
        frame = results[name]
        if not isinstance(frame, pd.DataFrame) or list(frame.columns) != columns:
            raise ValueError(f"Неверная схема {name}: ожидаются колонки {', '.join(columns)}")
    roles, clusters, top = results["nodes_roles"], results["clusters"], results["top_nodes"]
    for frame, name, columns in (
        (roles, "nodes_roles", ("gid", "cluster_id")),
        (clusters, "clusters", ("cluster_id", "n_nodes", "n_seed")),
        (top, "top_nodes", ("rank", "gid")),
    ):
        for column in columns:
            _integer_column(frame, column, name)
    if roles["gid"].duplicated().any() or set(roles["gid"]) != set(nodes["gid"]):
        raise ValueError("nodes_roles должен содержать каждый gid из nodes ровно один раз")
    if not roles["gid"].is_monotonic_increasing or not clusters["cluster_id"].is_monotonic_increasing:
        raise ValueError("nodes_roles и clusters должны быть отсортированы по gid и cluster_id")
    if not roles["role"].isin(ROLES).all():
        raise ValueError("nodes_roles содержит недопустимую роль")
    for column in ("role_score", "priority_score"):
        _finite_column(roles, column, "nodes_roles")
        values = roles[column]
        if values.isna().any() or not values.map(math.isfinite).all() or not values.between(0, 1).all():
            raise ValueError(f"nodes_roles.{column} должен быть конечным числом от 0 до 1")
    _text_column(roles, "evidence", "nodes_roles")
    evidence = roles["evidence"].astype("string")
    if evidence.isna().any() or evidence.str.strip().eq("").any() or evidence.str.len().gt(200).any():
        raise ValueError("nodes_roles.evidence должен быть непустым текстом не длиннее 200 символов")
    if clusters["cluster_id"].duplicated().any() or clusters.empty:
        raise ValueError("clusters должен содержать уникальные непустые cluster_id")
    if not set(roles["cluster_id"]).issubset(set(clusters["cluster_id"])):
        raise ValueError("nodes_roles ссылается на cluster_id, отсутствующий в clusters")
    joined = roles[["gid", "cluster_id"]].merge(nodes[["gid", "is_seed"]], on="gid")
    expected = joined.groupby("cluster_id").agg(n_nodes=("gid", "size"), n_seed=("is_seed", "sum"))
    actual = clusters.set_index("cluster_id").sort_index()
    if not expected.index.equals(actual.index) or not (expected["n_nodes"].to_numpy() == actual["n_nodes"].to_numpy()).all() or not (expected["n_seed"].to_numpy() == actual["n_seed"].to_numpy()).all():
        raise ValueError("n_nodes/n_seed в clusters не совпадают с nodes_roles и nodes")
    _finite_column(clusters, "sum_kzt_internal", "clusters")
    lookup = roles.set_index("gid")
    internal = edges.assign(_src_cluster=edges["src"].map(lookup["cluster_id"]), _dst_cluster=edges["dst"].map(lookup["cluster_id"]))
    sums = internal.loc[internal["_src_cluster"] == internal["_dst_cluster"]].groupby("_src_cluster")["sum_kzt"].agg(_money_sum)
    for cluster_id, total in actual["sum_kzt_internal"].items():
        if not _same_money(total, sums.get(cluster_id, 0)):
            raise ValueError("clusters.sum_kzt_internal не совпадает с внутренними рёбрами")
    for _, row in clusters.iterrows():
        try:
            gids = json.loads(row["top_gids"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("clusters.top_gids должен быть JSON-массивом") from exc
        if not isinstance(gids, list) or any(type(gid) is not int for gid in gids):
            raise ValueError("clusters.top_gids должен быть JSON-массивом целых gid")
        if not 1 <= len(gids) <= 5:
            raise ValueError("clusters.top_gids должен содержать от 1 до 5 gid")
        members = roles.loc[roles["cluster_id"] == row["cluster_id"]]
        if len(set(gids)) != len(gids) or not set(gids).issubset(set(members["gid"])):
            raise ValueError("clusters.top_gids содержит gid не из своего кластера")
        ordered = members.sort_values(["priority_score", "gid"], ascending=[False, True])["gid"].head(len(gids)).tolist()
        if gids != ordered:
            raise ValueError("clusters.top_gids нарушает порядок приоритета/gid")
        if not isinstance(row["hypothesis"], str) or not row["hypothesis"].strip():
            raise ValueError("clusters.hypothesis должен быть непустым текстом")
    if top["gid"].duplicated().any() or len(top) < min(20, len(nodes)):
        raise ValueError("top_nodes должен содержать не менее min(20, N) уникальных узлов")
    _finite_column(top, "priority_score", "top_nodes")
    _text_column(top, "why", "top_nodes")
    if top["rank"].tolist() != list(range(1, len(top) + 1)):
        raise ValueError("top_nodes.rank должен быть непрерывным от 1")
    if not set(top["gid"]).issubset(set(lookup.index)):
        raise ValueError("top_nodes содержит неизвестный gid")
    matched = top.join(lookup[["role", "priority_score"]], on="gid", rsuffix="_expected")
    if not (matched["role"] == matched["role_expected"]).all() or not (matched["priority_score"] == matched["priority_score_expected"]).all():
        raise ValueError("role или priority_score top_nodes не совпадает с nodes_roles")
    expected_top = roles.sort_values(["priority_score", "gid"], ascending=[False, True]).head(len(top))["gid"].tolist()
    if top["gid"].tolist() != expected_top or not top["why"].astype("string").str.strip().ne("").all():
        raise ValueError("top_nodes нарушает порядок priority_score/gid или содержит пустое why")


def _text_column(frame: pd.DataFrame, column: str, table: str) -> None:
    if not frame[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError(f"{table}.{column} должен быть непустым текстом")


def write_outputs(results: Mapping[str, pd.DataFrame], output_dir: str | Path) -> None:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    temp_paths: list[tuple[Path, Path]] = []
    try:
        for name, columns in OUTPUT_COLUMNS.items():
            target = directory / f"{name}.csv"
            with tempfile.NamedTemporaryFile(dir=directory, prefix=f".{name}.", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
            temp_paths.append((temporary, target))
            results[name].to_csv(temporary, columns=columns, index=False, encoding="utf-8")
            # Read identifiers as strings so the check never passes through float.
            reread = pd.read_csv(temporary, dtype=str, keep_default_na=False)
            if list(reread.columns) != columns or len(reread) != len(results[name]):
                raise ValueError(f"Не удалось повторно прочитать {name}.csv")
            for column in ("gid", "cluster_id", "evidence", "why", "top_gids", "hypothesis"):
                if column in columns and reread[column].tolist() != results[name][column].map(str).tolist():
                    raise ValueError(f"При записи {name}.{column} потеряны значения")
        for temporary, target in temp_paths:
            temporary.replace(target)
    finally:
        for temporary, _ in temp_paths:
            temporary.unlink(missing_ok=True)
