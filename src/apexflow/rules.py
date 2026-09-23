"""Explainable role and priority rules for the ApexFlow baseline."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


ROLE_ORDER = (
    "coordinator",
    "distributor",
    "consolidator",
    "terminal",
    "transit",
    "peripheral",
)


def _positive_quantile(
    series: pd.Series, quantile: float, floor: float, empty: float | None = None
) -> float:
    values = series[series > 0]
    if values.empty:
        return floor if empty is None else empty
    return max(floor, float(values.quantile(quantile)))


def derive_thresholds(features: pd.DataFrame) -> dict[str, float]:
    """Derive robust, dataset-relative thresholds without hard-coded identifiers."""
    return {
        "many_in_degree": _positive_quantile(features["in_degree"], 0.75, 3.0),
        "many_out_degree": _positive_quantile(features["out_degree"], 0.75, 5.0),
        "substantial_in_sum": _positive_quantile(features["in_sum"], 0.75, 1.0),
        "high_betweenness": _positive_quantile(
            features["betweenness"], 0.90, 0.0, empty=float("inf")
        ),
        "in_degree_scale": _positive_quantile(features["in_degree"], 0.95, 1.0),
        "out_degree_scale": _positive_quantile(features["out_degree"], 0.95, 1.0),
        "in_sum_scale": _positive_quantile(features["in_sum"], 0.95, 1.0),
        "out_sum_scale": _positive_quantile(features["out_sum"], 0.95, 1.0),
        "betweenness_scale": _positive_quantile(features["betweenness"], 0.95, 1.0),
        "seed_reach_scale": _positive_quantile(features["n_seed_reachable"], 0.95, 1.0),
        "neighbor_cluster_scale": _positive_quantile(
            features["neighbor_cluster_count"], 0.95, 1.0
        ),
    }


def _strength(series: pd.Series, scale: float) -> pd.Series:
    return (series.astype(float) / scale).clip(lower=0.0, upper=1.0)


def _minmax(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    lower, upper = float(values.min()), float(values.max())
    if upper <= lower:
        return pd.Series(0.0, index=values.index)
    return (values - lower) / (upper - lower)


def compute_priority(features: pd.DataFrame) -> pd.DataFrame:
    """Add a global review-priority score from volume, degree, centrality and reach."""
    result = features.copy()
    volume = (result["in_sum"] + result["out_sum"]).map(math.log1p)
    result["priority_score"] = (
        0.35 * _minmax(volume)
        + 0.25 * _minmax(result["unique_neighbor_count"])
        + 0.25 * _minmax(result["betweenness"])
        + 0.15 * _minmax(result["n_seed_reachable"])
    ).clip(lower=0.0, upper=1.0)
    return result


def assign_roles(features: pd.DataFrame) -> pd.DataFrame:
    """Classify nodes with conservative structural rules and role scores."""
    result = compute_priority(features)
    thresholds = derive_thresholds(result)

    has_inflow = result["in_sum"] > 0
    has_outflow = result["out_sum"] > 0
    eligible_endpoint = ~result["is_seed"] & ~result["is_boundary"]

    masks = {
        "transit": (
            eligible_endpoint
            & has_inflow
            & has_outflow
            & result["flow_ratio"].between(0.8, 1.2, inclusive="both")
        ),
        "terminal": (
            eligible_endpoint
            & has_inflow
            & (result["in_sum"] >= thresholds["substantial_in_sum"])
            & (result["flow_ratio"] <= 0.10)
        ),
        "consolidator": (
            has_inflow
            & (result["in_degree"] >= thresholds["many_in_degree"])
            & (result["in_sum"] >= thresholds["substantial_in_sum"])
        ),
        "distributor": (
            has_outflow
            & (result["out_degree"] >= thresholds["many_out_degree"])
        ),
        "coordinator": (
            (result["betweenness"] >= thresholds["high_betweenness"])
            & (result["n_seed_reachable"] >= 2)
            & (result["neighbor_cluster_count"] >= 2)
            & ~result["is_isolated"]
        ),
    }

    result["role"] = "peripheral"
    for role in ("transit", "terminal", "consolidator", "distributor", "coordinator"):
        result.loc[masks[role], "role"] = role

    in_strength = _strength(result["in_sum"], thresholds["in_sum_scale"])
    out_strength = _strength(result["out_sum"], thresholds["out_sum_scale"])
    in_degree_strength = _strength(result["in_degree"], thresholds["in_degree_scale"])
    out_degree_strength = _strength(result["out_degree"], thresholds["out_degree_scale"])
    centrality_strength = _strength(
        result["betweenness"], thresholds["betweenness_scale"]
    )
    seed_strength = _strength(
        result["n_seed_reachable"], thresholds["seed_reach_scale"]
    )
    cluster_strength = _strength(
        result["neighbor_cluster_count"], thresholds["neighbor_cluster_scale"]
    )
    transit_fit = (1 - (result["flow_ratio"] - 1).abs() / 0.2).clip(0.0, 1.0)
    terminal_fit = (1 - result["flow_ratio"] / 0.1).clip(0.0, 1.0)

    scores = pd.Series(0.0, index=result.index)
    scores.loc[result["role"] == "coordinator"] = (
        0.45 * centrality_strength + 0.30 * seed_strength + 0.25 * cluster_strength
    )
    scores.loc[result["role"] == "distributor"] = (
        0.55 * out_strength + 0.45 * out_degree_strength
    )
    scores.loc[result["role"] == "consolidator"] = (
        0.55 * in_strength + 0.45 * in_degree_strength
    )
    scores.loc[result["role"] == "terminal"] = (
        0.55 * in_strength + 0.25 * in_degree_strength + 0.20 * terminal_fit
    )
    scores.loc[result["role"] == "transit"] = (
        0.50 * transit_fit + 0.50 * pd.concat([in_strength, out_strength], axis=1).min(axis=1)
    )
    result["role_score"] = scores.clip(lower=0.0, upper=1.0)
    return result


def _money(value: Any) -> str:
    amount = float(value)
    if amount >= 1_000_000:
        return f"{amount / 1_000_000:.1f} млн KZT"
    if amount >= 1_000:
        return f"{amount / 1_000:.0f} тыс KZT"
    return f"{amount:.0f} KZT"


def _bounded(text: str, limit: int = 200) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def make_evidence(row: pd.Series) -> str:
    """Produce a short Russian explanation that states observed facts only."""
    role = row["role"]
    if role == "coordinator":
        text = (
            f"Связывает {int(row['n_seed_reachable'])} seed-направления и "
            f"{int(row['neighbor_cluster_count'])} соседних кластеров; "
            f"центральность {row['betweenness']:.3f}."
        )
    elif role == "distributor":
        text = (
            f"Отправляет {_money(row['out_sum'])} {int(row['out_degree'])} "
            "наблюдаемым получателям; есть признаки распределения."
        )
    elif role == "consolidator":
        text = (
            f"Получает от {int(row['in_degree'])} плательщиков "
            f"{_money(row['in_sum'])}; наблюдаемый отток {row['flow_ratio']:.0%}."
        )
    elif role == "terminal":
        text = (
            f"Получает {_money(row['in_sum'])} от {int(row['in_degree'])} плательщиков; "
            f"наблюдаемый отток {row['flow_ratio']:.0%}. Кандидат на конечного получателя."
        )
    elif role == "transit":
        text = (
            f"Получает {_money(row['in_sum'])}, отправляет {_money(row['out_sum'])}; "
            f"наблюдаемый коэффициент пропуска {row['flow_ratio']:.2f}."
        )
    elif bool(row["is_isolated"]):
        text = "В доступном графе нет входящих и исходящих связей."
    else:
        text = "Нет достаточного сочетания структурных признаков для отдельной роли."

    if bool(row["is_boundary"]):
        text += " Depth=4: исходящие связи могут быть обрезаны."
    return _bounded(text)


def make_priority_why(row: pd.Series) -> str:
    """Explain why a node is in the global review queue."""
    parts: list[str] = []
    turnover = float(row["in_sum"] + row["out_sum"])
    if turnover > 0:
        parts.append(f"оборот {_money(turnover)}")
    if int(row["unique_neighbor_count"]) > 0:
        parts.append(f"{int(row['unique_neighbor_count'])} контрагентов")
    if float(row["betweenness"]) > 0:
        parts.append(f"центральность {row['betweenness']:.3f}")
    if int(row["n_seed_reachable"]) > 0:
        parts.append(f"достижим из {int(row['n_seed_reachable'])} seed")
    if not parts:
        return "Нет выраженных структурных сигналов; позиция определена стабильным порядком."
    return _bounded("Приоритет: " + "; ".join(parts[:3]) + ".")
