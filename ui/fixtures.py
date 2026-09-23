"""An explicit synthetic dataset for developing the UI before pipeline output exists."""

from __future__ import annotations

import json

import pandas as pd

from ui.data import ViewData, validate_view_data


def build_synthetic_view_data() -> ViewData:
    """Return a small, clearly labelled example with a boundary node and isolate."""

    gids = [
        "9007199254740993",
        "10000000000000002",
        "10000000000000003",
        "10000000000000004",
        "10000000000000005",
        "10000000000000006",
        "10000000000000007",
    ]
    nodes = pd.DataFrame(
        {
            "gid": gids,
            "depth": [0, 1, 2, 3, 4, 2, 1],
            "is_seed": [True, False, False, False, False, False, False],
        }
    )
    edges = pd.DataFrame(
        {
            "src": [gids[0], gids[1], gids[1], gids[2], gids[5]],
            "dst": [gids[1], gids[2], gids[3], gids[4], gids[1]],
            "sum_kzt": [125000.0, 82000.0, 39000.0, 76000.0, 15000.0],
            "n_tx": [4, 2, 1, 3, 1],
            "depth": [1, 2, 2, 3, 2],
        }
    )
    nodes_roles = pd.DataFrame(
        {
            "gid": gids,
            "role": ["coordinator", "consolidator", "transit", "distributor", "terminal", "peripheral", "peripheral"],
            "role_score": [0.73, 0.86, 0.77, 0.68, 0.81, 0.22, 0.05],
            "cluster_id": [1, 1, 1, 1, 1, 1, 2],
            "priority_score": [0.82, 0.94, 0.88, 0.74, 0.79, 0.31, 0.08],
            "evidence": [
                "Синтетический пример: стартовый узел с наблюдаемыми исходящими связями.",
                "Синтетический пример: несколько входящих и исходящих контрагентов.",
                "Синтетический пример: входящий и исходящий поток в окружении.",
                "Синтетический пример: один источник и несколько направлений проверки.",
                "Синтетический пример: узел на границе глубины выборки.",
                "Синтетический пример: малое наблюдаемое окружение.",
                "Синтетический пример: изолированный узел без наблюдаемых связей.",
            ],
        }
    )
    clusters = pd.DataFrame(
        {
            "cluster_id": [1, 2],
            "n_nodes": [6, 1],
            "n_seed": [1, 0],
            "sum_kzt_internal": [337000.0, 0.0],
            "top_gids": [json.dumps(gids[:5]), json.dumps([gids[6]])],
            "hypothesis": [
                "Синтетическая группа со связанными переводами; не является результатом анализа.",
                "Синтетический изолят без наблюдаемых связей.",
            ],
        }
    )
    order = [gids[1], gids[2], gids[0], gids[4], gids[3], gids[5], gids[6]]
    rank_lookup = {gid: rank for rank, gid in enumerate(order, start=1)}
    top_nodes = (
        nodes_roles[["gid", "role", "priority_score", "evidence"]]
        .assign(rank=lambda frame: frame["gid"].map(rank_lookup))
        .sort_values("rank")
        .rename(columns={"evidence": "why"})[["rank", "gid", "role", "priority_score", "why"]]
    )
    return validate_view_data(
        nodes_roles,
        clusters,
        top_nodes,
        nodes,
        edges,
        source="Синтетический пример для разработки — не результаты анализа исходных данных",
    )
