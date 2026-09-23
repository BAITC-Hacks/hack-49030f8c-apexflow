from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from apexflow.analytics import (  # noqa: E402
    CLUSTERS_COLUMNS,
    NODES_ROLES_COLUMNS,
    TOP_NODES_COLUMNS,
    analyze,
)


def _transactions() -> pd.DataFrame:
    return pd.DataFrame(columns=["src", "dst", "date", "sum_kzt"])


def test_analyze_contract_is_deterministic_and_complete() -> None:
    nodes = pd.DataFrame(
        {
            "gid": list(range(1, 23)),
            "depth": [0] + [1] * 21,
            "is_seed": [True] + [False] * 21,
        }
    )
    edges = pd.DataFrame(
        {
            "src": list(range(1, 22)),
            "dst": list(range(2, 23)),
            "sum_kzt": [10_000 + value * 1_000 for value in range(21)],
            "n_tx": [1] * 21,
        }
    )

    first = analyze(nodes, edges, _transactions())
    second = analyze(nodes.sample(frac=1, random_state=7), edges.sample(frac=1, random_state=9), _transactions())

    assert list(first["nodes_roles"].columns) == NODES_ROLES_COLUMNS
    assert list(first["clusters"].columns) == CLUSTERS_COLUMNS
    assert list(first["top_nodes"].columns) == TOP_NODES_COLUMNS
    assert first["nodes_roles"]["gid"].tolist() == list(range(1, 23))
    assert len(first["top_nodes"]) == 20
    assert first["top_nodes"]["rank"].tolist() == list(range(1, 21))
    assert first["nodes_roles"]["role"].isin(
        {"consolidator", "transit", "distributor", "terminal", "coordinator", "peripheral"}
    ).all()
    assert first["nodes_roles"]["role_score"].between(0, 1).all()
    assert first["nodes_roles"]["priority_score"].between(0, 1).all()
    assert first["nodes_roles"]["evidence"].str.len().between(1, 200).all()
    assert first["clusters"]["top_gids"].map(json.loads).map(len).le(5).all()

    for name in first:
        assert_frame_equal(first[name], second[name])


def test_depth_four_never_receives_terminal_role() -> None:
    nodes = pd.DataFrame(
        {
            "gid": [1, 2, 3, 4, 5, 99, 100],
            "depth": [0, 0, 1, 2, 3, 4, 3],
            "is_seed": [True, True, False, False, False, False, False],
        }
    )
    edges = pd.DataFrame(
        {
            "src": [1, 99, 2, 3, 4],
            "dst": [99, 3, 100, 4, 5],
            "sum_kzt": [100_000, 100_000, 100_000, 100, 100],
            "n_tx": [1, 1, 1, 1, 1],
        }
    )

    roles = analyze(nodes, edges, _transactions())["nodes_roles"].set_index("gid")

    assert roles.loc[99, "role"] != "terminal"
    assert roles.loc[99, "role"] != "transit"
    assert roles.loc[100, "role"] == "terminal"
    assert "Depth=4" in roles.loc[99, "evidence"]
