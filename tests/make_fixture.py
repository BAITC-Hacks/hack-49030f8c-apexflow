"""Generate an explicitly synthetic dataset; never overwrite an existing file."""
from pathlib import Path
import sys

import pandas as pd


def make_fixture(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    if any((directory / f"{name}.parquet").exists() for name in ("nodes", "edges", "transactions")):
        raise ValueError("Fixture destination already contains input files")
    gids = [9007199254740993 + i for i in range(24)]
    nodes = pd.DataFrame({"gid": pd.Series(gids, dtype="int64"),
                          "depth": [0] + [1] * 20 + [4, 0, 2],
                          "is_seed": [True] + [False] * 21 + [True, False]})
    transactions = pd.DataFrame({"src": gids[:21], "dst": gids[1:22],
                                 "date": ["2026-07-01"] * 21, "sum_kzt": [100.25] * 21})
    edges = transactions[["src", "dst", "sum_kzt"]].assign(n_tx=1, depth=1)
    for name, frame in (("nodes", nodes), ("edges", edges), ("transactions", transactions)):
        frame.to_parquet(directory / f"{name}.parquet", index=False)


if __name__ == "__main__":
    make_fixture(Path(sys.argv[1]))
