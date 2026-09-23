"""Verify the promised Docker read-only mounts without writing any bytes."""
import errno
from pathlib import Path
import sys


def require_readonly(path):
    try:
        with Path(path).open("ab"):
            pass
    except OSError as exc:
        if exc.errno not in (errno.EROFS, errno.EACCES, errno.EPERM):
            raise
    else:
        raise AssertionError(f"Expected a read-only mount: {path}")


if __name__ == "__main__":
    for filename in ("nodes.parquet", "edges.parquet", "transactions.parquet"):
        require_readonly(Path("/app/data") / filename)
    if "--ui" in sys.argv:
        require_readonly("/app/output/nodes_roles.csv")
    print("Read-only mounts verified")
