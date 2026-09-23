"""Read the actual pipeline output using the unmodified UI data boundary."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apexflow.provenance import verify_run
from ui.data import load_view_data, find_node


def check(data_dir="/app/data", output_dir="/app/output"):
    manifest = verify_run(data_dir, output_dir)
    view = load_view_data(data_dir, output_dir)
    for gid in view.nodes["gid"]:
        assert find_node(view, str(gid)) is not None
    assert manifest["report"]["nodes"] == len(view.nodes)
    assert len(view.raw_csv) == 3
    print(f"UI contract: {len(view.nodes)} nodes searchable; 3 CSV downloads; verified hashes")


if __name__ == "__main__":
    check()
