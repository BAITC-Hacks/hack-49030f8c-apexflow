"""Local provenance and aggregate-only profiling; never uploads source data."""

from __future__ import annotations

import hashlib
from importlib.metadata import distributions
import json
import os
from pathlib import Path
import platform
import tempfile

import pandas as pd

from .io import INPUT_FILES, OUTPUT_COLUMNS


def fingerprint(path: Path) -> dict:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Файл изменился во время чтения: {path.name}")
    return {"sha256": digest.hexdigest(), "bytes": after.st_size}


def input_fingerprints(directory: Path) -> dict:
    return {name: fingerprint(directory / name) for name in INPUT_FILES.values()}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            handle.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def runtime_identity() -> dict:
    root = Path(__file__).resolve().parents[2]
    paths = list((root / "src" / "apexflow").glob("*.py"))
    paths += list((root / "ui").glob("*.py"))
    paths += [root / name for name in ("app.py", "Dockerfile", "compose.yaml")]
    paths += list(root.glob("requirements*.txt"))
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(b"\0")
            # Hash text canonically so Git's CRLF checkout on Windows does not
            # change source identity relative to the same commit on Linux.
            digest.update(path.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8"))
            digest.update(b"\0")
    return {
        "revision": os.environ.get("APEXFLOW_REVISION", "unknown"),
        "source_sha256": digest.hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": dict(sorted((d.metadata["Name"], d.version) for d in distributions())),
    }


def data_profile(nodes, edges, transactions) -> dict:
    endpoints = set(edges["src"]) | set(edges["dst"])
    dates = transactions["date"]
    tables = {"nodes": nodes, "edges": edges, "transactions": transactions}
    return {
        "tables": {name: {"rows": len(frame), "dtypes": {c: str(t) for c, t in frame.dtypes.items()},
                          "missing": {c: int(v) for c, v in frame.isna().sum().items()}}
                   for name, frame in tables.items()},
        "unique_gid": int(nodes["gid"].nunique()),
        "seeds": int(nodes["is_seed"].sum()),
        "isolates": int((~nodes["gid"].isin(endpoints)).sum()),
        "depth_counts": {str(k): int(v) for k, v in nodes["depth"].value_counts().sort_index().items()},
        "date_min": None if dates.empty else dates.min().isoformat(),
        "date_max": None if dates.empty else dates.max().isoformat(),
        "self_loops": int(edges["src"].eq(edges["dst"]).sum()),
        "duplicate_edge_pairs": int(edges.duplicated(["src", "dst"]).sum()),
        "identical_transaction_rows": int(transactions.duplicated().sum()),
        "amounts": {name: {"min": None if frame.empty else float(frame["sum_kzt"].min()),
                           "max": None if frame.empty else float(frame["sum_kzt"].max()),
                           "total": float(frame["sum_kzt"].astype(float).sum())}
                    for name, frame in tables.items() if "sum_kzt" in frame},
        "edges_transactions_match": True,
        "sum_tolerance": {"relative": 1e-12, "absolute_kzt": 1e-9},
    }


def verify_run(data_dir: str | Path, output_dir: str | Path) -> dict:
    """Fail closed for a failed/running/stale/tampered run, without writing."""
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    if (output_dir / ".pipeline.lock").exists():
        raise ValueError("Pipeline выполняется или оставил lock после сбоя")
    status_bytes = (output_dir / "run_status.json").read_bytes()
    status = json.loads(status_bytes)
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if (status.get("status") != "succeeded" or manifest.get("status") != "succeeded"
            or manifest.get("schema_version") != 1 or status.get("run_id") != manifest.get("run_id")):
        raise ValueError("Последний запуск не завершился успешно; старые CSV не являются свежими")
    if manifest.get("inputs") != input_fingerprints(data_dir):
        raise ValueError("Входные данные изменились после расчёта")
    outputs = {f"{name}.csv": fingerprint(output_dir / f"{name}.csv") for name in OUTPUT_COLUMNS}
    if manifest.get("outputs") != outputs:
        raise ValueError("CSV не совпадают с хэшами успешного запуска")
    if manifest.get("runtime", {}).get("source_sha256") != runtime_identity()["source_sha256"]:
        raise ValueError("Код изменился после расчёта; пересчитайте результаты текущим образом")
    if (output_dir / ".pipeline.lock").exists() or status_bytes != (output_dir / "run_status.json").read_bytes():
        raise ValueError("Запуск изменился во время проверки; повторите после завершения pipeline")
    return manifest
