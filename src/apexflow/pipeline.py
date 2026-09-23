"""Orchestration for a single, reproducible ApexFlow analysis."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from datetime import datetime, timezone
from uuid import uuid4

from .io import OUTPUT_COLUMNS, load_inputs, validate_outputs, write_outputs
from .provenance import atomic_json, data_profile, fingerprint, input_fingerprints, runtime_identity


def run_pipeline(data_dir: str | Path, output_dir: str | Path, *, acceptance: bool = False) -> dict:
    started = perf_counter()
    data_dir, output_dir = Path(data_dir).resolve(), Path(output_dir).resolve()
    if data_dir == output_dir or data_dir in output_dir.parents or output_dir in data_dir.parents:
        raise ValueError("Каталоги data и output должны быть раздельными и не вложенными")
    output_dir.mkdir(parents=True, exist_ok=True)
    lock = output_dir / ".pipeline.lock"
    try:
        handle = lock.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise RuntimeError("Другой pipeline использует output; после аварии проверьте процесс и lock") from exc
    run_id = uuid4().hex
    status = {"run_id": run_id, "status": "running", "started_at": datetime.now(timezone.utc).isoformat()}
    stage = "prepare"
    timings = {}
    with handle:
        handle.write(run_id)
        handle.flush()
        try:
            atomic_json(output_dir / "run_status.json", status)
            stage = "load_validate"
            tick = perf_counter()
            inputs = input_fingerprints(data_dir)
            nodes, edges, transactions = load_inputs(data_dir)
            profile = data_profile(nodes, edges, transactions)
            if acceptance:
                expected = (2248, 3119, 4840, 81)
                actual = (len(nodes), len(edges), len(transactions), int(nodes["is_seed"].sum()))
                if actual != expected:
                    raise ValueError(f"Приёмка полного набора: ожидалось {expected}, получено {actual}")
                dates = transactions["date"]
                if not (dates.dt.year.eq(2026) & dates.dt.month.eq(7)).all():
                    raise ValueError("Приёмка полного набора: ожидался июль 2026")
            timings[stage] = perf_counter() - tick
            stage = "analyze"
            tick = perf_counter()
            from .analytics import analyze
            results = analyze(nodes, edges, transactions)
            timings[stage] = perf_counter() - tick
            stage = "validate_outputs"
            tick = perf_counter()
            validate_outputs(results, nodes, edges)
            if inputs != input_fingerprints(data_dir):
                raise ValueError("Входные файлы изменились во время расчёта; повторите запуск")
            timings[stage] = perf_counter() - tick
            stage = "publish"
            tick = perf_counter()
            write_outputs(results, output_dir)
            outputs = {f"{name}.csv": fingerprint(output_dir / f"{name}.csv") for name in OUTPUT_COLUMNS}
            atomic_json(output_dir / "data_profile.json", {"inputs": inputs, **profile})
            timings[stage] = perf_counter() - tick
            report = {
                "nodes": len(nodes), "clusters": len(results["clusters"]), "top_nodes": len(results["top_nodes"]),
                "output": str(output_dir), "run_id": run_id,
            }
            runtime = runtime_identity()
            seconds = perf_counter() - started
            if acceptance and seconds >= 300:
                raise ValueError(f"Приёмка: pipeline занял {seconds:.3f} с, требуется <300 с")
            report["seconds"] = seconds
            manifest = {"schema_version": 1, **status, "status": "succeeded",
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                        "inputs": inputs, "outputs": outputs, "runtime": runtime,
                        "parameters": {"acceptance": acceptance, "analytics": "analyze defaults"},
                        "timings_seconds": timings, "report": report, "profile": profile}
            atomic_json(output_dir / "run_manifest.json", manifest)
            atomic_json(output_dir / "run_status.json", {**status, "status": "succeeded"})
            return report
        except Exception as exc:
            # Keep last successful manifest, but explicitly invalidate its freshness.
            atomic_json(output_dir / "run_status.json", {**status, "status": "failed", "stage": stage})
            raise RuntimeError(f"Этап {stage}: {exc}") from exc
        finally:
            handle.close()
            lock.unlink(missing_ok=True)
