from __future__ import annotations

import argparse
import sys
import json

from .pipeline import run_pipeline
from .provenance import verify_run


def main() -> int:
    parser = argparse.ArgumentParser(description="ApexFlow: Parquet → проверенные CSV")
    parser.add_argument("--data", default="data", help="каталог с тремя Parquet")
    parser.add_argument("--output", default="output", help="каталог для CSV")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--acceptance", action="store_true", help="проверить полный набор ТЗ и время <300 с")
    mode.add_argument("--verify-output", action="store_true", help="проверить хэши успешного запуска без пересчёта")
    parser.add_argument("--json", action="store_true", help="машиночитаемый итоговый отчёт")
    args = parser.parse_args()
    try:
        if args.verify_output:
            manifest = verify_run(args.data, args.output)
            print(json.dumps(manifest["report"], ensure_ascii=False) if args.json else "Хэши входов и результатов подтверждены")
            return 0
        report = run_pipeline(args.data, args.output, acceptance=args.acceptance)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Ошибка pipeline: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False) if args.json else f"Готово: nodes={report['nodes']}, clusters={report['clusters']}, top_nodes={report['top_nodes']}; {report['seconds']:.2f} c; {report['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
