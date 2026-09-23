from __future__ import annotations

import argparse
import sys

from .pipeline import run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="ApexFlow: Parquet → проверенные CSV")
    parser.add_argument("--data", default="data", help="каталог с тремя Parquet")
    parser.add_argument("--output", default="output", help="каталог для CSV")
    args = parser.parse_args()
    try:
        report = run_pipeline(args.data, args.output)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Ошибка pipeline: {exc}", file=sys.stderr)
        return 1
    print(f"Готово: nodes={report['nodes']}, clusters={report['clusters']}, top_nodes={report['top_nodes']}; {report['seconds']:.2f} c; {report['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
