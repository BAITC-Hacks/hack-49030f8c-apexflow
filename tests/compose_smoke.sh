#!/usr/bin/env bash
# Synthetic-only integration check. Run from the repository root with Bash/Docker.
set -euo pipefail
mkdir -p .ci
project_root="$PWD"
test_uid="$(id -u)"
test_gid="$(id -g)"
case "${OSTYPE:-}" in
  msys*|cygwin*)
    # Docker Desktop needs Windows bind paths; don't let MSYS rewrite /app paths.
    export MSYS_NO_PATHCONV=1
    project_root="$(pwd -W)"
    test_uid=10001
    test_gid=10001
    ;;
esac
run_dir="$(mktemp -d "$project_root/.ci/smoke.XXXXXX")"
export COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-apexflow-smoke}"
export APEXFLOW_IMAGE_TAG="$COMPOSE_PROJECT_NAME"
export APEXFLOW_DATA_DIR="$run_dir/data"
export APEXFLOW_OUTPUT_DIR="$run_dir/output"
export APEXFLOW_PORT="${APEXFLOW_PORT:-18501}"
export APEXFLOW_UID="$test_uid" APEXFLOW_GID="$test_gid"
mkdir -p "$APEXFLOW_DATA_DIR" "$APEXFLOW_OUTPUT_DIR" "$run_dir/reports"
cleanup() {
  docker compose logs --no-color > "$run_dir/reports/compose.log" 2>&1 || true
  docker compose ps -a > "$run_dir/reports/containers.txt" 2>&1 || true
  docker compose down --remove-orphans || true
}
trap cleanup EXIT
docker compose config --quiet
docker compose --profile test build pipeline tests
docker run --rm --network none "apexflow:$APEXFLOW_IMAGE_TAG" python -c \
  'from pathlib import Path; p=Path("/app"); assert not list(p.rglob("*.parquet")); assert not (p/"output").exists(); assert not (p/".git").exists(); assert not list(p.rglob(".env"))'
docker compose --profile test run --rm --no-deps -v "$run_dir/reports:/reports" tests \
  pytest -q -o cache_dir=/tmp/pytest-cache --junitxml=/reports/pytest.xml
docker compose --profile test run --rm --no-deps -v "$run_dir:/fixture" tests \
  python tests/make_fixture.py /fixture/data
docker compose up -d --no-build --force-recreate ui
ui_id="$(docker compose ps -q ui)"
for attempt in $(seq 1 30); do
  if [ "$(docker inspect -f '{{.State.Health.Status}}' "$ui_id")" = healthy ]; then break; fi
  sleep 2
done
test "$(docker inspect -f '{{.State.Health.Status}}' "$ui_id")" = healthy
curl --fail --silent --show-error "http://127.0.0.1:$APEXFLOW_PORT/_stcore/health"
docker compose run --rm --no-deps ui python tests/check_smoke.py
docker compose run --rm --no-deps ui python tests/check_mounts.py --ui
docker compose run --rm --no-deps pipeline python tests/check_mounts.py
cp "$APEXFLOW_OUTPUT_DIR/run_manifest.json" "$run_dir/first-manifest.json"
docker compose down
test -f "$APEXFLOW_OUTPUT_DIR/top_nodes.csv"
docker compose run --rm --no-deps pipeline
docker compose --profile test run --rm --no-deps -v "$run_dir:/fixture:ro" tests python -c \
  'import json; from pathlib import Path; a=json.loads(Path("/fixture/first-manifest.json").read_text()); b=json.loads(Path("/fixture/output/run_manifest.json").read_text()); assert a["run_id"] != b["run_id"]; assert a["inputs"] == b["inputs"]; assert a["outputs"] == b["outputs"]'
docker compose run --rm --no-deps pipeline python -m apexflow --verify-output
# Missing inputs must fail the dependency and prevent a fresh UI starting.
mv "$APEXFLOW_DATA_DIR/nodes.parquet" "$run_dir/nodes.saved.parquet"
if docker compose up -d --no-build --force-recreate ui; then
  echo 'ERROR: invalid input unexpectedly succeeded' >&2
  exit 1
fi
test -z "$(docker compose ps --status running -q ui)"
mv "$run_dir/nodes.saved.parquet" "$APEXFLOW_DATA_DIR/nodes.parquet"
if docker compose run --rm --no-deps pipeline python -m apexflow --verify-output; then
  echo 'ERROR: failed run was reported as fresh' >&2
  exit 1
fi
docker compose run --rm --no-deps pipeline
docker compose run --rm --no-deps ui python tests/check_smoke.py
echo 'PASS: real analytics, CSV/UI boundary, health, repeatability, persistence, failed-input gate'
