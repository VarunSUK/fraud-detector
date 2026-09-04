#!/usr/bin/env bash
# Runs every unit/regression test suite in the repo -- the same four jobs
# CI runs in .github/workflows/ci.yml, plus the integration test, in one
# command for local use before opening a PR.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FAILED=()

run_suite() {
  local name=$1
  shift
  echo
  echo "==> $name"
  if ! "$@"; then
    FAILED+=("$name")
  fi
}

run_suite "Go tests" bash -c "cd '$ROOT_DIR/go-inference' && go build ./... && go vet ./... && go test ./..."
run_suite "Python tests" bash -c "cd '$ROOT_DIR/python-ml' && python3 -m pytest tests/ -q"
run_suite "Analytics SQL tests" bash -c "cd '$ROOT_DIR' && python3 -m pytest analytics/test_run_report.py -q"
run_suite "Frontend tests" bash -c "cd '$ROOT_DIR/frontend' && npm test"
run_suite "Integration test" "$ROOT_DIR/scripts/run-integration-tests.sh"

echo
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "All test suites passed."
  exit 0
else
  echo "Failed: ${FAILED[*]}"
  exit 1
fi
