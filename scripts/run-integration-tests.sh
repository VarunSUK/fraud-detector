#!/usr/bin/env bash
# Integration test: trains a real (tiny) model set, boots the ml-serving
# sidecar and the Go inference API as real processes, and exercises the
# actual HTTP contract between them -- the same manual sequence used to
# verify this repo throughout development, formalized into a script.
#
# This intentionally does NOT use docker-compose: running the two binaries
# directly is faster, needs no Docker daemon, and the whole point is
# testing the same code paths CI/local dev already run.
#
# Usage: ./scripts/run-integration-tests.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="$(mktemp -d)"
MODELS_DIR="$WORK_DIR/models"
AUDIT_DB="$WORK_DIR/audit_log.db"
SIDECAR_PORT=8100
GO_PORT=8101
SIDECAR_PID=""
GO_PID=""

kill_port() {
  # Kill-by-PID (via $!) can miss the real process if it ended up as a
  # grandchild of a subshell rather than the subshell itself; killing
  # whatever is actually bound to the port is the reliable fallback.
  local port=$1
  local pid
  pid=$(lsof -ti ":$port" 2>/dev/null || true)
  [ -n "$pid" ] && kill $pid 2>/dev/null || true
}

cleanup() {
  [ -n "$GO_PID" ] && kill "$GO_PID" 2>/dev/null || true
  [ -n "$SIDECAR_PID" ] && kill "$SIDECAR_PID" 2>/dev/null || true
  sleep 0.2
  kill_port "$GO_PORT"
  kill_port "$SIDECAR_PORT"
  rm -rf "$WORK_DIR"
}
trap cleanup EXIT

wait_for_health() {
  local url=$1 name=$2
  for _ in $(seq 1 30); do
    if curl -sf "$url" > /dev/null 2>&1; then
      echo "  $name is up"
      return 0
    fi
    sleep 1
  done
  echo "  $name did not become healthy in time" >&2
  return 1
}

assert_contains() {
  local haystack=$1 needle=$2 label=$3
  if [[ "$haystack" != *"$needle"* ]]; then
    echo "FAIL: $label -- expected to find '$needle' in: $haystack" >&2
    exit 1
  fi
  echo "  PASS: $label"
}

echo "==> Training a small ensemble into $MODELS_DIR"
python3 "$ROOT_DIR/scripts/seed_audit_log.py" \
  --models-dir "$MODELS_DIR" --db "$AUDIT_DB" \
  --num-transactions 400 --train > "$WORK_DIR/seed.log" 2>&1

echo "==> Starting ml-serving sidecar on :$SIDECAR_PORT"
# `exec` as the final step makes bash replace the subshell process with
# python3 itself, instead of leaving python3 running as its child --
# without this, $! is the subshell's PID, and killing it can leave the
# actual server orphaned (reparented to init) and still bound to the port.
(cd "$ROOT_DIR/python-ml" && exec env MODELS_DIR="$MODELS_DIR" AUDIT_DB_PATH="$AUDIT_DB" \
  python3 -m uvicorn serve:app --app-dir src --port "$SIDECAR_PORT" \
  > "$WORK_DIR/sidecar.log" 2>&1) &
SIDECAR_PID=$!
disown
wait_for_health "http://127.0.0.1:$SIDECAR_PORT/health" "ml-serving"

echo "==> Building go-inference"
(cd "$ROOT_DIR/go-inference" && go build -o "$WORK_DIR/go-inference-bin" ./cmd/server)

echo "==> Starting go-inference on :$GO_PORT"
# Run the built binary directly rather than `go run` -- `go run` spawns the
# actual server as a child of a wrapper process, and killing the wrapper
# does not reliably kill that child, leaving an orphaned server bound to
# GO_PORT that breaks every subsequent run.
PORT="$GO_PORT" ML_SERVICE_URL="http://127.0.0.1:$SIDECAR_PORT" LOG_LEVEL=warn \
  "$WORK_DIR/go-inference-bin" > "$WORK_DIR/go.log" 2>&1 &
GO_PID=$!
disown
wait_for_health "http://127.0.0.1:$GO_PORT/health" "go-inference"

BASE="http://127.0.0.1:$GO_PORT"

echo "==> GET /health"
health=$(curl -sf "$BASE/health")
assert_contains "$health" '"status":"healthy"' "health reports healthy"
assert_contains "$health" "ml_service" "ml_service is registered"

echo "==> POST /api/v1/score"
score=$(curl -sf -X POST "$BASE/api/v1/score" -H "Content-Type: application/json" -d \
  '{"transaction":{"time":90000,"amount":2500,"transaction_id":"it_score","v14":-3.2,"v4":2.1}}')
assert_contains "$score" '"model":"ensemble"' "score used the ensemble"

echo "==> POST /api/v1/explain"
explain=$(curl -sf -X POST "$BASE/api/v1/explain" -H "Content-Type: application/json" -d \
  '{"transaction":{"time":90000,"amount":2500,"transaction_id":"it_explain","v14":-3.2,"v4":2.1}}')
assert_contains "$explain" "feature_contributions" "explain returned SHAP contributions"
assert_contains "$explain" "model_scores" "explain returned per-model scores"

echo "==> POST /api/v1/decision"
decision=$(curl -sf -X POST "$BASE/api/v1/decision" -H "Content-Type: application/json" -d \
  '{"transaction":{"time":90000,"amount":2500,"transaction_id":"it_decision","v14":-3.2,"v4":2.1},"account":{"credit_limit":5000,"current_balance":1000}}')
assert_contains "$decision" "narrative" "decision returned a narrative"
assert_contains "$decision" "reason_codes" "decision returned reason codes"

echo "==> GET /api/v1/analytics/summary"
analytics=$(curl -sf "$BASE/api/v1/analytics/summary")
assert_contains "$analytics" "funnel" "analytics returned a funnel"
assert_contains "$analytics" "score_deciles" "analytics returned score deciles"

echo "==> GET /api/v1/cases"
cases=$(curl -sf "$BASE/api/v1/cases")
assert_contains "$cases" "cases" "cases endpoint responded"

echo "==> GET /metrics (real Prometheus format, not a JSON stub)"
metrics=$(curl -sf "$BASE/metrics")
assert_contains "$metrics" "# HELP fraud_detection_requests_total" "metrics are real Prometheus exposition format"

echo
echo "All integration checks passed."
