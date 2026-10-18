#!/usr/bin/env bash
# Synta income-self-learn release script
# Deploy to Debian1 (marek@192.168.0.68)
# Usage:
#   ./release.sh deploy   — deploy to Debian1
#   ./release.sh rollback — restore backup
#   ./release.sh test     — run tests locally

set -euo pipefail

HOST="marek@192.168.0.68"
REMOTE_DIR="/home/marek/ai-v100/income-self-learn"
BACKUP_DIR="/home/marek/ai-v100/research/deploy/rollback/income-self-learn-$(date -u +%Y%m%dT%H%M%SZ)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SHA_EXPECTED_GIG="$(sha256sum "${SCRIPT_DIR}/income/gig_worker.py" | awk '{print $1}')"
SHA_EXPECTED_NEWS="$(sha256sum "${SCRIPT_DIR}/learning/news_collector.py" | awk '{print $1}')"
SHA_EXPECTED_MODEL="$(sha256sum "${SCRIPT_DIR}/learning/model_updater.py" | awk '{print $1}')"
SHA_EXPECTED_BOUNTY="$(sha256sum "${SCRIPT_DIR}/income/bounty_evaluator.py" | awk '{print $1}')"

function run_tests() {
  echo "[TEST] Running smoke tests..."
  cd "${SCRIPT_DIR}"
  python3 -m pytest tests/ -v --tb=short
  echo "[TEST] All tests passed."
}

function deploy() {
  echo "[DEPLOY] Target: ${HOST}:${REMOTE_DIR}"
  echo "[DEPLOY] Backing up existing to ${BACKUP_DIR}"

  ssh "${HOST}" "
    set -e
    mkdir -p '${BACKUP_DIR}'
    [ -d '${REMOTE_DIR}' ] && cp -r '${REMOTE_DIR}/.' '${BACKUP_DIR}/' || true
    mkdir -p '${REMOTE_DIR}/income'
    mkdir -p '${REMOTE_DIR}/learning'
    mkdir -p '${REMOTE_DIR}/dashboard'
    mkdir -p '${REMOTE_DIR}/data'
  "

  # Upload files
  scp "${SCRIPT_DIR}/income/gig_worker.py"           "${HOST}:${REMOTE_DIR}/income/"
  scp "${SCRIPT_DIR}/income/earnings_history_api.py" "${HOST}:${REMOTE_DIR}/income/"
  scp "${SCRIPT_DIR}/income/bounty_evaluator.py"     "${HOST}:${REMOTE_DIR}/income/"
  scp "${SCRIPT_DIR}/learning/news_collector.py"     "${HOST}:${REMOTE_DIR}/learning/"
  scp "${SCRIPT_DIR}/learning/social_collector.py"   "${HOST}:${REMOTE_DIR}/learning/"
  scp "${SCRIPT_DIR}/learning/quarterly_parser.py"   "${HOST}:${REMOTE_DIR}/learning/"
  scp "${SCRIPT_DIR}/learning/historical_fetcher.py" "${HOST}:${REMOTE_DIR}/learning/"
  scp "${SCRIPT_DIR}/learning/model_updater.py"      "${HOST}:${REMOTE_DIR}/learning/"
  scp "${SCRIPT_DIR}/dashboard/earnings_chart.js"    "${HOST}:${REMOTE_DIR}/dashboard/"
  # Deploy bounty-evaluator systemd service
  scp "${SCRIPT_DIR}/systemd/synta-bounty-evaluator.service" "${HOST}:/home/marek/.config/systemd/user/synta-bounty-evaluator.service"
  ssh "${HOST}" "systemctl --user daemon-reload; systemctl --user enable synta-bounty-evaluator; systemctl --user restart synta-bounty-evaluator || true"

  # Verify SHA on remote
  echo "[VERIFY] Checking SHA on remote..."
  ssh "${HOST}" "sha256sum '${REMOTE_DIR}/income/gig_worker.py'" | grep -q "${SHA_EXPECTED_GIG}" || {
    echo "[ERROR] SHA mismatch for gig_worker.py — rolling back"; rollback; exit 1; }
  ssh "${HOST}" "sha256sum '${REMOTE_DIR}/learning/news_collector.py'" | grep -q "${SHA_EXPECTED_NEWS}" || {
    echo "[ERROR] SHA mismatch for news_collector.py — rolling back"; rollback; exit 1; }
  ssh "${HOST}" "sha256sum '${REMOTE_DIR}/learning/model_updater.py'" | grep -q "${SHA_EXPECTED_MODEL}" || {
    echo "[ERROR] SHA mismatch for model_updater.py — rolling back"; rollback; exit 1; }

  ssh "${HOST}" "sha256sum '${REMOTE_DIR}/income/bounty_evaluator.py'" | grep -q "${SHA_EXPECTED_BOUNTY}" || {
    echo "[ERROR] SHA mismatch for bounty_evaluator.py — rolling back"; rollback; exit 1; }

  echo "[DEPLOY] SHA verified OK."

  # Write __init__.py files
  ssh "${HOST}" "
    touch '${REMOTE_DIR}/income/__init__.py'
    touch '${REMOTE_DIR}/learning/__init__.py'
    touch '${REMOTE_DIR}/__init__.py'
  "

  # Install python deps (yfinance optional)
  ssh "${HOST}" "
    pip3 install yfinance 2>/dev/null || true
    python3 -c 'import yfinance; print(\"yfinance ok\")' 2>/dev/null || echo 'yfinance not available (optional)'
  "

  echo "[DEPLOY] Done. Files at ${REMOTE_DIR}"
  echo "[DEPLOY] Backup at ${BACKUP_DIR}"

  # Quick smoke test on remote
  ssh "${HOST}" "
    cd '${REMOTE_DIR}'
    python3 -c '
import sys; sys.path.insert(0, \".\")
from income.gig_worker import GigTask, _is_executable
t = GigTask(\"x\",\"test\",\"python script\",\"automation\",\"freelance\",\"http://x\",10.0,\"2026-10-18\")
assert _is_executable(t)
from learning.model_updater import update_signals
sigs = update_signals([\"BTC\"])
assert len(sigs) == 1
print(\"Remote smoke test PASS\")
'
  "
}

function rollback() {
  echo "[ROLLBACK] Finding latest backup..."
  LATEST_BACKUP=$(ssh "${HOST}" "ls -1d '$(dirname ${BACKUP_DIR})'/*income-self-learn* 2>/dev/null | sort | tail -1")
  if [ -z "${LATEST_BACKUP}" ]; then
    echo "[ROLLBACK] No backup found."
    exit 1
  fi
  echo "[ROLLBACK] Restoring from ${LATEST_BACKUP}"
  ssh "${HOST}" "
    rm -rf '${REMOTE_DIR}'
    cp -r '${LATEST_BACKUP}/.' '${REMOTE_DIR}/'
  "
  echo "[ROLLBACK] Done."
}

CMD="${1:-help}"
case "${CMD}" in
  deploy)   run_tests && deploy ;;
  rollback) rollback ;;
  test)     run_tests ;;
  *)
    echo "Usage: $0 [deploy|rollback|test]"
    exit 1
    ;;
esac
