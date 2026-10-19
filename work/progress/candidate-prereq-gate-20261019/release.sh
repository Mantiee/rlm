#!/usr/bin/env bash
# release.sh — candidate-prereq-gate-20261019
# Deploys synta-paper-candidate-lab.py fixing:
#   1. paper-v100-mission backend allowlist
#   2. reconcile_pre_request_state(state, now) + next_review_at=now+300
#   3. bind_history_row_contract()
#
# PRE-SHA:  2cedeef6f331335edd36cdd36b11491ec3597b42db40be4191a70a585cfc90a7
# POST-SHA: 680034af84963f68553fe5ec736d6b8acda398554a032a642e95e95783960810
#
# Run as: bash release.sh [--check | --rollback]
# Requires: ssh access to synta-v100 host (Codex deployer only)
# Never run from Windows; execute from Debian1 or via Codex deploy runner.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="/home/marek/ai-v100/bin/synta-paper-candidate-lab.py"
SOURCE="${SCRIPT_DIR}/synta-paper-candidate-lab.py"
BACKUP_DIR="/home/marek/ai-v100/backups/candidate-prereq-gate-20261019"
BACKUP_FILE="${BACKUP_DIR}/synta-paper-candidate-lab.py.pre"
SERVICE="synta-paper-candidate-lab.service"

PRE_SHA="2cedeef6f331335edd36cdd36b11491ec3597b42db40be4191a70a585cfc90a7"
POST_SHA="680034af84963f68553fe5ec736d6b8acda398554a032a642e95e95783960810"

sha256() { sha256sum "$1" | awk '{print $1}'; }

check_mode() {
    echo "=== CHECK MODE ==="
    local actual
    actual="$(sha256 "${TARGET}")"
    if [ "${actual}" = "${PRE_SHA}" ]; then
        echo "OK: target matches pre-image SHA"
    elif [ "${actual}" = "${POST_SHA}" ]; then
        echo "OK: target already at post-image SHA (already deployed)"
    else
        echo "ERROR: target SHA ${actual} matches neither pre nor post"
        exit 1
    fi
    echo "source SHA: $(sha256 "${SOURCE}")"
}

deploy() {
    echo "=== DEPLOY ==="

    # Verify source is what we expect
    local src_sha
    src_sha="$(sha256 "${SOURCE}")"
    if [ "${src_sha}" != "${POST_SHA}" ]; then
        echo "ERROR: source SHA ${src_sha} != expected ${POST_SHA}"
        exit 1
    fi

    # Verify current target is the pre-image
    local current_sha
    current_sha="$(sha256 "${TARGET}")"
    if [ "${current_sha}" != "${PRE_SHA}" ]; then
        if [ "${current_sha}" = "${POST_SHA}" ]; then
            echo "Already at post-image SHA; nothing to do."
            exit 0
        fi
        echo "ERROR: target SHA ${current_sha} != pre-image ${PRE_SHA}"
        echo "Refusing to deploy over unknown version."
        exit 1
    fi

    # Physical backup
    mkdir -p "${BACKUP_DIR}"
    cp -p "${TARGET}" "${BACKUP_FILE}"
    echo "Backup: ${BACKUP_FILE}"

    # Atomic replace
    local tmp="${TARGET}.tmp.$$"
    cp "${SOURCE}" "${tmp}"
    chmod --reference="${TARGET}" "${tmp}"
    mv "${tmp}" "${TARGET}"

    # Verify post-image
    local post_sha
    post_sha="$(sha256 "${TARGET}")"
    if [ "${post_sha}" != "${POST_SHA}" ]; then
        echo "ERROR: post-deploy SHA ${post_sha} != expected ${POST_SHA}"
        echo "Rolling back…"
        cp -p "${BACKUP_FILE}" "${TARGET}"
        exit 1
    fi
    echo "Deploy OK: ${TARGET} → ${POST_SHA}"

    # Restart service
    echo "Restarting ${SERVICE}…"
    systemctl restart "${SERVICE}"
    sleep 2
    systemctl is-active --quiet "${SERVICE}" && echo "Service active." || {
        echo "ERROR: service failed to start; rolling back"
        cp -p "${BACKUP_FILE}" "${TARGET}"
        systemctl restart "${SERVICE}" || true
        exit 1
    }
}

rollback() {
    echo "=== ROLLBACK ==="
    if [ ! -f "${BACKUP_FILE}" ]; then
        echo "ERROR: backup not found at ${BACKUP_FILE}"
        exit 1
    fi
    local bkp_sha
    bkp_sha="$(sha256 "${BACKUP_FILE}")"
    if [ "${bkp_sha}" != "${PRE_SHA}" ]; then
        echo "ERROR: backup SHA ${bkp_sha} != pre-image ${PRE_SHA}"
        exit 1
    fi
    cp -p "${BACKUP_FILE}" "${TARGET}"
    echo "Rollback OK: ${TARGET} → ${PRE_SHA}"
    systemctl restart "${SERVICE}" && echo "Service restarted." || true
}

case "${1:-deploy}" in
    --check)   check_mode ;;
    --rollback) rollback ;;
    deploy|"") deploy ;;
    *) echo "Usage: release.sh [--check | --rollback | deploy]"; exit 1 ;;
esac
