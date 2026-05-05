#!/usr/bin/env bash
# Dispatcher for the minimal_thrend hardware smoke fixture.
#
# Usage:
#   bash run.sh reference   # run checked-in reference bundle
#   bash run.sh candidate   # run generated candidate bundle through support
#   bash run.sh both        # run reference then candidate

set -euo pipefail

MODE="${1:-reference}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$REPO_ROOT" ]]; then
  echo "[minimal_thrend] ERROR: could not locate git repo root" >&2
  exit 1
fi

case "$MODE" in
  reference|ref)
    cd "$SCRIPT_DIR/reference"
    bash run.sh
    ;;
  candidate|cand)
    bash "$REPO_ROOT/compiler/test/CodeGen/VC4/Support/run_candidate_codegen_test.sh" minimal_thrend run
    ;;
  both)
    cd "$SCRIPT_DIR/reference"
    bash run.sh
    bash "$REPO_ROOT/compiler/test/CodeGen/VC4/Support/run_candidate_codegen_test.sh" minimal_thrend run
    ;;
  *)
    echo "usage: bash run.sh [reference|candidate|both]" >&2
    exit 2
    ;;
esac
