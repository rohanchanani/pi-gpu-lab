#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$REPO_ROOT" ]]; then
  REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../../../.." && pwd)"
fi
SUPPORT_DIR="$REPO_ROOT/compiler/test/CodeGen/VC4/Support"
PHASE="${1:-reference}"
shift || true
if [[ "${VC4_RUN_QUARANTINED:-0}" != "1" ]]; then
  cat >&2 <<'EOF'
vpm_slice_visibility is quarantined.

It was a hardware hypothesis test for VPM slice/storage visibility, and its
observations are not a stable correctness oracle for the generated VC4 hardware
fixture corpus. Set VC4_RUN_QUARANTINED=1 to run it manually.
EOF
  exit 2
fi
case "$PHASE" in
  reference|candidate)
    exec bash "$SUPPORT_DIR/run_hardware_test.sh" "$SCRIPT_DIR" "$PHASE" "$SCRIPT_DIR/expected.json" "$@" ;;
  generate|assemble|build|run|workdir|clean|all)
    exec bash "$SUPPORT_DIR/run_candidate_codegen_test.sh" vpm_slice_visibility "$PHASE" "$@" ;;
  *)
    echo "usage: run.sh [reference|candidate|generate|assemble|build|run|workdir|clean|all]" >&2
    exit 2 ;;
esac
