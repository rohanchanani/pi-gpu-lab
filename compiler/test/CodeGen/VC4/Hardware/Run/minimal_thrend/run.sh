#!/usr/bin/env bash
# Convenience dispatcher for the minimal_thrend hardware/codegen fixture.
#
#   ./run.sh reference   # run immutable reference side through the hardware runner
#   ./run.sh candidate   # run generated candidate side through the hardware runner
#   ./run.sh generate    # generate candidate artifacts under .vc4_auto
#   ./run.sh assemble    # assemble generated candidate qasm
#   ./run.sh build       # build generated candidate workdir without hardware execution

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$REPO_ROOT" ]]; then
  REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../../../.." && pwd)"
fi
SUPPORT_DIR="$REPO_ROOT/compiler/test/CodeGen/VC4/Support"
PHASE="${1:-reference}"
shift || true

usage() {
  cat >&2 <<'EOF'
usage: run.sh [reference|candidate|generate|assemble|build|run|workdir|clean|all]
EOF
}

case "$PHASE" in
  reference|candidate)
    exec bash "$SUPPORT_DIR/run_hardware_test.sh" "$SCRIPT_DIR" "$PHASE" "$@"
    ;;
  generate|assemble|build|run|workdir|clean|all)
    exec bash "$SUPPORT_DIR/run_candidate_codegen_test.sh" minimal_thrend "$PHASE" "$@"
    ;;
  *)
    usage
    exit 2
    ;;
esac
