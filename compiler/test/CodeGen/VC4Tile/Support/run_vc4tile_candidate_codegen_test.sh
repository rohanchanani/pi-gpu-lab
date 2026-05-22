#!/usr/bin/env bash
# Generate VC4Tile candidate artifacts without mutating checked-in expected artifacts.
# The preserved candidate trace is:
#   input.vc4tile.mlir -> lowered.ssavc4.mlir -> scheduled.vc4.mlir -> bundle/
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
usage:
  run_vc4tile_candidate_codegen_test.sh --self-test
  run_vc4tile_candidate_codegen_test.sh TEST_NAME [generate|all|workdir|clean]
USAGE
}

if [[ $# -eq 1 && "$1" == "--self-test" ]]; then
  printf '[vc4tile-candidate] self-test ok\n'
  exit 0
fi

if [[ $# -lt 2 ]]; then
  usage
  exit 2
fi

TEST_NAME="$1"
PHASE="$2"
shift 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$REPO_ROOT" ]]; then
  REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
fi

TEST_ROOT="$REPO_ROOT/compiler/test/CodeGen/VC4Tile/Hardware/Run/$TEST_NAME"
INPUT_MLIR="$TEST_ROOT/input.vc4tile.mlir"
if [[ ! -f "$INPUT_MLIR" && -f "$TEST_ROOT/input.mlir" ]]; then
  INPUT_MLIR="$TEST_ROOT/input.mlir"
fi

AUTO_ROOT_RAW="${VC4_CODEGEN_STATE_ROOT:-.vc4_auto/vc4tile_m4}"
case "$AUTO_ROOT_RAW" in
  /*) AUTO_ROOT="$AUTO_ROOT_RAW" ;;
  *) AUTO_ROOT="$REPO_ROOT/$AUTO_ROOT_RAW" ;;
esac

CANDIDATE_ROOT="$AUTO_ROOT/candidates/$TEST_NAME"
INPUT_COPY="$CANDIDATE_ROOT/input.vc4tile.mlir"
LOWERED_SSAVC4="$CANDIDATE_ROOT/lowered.ssavc4.mlir"
SCHEDULED_VC4="$CANDIDATE_ROOT/scheduled.vc4.mlir"
BUNDLE_DIR="$CANDIDATE_ROOT/bundle"

log() { printf '[vc4tile-candidate] %s\n' "$*"; }
fail() { printf '[vc4tile-candidate] ERROR: %s\n' "$*" >&2; exit 1; }
relpath() { case "$1" in "$REPO_ROOT"/*) printf '%s\n' "${1#$REPO_ROOT/}" ;; *) printf '%s\n' "$1" ;; esac; }
require_file() { [[ -f "$1" ]] || fail "required file not found: $(relpath "$1")"; }

find_tool() {
  local tool="$1"
  if [[ -x "$REPO_ROOT/compiler/build/bin/$tool" ]]; then printf '%s\n' "$REPO_ROOT/compiler/build/bin/$tool"; return 0; fi
  if command -v "$tool" >/dev/null 2>&1; then command -v "$tool"; return 0; fi
  fail "could not find required tool: $tool"
}

check_fixture() {
  require_file "$INPUT_MLIR"
}

check_generated_bundle() {
  require_file "$BUNDLE_DIR/manifest.json"
  require_file "$BUNDLE_DIR/kernel_launch.c"
  require_file "$BUNDLE_DIR/kernel_launch.h"
}

generate_candidate() {
  check_fixture
  local vc4_opt vc4_codegen
  vc4_opt="$(find_tool vc4-opt)"
  vc4_codegen="$(find_tool vc4-codegen)"
  rm -rf "$CANDIDATE_ROOT"
  mkdir -p "$CANDIDATE_ROOT" "$BUNDLE_DIR"
  cp "$INPUT_MLIR" "$INPUT_COPY"
  log "lowering $(relpath "$INPUT_COPY") to $(relpath "$LOWERED_SSAVC4")"
  "$vc4_opt" "$INPUT_COPY" --convert-vc4tile-to-ssavc4 -o "$LOWERED_SSAVC4"
  log "lowering $(relpath "$LOWERED_SSAVC4") to $(relpath "$SCHEDULED_VC4")"
  "$vc4_opt" "$LOWERED_SSAVC4" --convert-ssavc4-to-vc4 -o "$SCHEDULED_VC4"
  log "generating bundle artifacts in $(relpath "$BUNDLE_DIR")"
  "$vc4_codegen" "$SCHEDULED_VC4" --emit-bundle "$BUNDLE_DIR"
  check_generated_bundle
}

case "$PHASE" in
  generate|all)
    generate_candidate
    ;;
  workdir)
    printf '%s\n' "$CANDIDATE_ROOT"
    ;;
  clean)
    rm -rf "$CANDIDATE_ROOT"
    ;;
  *)
    usage
    exit 2
    ;;
esac
