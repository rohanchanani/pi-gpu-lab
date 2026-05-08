#!/usr/bin/env bash
# Generate, assemble, build, or run a VC4 codegen candidate bundle for a
# hardware ground-truth test without mutating the checked-in reference bundle.
#
# Usage:
#   run_candidate_codegen_test.sh TEST_NAME generate
#   run_candidate_codegen_test.sh TEST_NAME assemble
#   run_candidate_codegen_test.sh TEST_NAME build
#   run_candidate_codegen_test.sh TEST_NAME run

set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
usage: run_candidate_codegen_test.sh TEST_NAME [generate|assemble|build|run|all|workdir|clean]
USAGE
}

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

TEST_ROOT="$REPO_ROOT/compiler/test/CodeGen/VC4/Hardware/Run/$TEST_NAME"
INPUT_MLIR="$TEST_ROOT/input.mlir"
EXPECTED_JSON="$TEST_ROOT/expected.json"
REFERENCE_DIR="$TEST_ROOT/reference"
AUTO_ROOT="$REPO_ROOT/.vc4_auto/codegen_m1"
GENERATED_DIR="$AUTO_ROOT/candidates/$TEST_NAME"
HARDWARE_ROOT="$AUTO_ROOT/hardware/$TEST_NAME"
WORK_DIR="$HARDWARE_ROOT/candidate_work"

log() {
  printf '[vc4-candidate] %s\n' "$*"
}

fail() {
  printf '[vc4-candidate] ERROR: %s\n' "$*" >&2
  exit 1
}

relpath() {
  local path="$1"
  case "$path" in
    "$REPO_ROOT"/*) printf '%s\n' "${path#$REPO_ROOT/}" ;;
    *) printf '%s\n' "$path" ;;
  esac
}

require_file() {
  [[ -f "$1" ]] || fail "required file not found: $(relpath "$1")"
}

require_dir() {
  [[ -d "$1" ]] || fail "required directory not found: $(relpath "$1")"
}

find_tool() {
  local tool="$1"
  if [[ -x "$REPO_ROOT/compiler/build/bin/$tool" ]]; then
    printf '%s\n' "$REPO_ROOT/compiler/build/bin/$tool"
    return 0
  fi
  if command -v "$tool" >/dev/null 2>&1; then
    command -v "$tool"
    return 0
  fi
  fail "could not find required tool: $tool"
}

check_fixture() {
  require_dir "$TEST_ROOT"
  require_file "$INPUT_MLIR"
  require_file "$EXPECTED_JSON"
  require_dir "$REFERENCE_DIR"
  require_file "$REFERENCE_DIR/mailbox.c"
  require_file "$REFERENCE_DIR/mailbox.h"
  require_dir "$TEST_ROOT/share"
}

copy_assembler_share() {
  # vc4asm -c resolves ../share/vc4tmpl/template.h from the generated bundle
  # directory.  Also provide ./share for older local assembler builds.
  if [[ -d "$TEST_ROOT/share" ]]; then
    mkdir -p "$AUTO_ROOT/candidates"
    rm -rf "$AUTO_ROOT/candidates/share" "$GENERATED_DIR/share"
    cp -R "$TEST_ROOT/share" "$AUTO_ROOT/candidates/share"
    cp -R "$TEST_ROOT/share" "$GENERATED_DIR/share"
  fi
}

check_generated_bundle() {
  require_file "$GENERATED_DIR/kernel.qasm"
  require_file "$GENERATED_DIR/kernel_launch.c"
  require_file "$GENERATED_DIR/kernel_launch.h"
  require_file "$GENERATED_DIR/manifest.json"
}

run_vc4_codegen() {
  local vc4_codegen
  vc4_codegen="$(find_tool vc4-codegen)"
  rm -rf "$GENERATED_DIR"
  mkdir -p "$GENERATED_DIR"
  log "generating $(relpath "$GENERATED_DIR") from $(relpath "$INPUT_MLIR")"
  "$vc4_codegen" "$INPUT_MLIR" --emit-bundle "$GENERATED_DIR"
  check_generated_bundle
  copy_assembler_share
}

ensure_generated() {
  check_fixture
  if [[ -f "$GENERATED_DIR/kernel.qasm" && \
        -f "$GENERATED_DIR/kernel_launch.c" && \
        -f "$GENERATED_DIR/kernel_launch.h" && \
        -f "$GENERATED_DIR/manifest.json" ]]; then
    log "using existing generated candidate artifacts: $(relpath "$GENERATED_DIR")"
    copy_assembler_share
    return 0
  fi
  run_vc4_codegen
}

assemble_candidate() {
  ensure_generated
  local vc4asm_tool
  vc4asm_tool="$(find_tool vc4asm)"
  log "assembling $(relpath "$GENERATED_DIR/kernel.qasm") as kernelshader.c/.h"
  local out
  if ! out="$(
    (
      cd "$GENERATED_DIR"
      rm -f kernelshader.c kernelshader.h
      "$vc4asm_tool" -c kernelshader.c -h kernelshader.h kernel.qasm
    ) 2>&1
  )"; then
    printf '%s\n' "$out" >&2
    fail "vc4asm failed for $TEST_NAME"
  fi
  if [[ -n "$out" ]]; then
    printf '%s\n' "$out" >&2
    fail "vc4asm produced unexpected output for $TEST_NAME"
  fi
  require_file "$GENERATED_DIR/kernelshader.c"
  require_file "$GENERATED_DIR/kernelshader.h"
}

extract_public_name() {
  require_file "$GENERATED_DIR/manifest.json"
  python3 - "$GENERATED_DIR/manifest.json" <<'PY_PUBLIC'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text())
public = data.get("public_name") or data.get("c_entry_point")
if not isinstance(public, str) or not public:
    launch = data.get("launch_abi")
    if isinstance(launch, dict):
        public = launch.get("public_name")
if not isinstance(public, str) or not public:
    raise SystemExit("manifest.json does not contain public_name or c_entry_point")
print(public)
PY_PUBLIC
}

write_candidate_harness() {
  local public_name="$1"
  local harness_name="$2"
  cat > "$WORK_DIR/$harness_name" <<EOF
#include "kernel_launch.h"

/*
 * Compile-time candidate harness for $TEST_NAME.
 *
 * This slice only proves that vc4-codegen output can be generated, assembled,
 * and built into a candidate-side workdir.  Real runtime binding and hardware
 * execution are deliberately left to the hardware-smoke slice.
 */
void notmain(void) {
  (void)&$public_name;
}
EOF
}

find_reference_harness_basename() {
  local -a matches
  mapfile -t matches < <(find "$REFERENCE_DIR" -maxdepth 1 -type f -name '3-test-*.c' -print | sort)
  if [[ "${#matches[@]}" -ne 1 ]]; then
    fail "expected exactly one reference harness matching reference/3-test-*.c"
  fi
  basename "${matches[0]}"
}

write_candidate_makefile() {
  local harness_name="$1"
  cat > "$WORK_DIR/Makefile" <<EOF
CC ?= arm-none-eabi-gcc
CFLAGS ?= -std=gnu99 -O2 -Wall -ffreestanding -I.
OBJDIR := objs
OBJS := \
  \$(OBJDIR)/${harness_name%.c}.o \
  \$(OBJDIR)/kernel_launch.o \
  \$(OBJDIR)/kernelshader.o

.PHONY: all clean run

all: \$(OBJS)

\$(OBJDIR):
	mkdir -p \$(OBJDIR)

\$(OBJDIR)/%.o: %.c | \$(OBJDIR)
	\$(CC) \$(CFLAGS) -c \$< -o \$@

run: all
	@echo "candidate hardware execution is intentionally not run by the m1-09 build gate" >&2
	@exit 1

clean:
	rm -rf \$(OBJDIR)
EOF
}

write_workdir_run_sh() {
  cat > "$WORK_DIR/run.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
make run
EOF
  chmod +x "$WORK_DIR/run.sh"
}

prepare_workdir() {
  assemble_candidate
  local public_name harness_name
  public_name="$(extract_public_name)"
  if ! [[ "$public_name" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
    fail "generated public_name is not a valid C identifier: $public_name"
  fi
  harness_name="$(find_reference_harness_basename)"

  log "preparing candidate workdir $(relpath "$WORK_DIR")"
  rm -rf "$WORK_DIR"
  mkdir -p "$WORK_DIR"

  cp "$REFERENCE_DIR/mailbox.c" "$WORK_DIR/mailbox.c"
  cp "$REFERENCE_DIR/mailbox.h" "$WORK_DIR/mailbox.h"
  cp "$GENERATED_DIR/kernel.qasm" "$WORK_DIR/kernel.qasm"
  cp "$GENERATED_DIR/kernel_launch.c" "$WORK_DIR/kernel_launch.c"
  cp "$GENERATED_DIR/kernel_launch.h" "$WORK_DIR/kernel_launch.h"
  cp "$GENERATED_DIR/kernelshader.c" "$WORK_DIR/kernelshader.c"
  cp "$GENERATED_DIR/kernelshader.h" "$WORK_DIR/kernelshader.h"

  write_candidate_harness "$public_name" "$harness_name"
  write_candidate_makefile "$harness_name"
  write_workdir_run_sh

  cat > "$WORK_DIR/README.generated.md" <<EOF
# Generated candidate workdir

This directory is generated by compiler/test/CodeGen/VC4/Support/run_candidate_codegen_test.sh.
It is intentionally under .vc4_auto and should not be committed.

Test: $TEST_NAME
Public launcher: $public_name
Harness: $harness_name
EOF
  log "candidate workdir ready: $(relpath "$WORK_DIR")"
}

build_candidate() {
  prepare_workdir
  log "building candidate in $(relpath "$WORK_DIR")"
  (
    cd "$WORK_DIR"
    make
  )
}

run_candidate() {
  prepare_workdir
  log "running candidate hardware hook from $(relpath "$WORK_DIR")"
  (
    cd "$WORK_DIR"
    bash run.sh
  )
}

case "$PHASE" in
  clean)
    rm -rf "$GENERATED_DIR" "$HARDWARE_ROOT"
    log "removed generated candidate state for $TEST_NAME"
    ;;
  generate)
    check_fixture
    run_vc4_codegen
    ;;
  assemble)
    assemble_candidate
    ;;
  build)
    build_candidate
    ;;
  run)
    run_candidate
    ;;
  all)
    run_candidate
    ;;
  workdir)
    prepare_workdir
    printf '%s\n' "$WORK_DIR"
    ;;
  *)
    usage
    exit 2
    ;;
esac
