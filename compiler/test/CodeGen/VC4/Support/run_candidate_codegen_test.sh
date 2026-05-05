#!/usr/bin/env bash
# Generate, assemble, build, or run a VC4 codegen candidate bundle for a
# hardware ground-truth test without mutating the checked-in reference bundle.
#
# Usage:
#   bash compiler/test/CodeGen/VC4/Support/run_candidate_codegen_test.sh TEST_NAME generate
#   bash compiler/test/CodeGen/VC4/Support/run_candidate_codegen_test.sh TEST_NAME assemble
#   bash compiler/test/CodeGen/VC4/Support/run_candidate_codegen_test.sh TEST_NAME build
#   bash compiler/test/CodeGen/VC4/Support/run_candidate_codegen_test.sh TEST_NAME run
#
# Candidate source artifacts live under:
#   .vc4_auto/codegen_m1/candidates/TEST_NAME/
#
# Candidate build/run work directories live under:
#   .vc4_auto/codegen_m1/hardware/TEST_NAME/candidate_work/
#
# Reference bundles under compiler/test/CodeGen/VC4/Hardware/Run/*/reference
# are read-only inputs to this script.

set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: run_candidate_codegen_test.sh TEST_NAME PHASE

PHASE is one of: generate, assemble, build, run, clean, workdir, all
EOF
}

if [[ $# -lt 2 ]]; then
  usage
  exit 2
fi

TEST_NAME="$1"
PHASE="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$REPO_ROOT" ]]; then
  echo "[vc4-candidate] ERROR: could not locate git repo root from $SCRIPT_DIR" >&2
  exit 1
fi

TEST_ROOT="$REPO_ROOT/compiler/test/CodeGen/VC4/Hardware/Run/$TEST_NAME"
REFERENCE_DIR="$TEST_ROOT/reference"
INPUT_MLIR="$TEST_ROOT/input.mlir"
EXPECTED_JSON="$TEST_ROOT/expected.json"
AUTO_ROOT="$REPO_ROOT/.vc4_auto/codegen_m1"
GENERATED_DIR="$AUTO_ROOT/candidates/$TEST_NAME"
HARDWARE_ROOT="$AUTO_ROOT/hardware/$TEST_NAME"
WORK_DIR="$HARDWARE_ROOT/candidate_work"
VC4_CODEGEN="$REPO_ROOT/compiler/build/bin/vc4-codegen"

log() {
  printf '[vc4-candidate] %s\n' "$*"
}

fail() {
  printf '[vc4-candidate] ERROR: %s\n' "$*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || fail "missing required file: $1"
}

require_dir() {
  [[ -d "$1" ]] || fail "missing required directory: $1"
}

find_reference_harness() {
  local -a matches
  mapfile -t matches < <(find "$REFERENCE_DIR" -maxdepth 1 -type f -name '3-test-*.c' | sort)
  if [[ ${#matches[@]} -ne 1 ]]; then
    printf '[vc4-candidate] ERROR: expected exactly one reference harness matching reference/3-test-*.c; found %d\n' "${#matches[@]}" >&2
    printf '%s\n' "${matches[@]}" >&2
    exit 1
  fi
  printf '%s\n' "${matches[0]}"
}

extract_public_name() {
  python3 - "$GENERATED_DIR/manifest.json" "$INPUT_MLIR" "$TEST_NAME" <<'PY'
import json
import re
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
input_mlir = Path(sys.argv[2])
test_name = sys.argv[3]

def walk(obj):
    if isinstance(obj, dict):
        for key in ("public_name", "launcher", "launch", "name"):
            value = obj.get(key)
            if key == "public_name" and isinstance(value, str) and value:
                return value
        for value in obj.values():
            found = walk(value)
            if found:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = walk(value)
            if found:
                return found
    return None

if manifest.exists():
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        found = walk(data)
        if found:
            print(found)
            raise SystemExit(0)
    except Exception:
        pass

if input_mlir.exists():
    text = input_mlir.read_text(encoding="utf-8", errors="replace")
    m = re.search(r'public_name\s*=\s*"([^"]+)"', text)
    if m:
        print(m.group(1))
        raise SystemExit(0)

print(f"{test_name}_launch")
PY
}

extract_shader_base() {
  python3 - "$GENERATED_DIR/kernel_launch.c" "$TEST_NAME" <<'PY'
import re
import sys
from pathlib import Path

launch_c = Path(sys.argv[1])
test_name = sys.argv[2]
if launch_c.exists():
    text = launch_c.read_text(encoding="utf-8", errors="replace")
    # Prefer the exact shader header included by the generated launcher.
    for m in re.finditer(r'#\s*include\s+"([^"]*shader\.h)"', text):
        name = Path(m.group(1)).name
        if name.endswith(".h"):
            print(name[:-2])
            raise SystemExit(0)
# Deterministic fallback for Milestone 1 skeletons.
print("kernelshader")
PY
}

check_fixture() {
  require_dir "$TEST_ROOT"
  require_file "$INPUT_MLIR"
  require_file "$EXPECTED_JSON"
  require_dir "$REFERENCE_DIR"
  require_file "$REFERENCE_DIR/mailbox.c"
  require_file "$REFERENCE_DIR/mailbox.h"
}

ensure_generated() {
  check_fixture
  if [[ -f "$GENERATED_DIR/kernel.qasm" && -f "$GENERATED_DIR/kernel_launch.c" && -f "$GENERATED_DIR/kernel_launch.h" && -f "$GENERATED_DIR/manifest.json" ]]; then
    log "using existing generated candidate artifacts: ${GENERATED_DIR#$REPO_ROOT/}"
    return 0
  fi
  require_file "$VC4_CODEGEN"
  mkdir -p "$GENERATED_DIR"
  log "generating candidate artifacts into ${GENERATED_DIR#$REPO_ROOT/}"
  "$VC4_CODEGEN" "$INPUT_MLIR" --emit-bundle "$GENERATED_DIR"
  require_file "$GENERATED_DIR/kernel.qasm"
  require_file "$GENERATED_DIR/kernel_launch.c"
  require_file "$GENERATED_DIR/kernel_launch.h"
  require_file "$GENERATED_DIR/manifest.json"
}

assemble_candidate() {
  ensure_generated
  command -v vc4asm >/dev/null 2>&1 || fail "vc4asm not found in PATH"
  local shader_base
  shader_base="$(extract_shader_base)"
  log "assembling ${GENERATED_DIR#$REPO_ROOT/}/kernel.qasm as ${shader_base}.c/.h"
  rm -f "$GENERATED_DIR/$shader_base.c" "$GENERATED_DIR/$shader_base.h"
  local out
  if ! out="$(cd "$GENERATED_DIR" && vc4asm -c "$shader_base.c" -h "$shader_base.h" kernel.qasm 2>&1)"; then
    printf '%s\n' "$out"
    fail "vc4asm failed for $TEST_NAME"
  fi
  if [[ -n "$out" ]]; then
    printf '%s\n' "$out"
  fi
  require_file "$GENERATED_DIR/$shader_base.c"
  require_file "$GENERATED_DIR/$shader_base.h"
}

write_candidate_makefile() {
  local harness_name="$1"
  local shader_base="$2"
  cat > "$WORK_DIR/Makefile" <<EOF
LIBS += \$(CS240LX_2025_PATH)/lib/libgcc.a \$(CS240LX_2025_PATH)/libpi/libpi.a

export OPT_LEVEL := -O3

COMMON_SRC := mailbox.c kernel_launch.c ${shader_base}.c

PROGS := ${harness_name}

STAFF_OBJS += \$(CS240LX_2025_PATH)/libpi/staff-objs/staff-hw-spi.o
STAFF_OBJS += \$(CS240LX_2025_PATH)/libpi/staff-objs/kmalloc.o

RUN ?= 0

BOOTLOADER = pi-install
EXCLUDE ?= grep -v simple_boot
GREP_STR := 'HASH:\|ERROR:\|PANIC:\|SUCCESS:\|VC4_TEST_RESULT\|NRF:'
include \$(CS240LX_2025_PATH)/libpi/mk/Makefile.robust
EOF
}

prepare_workdir() {
  assemble_candidate
  local public_name shader_base harness harness_name
  public_name="$(extract_public_name)"
  shader_base="$(extract_shader_base)"
  harness="$(find_reference_harness)"
  harness_name="$(basename "$harness")"

  log "preparing candidate workdir ${WORK_DIR#$REPO_ROOT/}"
  rm -rf "$WORK_DIR"
  mkdir -p "$WORK_DIR"

  cp "$REFERENCE_DIR/mailbox.c" "$WORK_DIR/mailbox.c"
  cp "$REFERENCE_DIR/mailbox.h" "$WORK_DIR/mailbox.h"
  cp "$harness" "$WORK_DIR/$harness_name"

  cp "$GENERATED_DIR/kernel_launch.c" "$WORK_DIR/kernel_launch.c"
  cp "$GENERATED_DIR/kernel_launch.h" "$WORK_DIR/kernel_launch.h"
  # Reference harnesses include the semantic public launch header name.
  cp "$GENERATED_DIR/kernel_launch.h" "$WORK_DIR/${public_name}.h"

  cp "$GENERATED_DIR/$shader_base.c" "$WORK_DIR/$shader_base.c"
  cp "$GENERATED_DIR/$shader_base.h" "$WORK_DIR/$shader_base.h"

  # Also provide the common reference-style shader/header names as aliases for
  # generated launchers or harnesses that choose test-name-based includes.
  if [[ "$shader_base" != "${TEST_NAME}shader" ]]; then
    cp "$GENERATED_DIR/$shader_base.c" "$WORK_DIR/${TEST_NAME}shader.c"
    cp "$GENERATED_DIR/$shader_base.h" "$WORK_DIR/${TEST_NAME}shader.h"
  fi

  write_candidate_makefile "$harness_name" "$shader_base"

  cat > "$WORK_DIR/run.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
make RUN=1
EOF
  chmod +x "$WORK_DIR/run.sh"

  cat > "$WORK_DIR/README.generated.md" <<EOF
# Generated candidate workdir

This directory is generated by compiler/test/CodeGen/VC4/Support/run_candidate_codegen_test.sh.
It is intentionally under .vc4_auto and should not be committed.

Test: ${TEST_NAME}
Public launcher: ${public_name}
Shader base: ${shader_base}
Harness: ${harness_name}
EOF
}

build_candidate() {
  prepare_workdir
  log "building candidate in ${WORK_DIR#$REPO_ROOT/}"
  (cd "$WORK_DIR" && make RUN=0)
}

run_candidate() {
  prepare_workdir
  log "running candidate on hardware in ${WORK_DIR#$REPO_ROOT/}"
  (cd "$WORK_DIR" && bash run.sh)
}

clean_candidate() {
  log "removing generated candidate hardware workdir ${HARDWARE_ROOT#$REPO_ROOT/}"
  rm -rf "$HARDWARE_ROOT"
}

case "$PHASE" in
  generate)
    rm -rf "$GENERATED_DIR"
    ensure_generated
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
  clean)
    clean_candidate
    ;;
  *)
    usage
    exit 2
    ;;
esac
