#!/usr/bin/env bash
# Generate, assemble, build, or run a VC4 codegen candidate program bundle for
# a hardware ground-truth test without mutating the checked-in reference side.
#
# M2 additions:
#   * VC4_CODEGEN_STATE_ROOT selects .vc4_auto/codegen_m2 cleanly.
#   * manifest-v2 kernels[] are treated as the general case, including the
#     single-kernel case.
#   * every manifest-listed qasm_path is assembled to its code_symbol .c/.h.
#   * candidate/<test>_candidate_harness.c is preferred when present, so M2
#     CUDA-like device-pointer harnesses can coexist with immutable references.

set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
usage: run_candidate_codegen_test.sh TEST_NAME [generate|assemble|build|run|all|workdir|clean]
USAGE
}

if [[ $# -lt 2 ]]; then usage; exit 2; fi

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
CANDIDATE_DIR="$TEST_ROOT/candidate"
AUTO_ROOT_RAW="${VC4_CODEGEN_STATE_ROOT:-.vc4_auto/codegen_m1}"
case "$AUTO_ROOT_RAW" in
  /*) AUTO_ROOT="$AUTO_ROOT_RAW" ;;
  *) AUTO_ROOT="$REPO_ROOT/$AUTO_ROOT_RAW" ;;
esac
GENERATED_DIR="$AUTO_ROOT/candidates/$TEST_NAME"
HARDWARE_ROOT="$AUTO_ROOT/hardware/$TEST_NAME"
WORK_DIR="$HARDWARE_ROOT/candidate_work"

log() { printf '[vc4-candidate] %s\n' "$*"; }
fail() { printf '[vc4-candidate] ERROR: %s\n' "$*" >&2; exit 1; }
relpath() { case "$1" in "$REPO_ROOT"/*) printf '%s\n' "${1#$REPO_ROOT/}" ;; *) printf '%s\n' "$1" ;; esac; }
require_file() { [[ -f "$1" ]] || fail "required file not found: $(relpath "$1")"; }
require_dir() { [[ -d "$1" ]] || fail "required directory not found: $(relpath "$1")"; }

find_tool() {
  local tool="$1"
  if [[ -x "$REPO_ROOT/compiler/build/bin/$tool" ]]; then printf '%s\n' "$REPO_ROOT/compiler/build/bin/$tool"; return 0; fi
  if command -v "$tool" >/dev/null 2>&1; then command -v "$tool"; return 0; fi
  fail "could not find required tool: $tool"
}

check_fixture() {
  require_dir "$TEST_ROOT"
  require_file "$INPUT_MLIR"
  require_file "$EXPECTED_JSON"
  require_dir "$REFERENCE_DIR"
  require_file "$REFERENCE_DIR/Makefile"
  require_file "$REFERENCE_DIR/mailbox.c"
  require_file "$REFERENCE_DIR/mailbox.h"
}

share_source_dir() {
  if [[ -d "$TEST_ROOT/share" ]]; then printf '%s\n' "$TEST_ROOT/share"; return 0; fi
  if [[ -d "$REPO_ROOT/compiler/test/CodeGen/VC4/Hardware/Run/saxpy_full/share" ]]; then
    printf '%s\n' "$REPO_ROOT/compiler/test/CodeGen/VC4/Hardware/Run/saxpy_full/share"; return 0
  fi
  return 1
}

copy_assembler_share_to() {
  local dst_parent="$1" src
  if src="$(share_source_dir)"; then
    mkdir -p "$dst_parent"
    rm -rf "$dst_parent/share"
    cp -R "$src" "$dst_parent/share"
  fi
}

copy_assembler_share() {
  copy_assembler_share_to "$GENERATED_DIR"
  copy_assembler_share_to "$AUTO_ROOT/candidates"
}

check_generated_bundle() {
  require_file "$GENERATED_DIR/manifest.json"
  require_file "$GENERATED_DIR/kernel_launch.c"
  require_file "$GENERATED_DIR/kernel_launch.h"
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
  if [[ -f "$GENERATED_DIR/manifest.json" && -f "$GENERATED_DIR/kernel_launch.c" && -f "$GENERATED_DIR/kernel_launch.h" ]]; then
    log "using existing generated candidate artifacts: $(relpath "$GENERATED_DIR")"
    copy_assembler_share
    return 0
  fi
  run_vc4_codegen
}

manifest_kernel_records() {
  require_file "$GENERATED_DIR/manifest.json"
  python3 - "$GENERATED_DIR" <<'PY_RECORDS'
import json, re, sys
from pathlib import Path
bundle = Path(sys.argv[1])
data = json.loads((bundle / 'manifest.json').read_text())

def ident(s):
    s = re.sub(r'[^0-9A-Za-z_]', '_', str(s or ''))
    if not s or not re.match(r'[A-Za-z_]', s[0]):
        s = 'vc4_' + s
    return s

kernels = data.get('kernels')
if isinstance(kernels, list):
    for i, k in enumerate(kernels):
        qasm = k.get('qasm_path') or ('kernel.qasm' if i == 0 else f'kernel_{i}.qasm')
        code = k.get('code_symbol') or ident(k.get('public_name') or k.get('symbol_name') or f'kernel_{i}') + 'shader'
        public = k.get('public_name') or k.get('symbol_name') or f'kernel_{i}'
        print(f'{qasm}\t{code}\t{public}')
else:
    public = data.get('public_name') or data.get('c_entry_point') or data.get('kernel') or 'kernel'
    print(f'kernel.qasm\tkernelshader\t{public}')
PY_RECORDS
}

assemble_candidate() {
  ensure_generated
  local vc4asm_tool
  vc4asm_tool="$(find_tool vc4asm)"
  local records qasm_rel code_symbol public qasm_path out_c out_h out
  mapfile -t records < <(manifest_kernel_records)
  [[ "${#records[@]}" -ge 1 ]] || fail "manifest contains no kernels to assemble"
  for record in "${records[@]}"; do
    IFS=$'\t' read -r qasm_rel code_symbol public <<<"$record"
    qasm_path="$GENERATED_DIR/$qasm_rel"
    require_file "$qasm_path"
    out_c="$GENERATED_DIR/${code_symbol}.c"
    out_h="$GENERATED_DIR/${code_symbol}.h"
    log "assembling $(relpath "$qasm_path") as ${code_symbol}.c/.h"
    if ! out="$(
      (
        cd "$GENERATED_DIR" &&
        rm -f "${code_symbol}.c" "${code_symbol}.h" &&
        "$vc4asm_tool" -c "${code_symbol}.c" -h "${code_symbol}.h" "$qasm_rel"
      ) 2>&1
    )"; then
      printf '%s\n' "$out" >&2
      fail "vc4asm failed for $TEST_NAME kernel $public"
    fi
    if [[ -n "$out" ]]; then
      printf '%s\n' "$out" >&2
      fail "vc4asm produced unexpected output for $TEST_NAME kernel $public"
    fi
    require_file "$out_c"; require_file "$out_h"
  done
  # Compatibility aliases for existing one-kernel M1 harnesses.
  if [[ "${#records[@]}" -eq 1 ]]; then
    IFS=$'\t' read -r qasm_rel code_symbol public <<<"${records[0]}"
    cp "$GENERATED_DIR/${code_symbol}.c" "$GENERATED_DIR/kernelshader.c"
    cp "$GENERATED_DIR/${code_symbol}.h" "$GENERATED_DIR/kernelshader.h"
  fi
}

extract_public_name() {
  python3 - "$GENERATED_DIR/manifest.json" <<'PY_PUBLIC'
import json, sys
from pathlib import Path
m=json.loads(Path(sys.argv[1]).read_text())
ks=m.get('kernels')
if isinstance(ks, list) and len(ks)==1:
    p=ks[0].get('public_name') or ks[0].get('symbol_name')
else:
    p=m.get('public_name') or m.get('c_entry_point') or m.get('kernel')
if not isinstance(p,str) or not p:
    raise SystemExit('manifest public name not found')
print(p)
PY_PUBLIC
}

derive_kernel_base() {
  local public_name="$1"
  if [[ "$public_name" == *_launch ]]; then printf '%s\n' "${public_name%_launch}"; else printf '%s\n' "$public_name"; fi
}

select_harness_path() {
  local candidate="$CANDIDATE_DIR/${TEST_NAME}_candidate_harness.c"
  if [[ -f "$candidate" ]]; then printf '%s\n' "$candidate"; return 0; fi
  local named="$CANDIDATE_DIR/${TEST_NAME}_harness.c"
  if [[ -f "$named" ]]; then printf '%s\n' "$named"; return 0; fi
  local -a matches
  mapfile -t matches < <(find "$REFERENCE_DIR" -maxdepth 1 -type f -name '3-test-*.c' -print | sort)
  if [[ "${#matches[@]}" -eq 1 ]]; then printf '%s\n' "${matches[0]}"; return 0; fi
  mapfile -t matches < <(find "$REFERENCE_DIR" -maxdepth 1 -type f -name "${TEST_NAME}_harness.c" -print | sort)
  if [[ "${#matches[@]}" -eq 1 ]]; then printf '%s\n' "${matches[0]}"; return 0; fi
  fail "could not find candidate/<test>_candidate_harness.c or exactly one reference harness"
}

write_candidate_makefile() {
  local harness_name="$1"
  shift
  local common_src="mailbox.c kernel_launch.c $*"
  cat > "$WORK_DIR/Makefile" <<EOF_MAKE
LIBS += \$(CS240LX_2025_PATH)/lib/libgcc.a \$(CS240LX_2025_PATH)/libpi/libpi.a

export OPT_LEVEL := -O3

COMMON_SRC := ${common_src}

PROGS := ${harness_name}

STAFF_OBJS += \$(CS240LX_2025_PATH)/libpi/staff-objs/staff-hw-spi.o
STAFF_OBJS += \$(CS240LX_2025_PATH)/libpi/staff-objs/kmalloc.o

RUN ?= 0

BOOTLOADER = pi-install
EXCLUDE ?= grep -v simple_boot
GREP_STR := 'HASH:\|ERROR:\|PANIC:\|SUCCESS:\|VC4_TEST_RESULT\|VC4_RUNTIME_LAYOUT\|VC4_KERNEL_LAUNCH\|NRF:'
include \$(CS240LX_2025_PATH)/libpi/mk/Makefile.robust
EOF_MAKE
}

write_workdir_run_sh() {
  local bin_name="$1"
  cat > "$WORK_DIR/run.sh" <<EOF_RUN
#!/usr/bin/env bash
set -euo pipefail

echo "RUNNING MAKE"
make RUN=0 ${bin_name}

echo "RUNNING PI INSTALL"
"\${VC4_PI_INSTALL_CMD:-pi-install}" "./${bin_name}"
EOF_RUN
  chmod +x "$WORK_DIR/run.sh"
}

prepare_workdir() {
  assemble_candidate
  local public_name kernel_base harness_path harness_name bin_name shader_sources
  public_name="$(extract_public_name || true)"
  kernel_base="$(derive_kernel_base "${public_name:-$TEST_NAME}")"
  harness_path="$(select_harness_path)"
  harness_name="$(basename "$harness_path")"
  bin_name="${harness_name%.c}.bin"

  log "preparing candidate workdir $(relpath "$WORK_DIR")"
  rm -rf "$WORK_DIR"
  mkdir -p "$WORK_DIR"
  copy_assembler_share_to "$HARDWARE_ROOT"
  copy_assembler_share_to "$WORK_DIR"

  cp "$harness_path" "$WORK_DIR/$harness_name"
  cp "$REFERENCE_DIR/mailbox.c" "$WORK_DIR/mailbox.c"
  cp "$REFERENCE_DIR/mailbox.h" "$WORK_DIR/mailbox.h"
  if [[ -f "$SCRIPT_DIR/vc4_m2_candidate_test_helpers.h" ]]; then
    cp "$SCRIPT_DIR/vc4_m2_candidate_test_helpers.h" "$WORK_DIR/vc4_m2_candidate_test_helpers.h"
  fi

  cp "$GENERATED_DIR/manifest.json" "$WORK_DIR/manifest.json"
  cp "$GENERATED_DIR/kernel_launch.c" "$WORK_DIR/kernel_launch.c"
  cp "$GENERATED_DIR/kernel_launch.h" "$WORK_DIR/kernel_launch.h"

  shader_sources=()
  while IFS=$'\t' read -r qasm_rel code_symbol public; do
    cp "$GENERATED_DIR/$qasm_rel" "$WORK_DIR/$(basename "$qasm_rel")"
    cp "$GENERATED_DIR/${code_symbol}.c" "$WORK_DIR/${code_symbol}.c"
    cp "$GENERATED_DIR/${code_symbol}.h" "$WORK_DIR/${code_symbol}.h"
    shader_sources+=("${code_symbol}.c")
    # Compatibility aliases for old reference harness fallback.
    if [[ "${#shader_sources[@]}" -eq 1 ]]; then
      cp "$GENERATED_DIR/${code_symbol}.c" "$WORK_DIR/kernelshader.c"
      cp "$GENERATED_DIR/${code_symbol}.h" "$WORK_DIR/kernelshader.h"
      cp "$GENERATED_DIR/${code_symbol}.c" "$WORK_DIR/${kernel_base}shader.c"
      cp "$GENERATED_DIR/${code_symbol}.h" "$WORK_DIR/${kernel_base}shader.h"
      cp "$GENERATED_DIR/kernel_launch.c" "$WORK_DIR/${kernel_base}_launch.c"
      cp "$GENERATED_DIR/kernel_launch.h" "$WORK_DIR/${kernel_base}_launch.h"
    fi
  done < <(manifest_kernel_records)

  write_candidate_makefile "$harness_name" "${shader_sources[@]}"
  write_workdir_run_sh "$bin_name"

  cat > "$WORK_DIR/README.generated.md" <<EOF_README
# Generated candidate hardware workdir

Generated by compiler/test/CodeGen/VC4/Support/run_candidate_codegen_test.sh.
This directory is under .vc4_auto and must not be committed.

Test: $TEST_NAME
Harness: $harness_name
Binary: $bin_name
EOF_README
  log "candidate workdir ready: $(relpath "$WORK_DIR")"
}

build_candidate() {
  prepare_workdir
  local harness_name bin_name
  harness_name="$(basename "$(select_harness_path)")"
  bin_name="${harness_name%.c}.bin"
  log "building candidate binary in $(relpath "$WORK_DIR") without hardware execution"
  (cd "$WORK_DIR" && make RUN=0 "$bin_name")
  require_file "$WORK_DIR/$bin_name"
}

run_candidate() {
  prepare_workdir
  log "running candidate hardware workdir $(relpath "$WORK_DIR")"
  (cd "$WORK_DIR" && bash run.sh)
}

case "$PHASE" in
  clean) rm -rf "$GENERATED_DIR" "$HARDWARE_ROOT"; log "removed generated candidate state for $TEST_NAME" ;;
  generate) check_fixture; run_vc4_codegen ;;
  assemble) assemble_candidate ;;
  build) build_candidate ;;
  run|all) run_candidate ;;
  workdir) prepare_workdir; printf '%s\n' "$WORK_DIR" ;;
  *) usage; exit 2 ;;
esac
