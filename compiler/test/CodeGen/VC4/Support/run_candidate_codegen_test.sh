#!/usr/bin/env bash
# Generate, assemble, build, or run a VC4 codegen candidate program bundle for
# a hardware ground-truth test without mutating the checked-in reference side.
#
# M2 program-bundle behavior:
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
BUNDLE_ONLY_FIXTURE=0

# The M2 verifier uses multi_kernel_minimal as a program-bundle smoke fixture.
# A later runtime/heap slice may provide a full CUDA-like candidate harness for
# the same name, but this slice only needs manifest-driven generation, all-QASM
# assembly, and a link smoke check.  Keep that smoke path active even when a
# Hardware/Run skeleton exists so we do not compile future ABI helpers here.
if [[ "$TEST_NAME" == "multi_kernel_minimal" ]]; then
  BUNDLE_ONLY_FIXTURE=1
  EXPECTED_JSON=""
  emit_input="$REPO_ROOT/compiler/test/CodeGen/VC4/Emit/emit-multi-kernel-manifest-v2.mlir"
  if [[ ! -f "$INPUT_MLIR" && -f "$emit_input" ]]; then
    INPUT_MLIR="$emit_input"
  fi
  REFERENCE_DIR="$REPO_ROOT/compiler/test/CodeGen/VC4/Hardware/Run/saxpy_full/reference"
  CANDIDATE_DIR="$TEST_ROOT/candidate"
fi

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
  if [[ "$BUNDLE_ONLY_FIXTURE" -eq 1 ]]; then
    require_file "$INPUT_MLIR"
    return 0
  fi
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

manifest_kernel_records() {
  require_file "$GENERATED_DIR/manifest.json"
  python3 - "$GENERATED_DIR" <<'PY_RECORDS'
import json, re, sys
from pathlib import Path, PurePosixPath
bundle = Path(sys.argv[1])
data = json.loads((bundle / 'manifest.json').read_text())

if data.get('schema_version') != 2:
    raise SystemExit('manifest schema_version 2 is required for M2 candidate assembly')

kernels = data.get('kernels')
if not isinstance(kernels, list) or not kernels:
    raise SystemExit('manifest must contain non-empty kernels[] for M2 candidate assembly')

seen_qasm = set()
seen_code = set()
for i, k in enumerate(kernels):
    if not isinstance(k, dict):
        raise SystemExit(f'manifest kernels[{i}] must be an object')
    qasm = k.get('qasm_path')
    code = k.get('code_symbol')
    public = k.get('public_name') or k.get('symbol_name')
    if not isinstance(qasm, str) or not qasm:
        raise SystemExit(f'manifest kernels[{i}] missing qasm_path')
    if qasm == 'kernel.qasm':
        raise SystemExit('M2 bundles must use kernels[].qasm_path, not a root-level singleton QASM artifact')
    qasm_path = PurePosixPath(qasm)
    if qasm_path.is_absolute() or any(part in ('', '.', '..') for part in qasm_path.parts):
        raise SystemExit(f'manifest kernels[{i}] has unsafe qasm_path {qasm!r}')
    if not (bundle / qasm).is_file():
        raise SystemExit(f'manifest kernels[{i}] qasm_path does not exist: {qasm}')
    if not isinstance(code, str) or not code:
        raise SystemExit(f'manifest kernels[{i}] missing code_symbol')
    if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', code):
        raise SystemExit(f'manifest kernels[{i}] code_symbol is not a C identifier: {code!r}')
    if not isinstance(public, str) or not public:
        raise SystemExit(f'manifest kernels[{i}] missing public_name/symbol_name')
    if qasm in seen_qasm:
        raise SystemExit(f'duplicate qasm_path in manifest: {qasm}')
    if code in seen_code:
        raise SystemExit(f'duplicate code_symbol in manifest: {code}')
    seen_qasm.add(qasm)
    seen_code.add(code)
    print(f'{qasm}\t{code}\t{public}')
PY_RECORDS
}

check_generated_bundle() {
  require_file "$GENERATED_DIR/manifest.json"
  require_file "$GENERATED_DIR/kernel_launch.c"
  require_file "$GENERATED_DIR/kernel_launch.h"
  manifest_kernel_records >/dev/null
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
    check_generated_bundle
    copy_assembler_share
    return 0
  fi
  run_vc4_codegen
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
}

extract_public_name() {
  python3 - "$GENERATED_DIR/manifest.json" <<'PY_PUBLIC'
import json, sys
from pathlib import Path
m=json.loads(Path(sys.argv[1]).read_text())
ks=m.get('kernels')
if isinstance(ks, list) and ks:
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

write_bundle_smoke_makefile() {
  local harness_name="$1"
  shift
  local common_src="$*"
  cat > "$WORK_DIR/Makefile" <<EOF_MAKE
LIBS += \$(CS240LX_2025_PATH)/lib/libgcc.a \$(CS240LX_2025_PATH)/libpi/libpi.a

export OPT_LEVEL := -O3

COMMON_SRC := ${common_src}

PROGS := ${harness_name}

RUN ?= 0

BOOTLOADER = pi-install
EXCLUDE ?= grep -v simple_boot
GREP_STR := 'HASH:\|ERROR:\|PANIC:\|SUCCESS:\|VC4_TEST_RESULT\|NRF:'
include \$(CS240LX_2025_PATH)/libpi/mk/Makefile.robust
EOF_MAKE
}

write_workdir_run_sh() {
  local bin_name="$1"
  printf '%s\n' "$bin_name" > "$WORK_DIR/.vc4_candidate_bin"
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

write_header_alias() {
  local alias_name="$1"
  local code_symbol="$2"
  cat > "$WORK_DIR/${alias_name}.h" <<EOF_ALIAS
#ifndef VC4_CODEGEN_${alias_name}_ALIAS_H
#define VC4_CODEGEN_${alias_name}_ALIAS_H
#include "${code_symbol}.h"
#define ${alias_name} ${code_symbol}
#endif
EOF_ALIAS
}

write_bundle_smoke_harness() {
  local harness_name="$1"
  python3 - "$GENERATED_DIR/manifest.json" "$WORK_DIR/$harness_name" <<'PY_HARNESS'
import json, re, sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2])
kernels = manifest.get('kernels') or []
lines = [
    '#include <stdint.h>',
]
for kernel in kernels:
    code = kernel['code_symbol']
    lines.append(f'#include "{code}.h"')
lines.append('')
lines.append('void notmain(void) {')
lines.append('  volatile uint32_t vc4_codegen_bundle_checksum = 0;')
for kernel in kernels:
    code = kernel['code_symbol']
    if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', code):
        raise SystemExit(f'bad code_symbol: {code!r}')
    lines.append(f'  vc4_codegen_bundle_checksum ^= {code}[0];')
lines.append('  (void)vc4_codegen_bundle_checksum;')
lines.append('}')
lines.append('')
out.write_text('\n'.join(lines))
PY_HARNESS
}

copy_manifest_artifacts_to_workdir() {
  local qasm_rel code_symbol public
  while IFS=$'\t' read -r qasm_rel code_symbol public; do
    cp "$GENERATED_DIR/$qasm_rel" "$WORK_DIR/$(basename "$qasm_rel")"
    cp "$GENERATED_DIR/${code_symbol}.c" "$WORK_DIR/${code_symbol}.c"
    cp "$GENERATED_DIR/${code_symbol}.h" "$WORK_DIR/${code_symbol}.h"
    printf '%s\n' "${code_symbol}.c"
  done < <(manifest_kernel_records)
}

prepare_bundle_only_workdir() {
  assemble_candidate
  local harness_name bin_name shader_sources
  harness_name="${TEST_NAME}_bundle_smoke.c"
  bin_name="${harness_name%.c}.bin"

  log "preparing program-bundle smoke workdir $(relpath "$WORK_DIR")"
  rm -rf "$WORK_DIR"
  mkdir -p "$WORK_DIR"
  copy_assembler_share_to "$HARDWARE_ROOT"
  copy_assembler_share_to "$WORK_DIR"

  cp "$GENERATED_DIR/manifest.json" "$WORK_DIR/manifest.json"
  cp "$GENERATED_DIR/kernel_launch.c" "$WORK_DIR/kernel_launch.c"
  cp "$GENERATED_DIR/kernel_launch.h" "$WORK_DIR/kernel_launch.h"

  shader_sources=()
  while IFS= read -r src; do
    shader_sources+=("$src")
  done < <(copy_manifest_artifacts_to_workdir)

  write_bundle_smoke_harness "$harness_name"
  write_bundle_smoke_makefile "$harness_name" "${shader_sources[@]}"
  write_workdir_run_sh "$bin_name"

  cat > "$WORK_DIR/README.generated.md" <<EOF_README
# Generated M2 program-bundle smoke workdir

Generated by compiler/test/CodeGen/VC4/Support/run_candidate_codegen_test.sh.
This directory is under .vc4_auto and must not be committed.

Test: $TEST_NAME
Harness: $harness_name
Binary: $bin_name
EOF_README
  log "program-bundle smoke workdir ready: $(relpath "$WORK_DIR")"
}

prepare_workdir() {
  if [[ "$BUNDLE_ONLY_FIXTURE" -eq 1 ]]; then
    prepare_bundle_only_workdir
    return 0
  fi

  assemble_candidate
  local public_name kernel_base harness_path harness_name bin_name shader_sources first_code_symbol
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
  first_code_symbol=""
  while IFS=$'\t' read -r qasm_rel code_symbol public; do
    cp "$GENERATED_DIR/$qasm_rel" "$WORK_DIR/$(basename "$qasm_rel")"
    cp "$GENERATED_DIR/${code_symbol}.c" "$WORK_DIR/${code_symbol}.c"
    cp "$GENERATED_DIR/${code_symbol}.h" "$WORK_DIR/${code_symbol}.h"
    shader_sources+=("${code_symbol}.c")
    if [[ -z "$first_code_symbol" ]]; then
      first_code_symbol="$code_symbol"
      # Compatibility headers for older harnesses that include legacy shader
      # names.  These are aliases to the manifest-declared code_symbol; all
      # linked code arrays still come from kernels[].code_symbol.
      write_header_alias "kernelshader" "$code_symbol"
      write_header_alias "${kernel_base}shader" "$code_symbol"
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
  local bin_name
  require_file "$WORK_DIR/.vc4_candidate_bin"
  bin_name="$(cat "$WORK_DIR/.vc4_candidate_bin")"
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
