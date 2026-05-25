#!/usr/bin/env bash
# Generate, assemble, build, or run a VC4Tile codegen candidate program bundle for
# a hardware ground-truth test without mutating the checked-in reference side.
#
# M2 program-bundle behavior:
#   * VC4_CODEGEN_STATE_ROOT selects .vc4_auto/vc4tile_m4 cleanly.
#   * manifest-v2 kernels[] are treated as the general case, including the
#     single-kernel case.
#   * every manifest-listed qasm_path is assembled to its code_symbol .c/.h.
#   * candidate/<test>_candidate_harness.c is required for normal M2 hardware
#     candidates; reference harness fallback is debug-only and opt-in.

set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
usage:
  run_vc4tile_candidate_codegen_test.sh --self-test
  run_vc4tile_candidate_codegen_test.sh TEST_NAME [generate|assemble|build|run|all|workdir|clean]
USAGE
}

if [[ $# -eq 1 && "$1" == "--self-test" ]]; then
  TEST_NAME="__self_test__"
  PHASE="self-test"
  shift
elif [[ $# -lt 2 ]]; then
  usage
  exit 2
else
  TEST_NAME="$1"
  PHASE="$2"
  shift 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$REPO_ROOT" ]]; then
  REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
fi

TEST_ROOT="$REPO_ROOT/compiler/test/CodeGen/VC4Tile/Hardware/Run/$TEST_NAME"
INPUT_MLIR="$TEST_ROOT/input.mlir"
if [[ ! -f "$INPUT_MLIR" && -f "$TEST_ROOT/input.vc4tile.mlir" ]]; then
  INPUT_MLIR="$TEST_ROOT/input.vc4tile.mlir"
fi
EXPECTED_JSON="$TEST_ROOT/expected.json"
REFERENCE_DIR="$TEST_ROOT/reference"
CANDIDATE_DIR="$TEST_ROOT/candidate"
BUNDLE_ONLY_FIXTURE=0

AUTO_ROOT_RAW="${VC4_CODEGEN_STATE_ROOT:-.vc4_auto/vc4tile_m4}"
case "$AUTO_ROOT_RAW" in
  /*) AUTO_ROOT="$AUTO_ROOT_RAW" ;;
  *) AUTO_ROOT="$REPO_ROOT/$AUTO_ROOT_RAW" ;;
esac
GENERATED_DIR="$AUTO_ROOT/candidates/$TEST_NAME"
HARDWARE_ROOT="$AUTO_ROOT/hardware/$TEST_NAME"
WORK_DIR="$HARDWARE_ROOT/candidate_work"

log() { printf '[vc4tile-candidate] %s\n' "$*"; }
fail() { printf '[vc4tile-candidate] ERROR: %s\n' "$*" >&2; exit 1; }
relpath() { case "$1" in "$REPO_ROOT"/*) printf '%s\n' "${1#$REPO_ROOT/}" ;; *) printf '%s\n' "$1" ;; esac; }
require_file() { [[ -f "$1" ]] || fail "required file not found: $(relpath "$1")"; }
require_dir() { [[ -d "$1" ]] || fail "required directory not found: $(relpath "$1")"; }

vc4_candidate_transient_preboot_failure_reason() {
  local log_path="$1"
  [[ -f "$log_path" ]] || return 1

  if grep -Eq 'VC4_RUNTIME_LAYOUT|VC4_KERNEL_LAUNCH|VC4_TEST_RESULT|VC4_KERNEL_LAUNCH_ERROR|VC4_HEAP_STATS|DONE!!!' "$log_path"; then
    return 1
  fi

  if grep -Fq 'tty-USB read() returned 0 bytes.  r/pi not responding [reboot it?]' "$log_path"; then
    printf '%s\n' 'tty-USB read returned 0 bytes'
    return 0
  fi
  if grep -Fq 'GET_CODE op mismatch' "$log_path"; then
    printf '%s\n' 'GET_CODE op mismatch'
    return 0
  fi
  if grep -Fq 'PANIC:pi-boot failed' "$log_path"; then
    printf '%s\n' 'PANIC:pi-boot failed'
    return 0
  fi
  if grep -Fq 'simple-boot.c:ck_eq32' "$log_path"; then
    printf '%s\n' 'simple-boot.c:ck_eq32'
    return 0
  fi
  if grep -Fq 'waiting for a start' "$log_path"; then
    printf '%s\n' 'waiting for a start'
    return 0
  fi
  if grep -Eq 'pi-install: .* about to boot' "$log_path"; then
    printf '%s\n' 'pi-install about to boot'
    return 0
  fi

  return 1
}

vc4_candidate_is_transient_preboot_failure() {
  vc4_candidate_transient_preboot_failure_reason "$1" >/dev/null
}

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
}

manifest_kernel_records() {
  require_file "$GENERATED_DIR/manifest.json"
  python3 - "$GENERATED_DIR" <<'PY_RECORDS'
import json, re, sys
from pathlib import Path, PurePosixPath
bundle = Path(sys.argv[1])
data = json.loads((bundle / 'manifest.json').read_text())

if data.get('schema_version') != 2:
    raise SystemExit('manifest schema_version 2 is required for VC4Tile candidate assembly')

kernels = data.get('kernels')
if not isinstance(kernels, list) or not kernels:
    raise SystemExit('manifest must contain non-empty kernels[] for VC4Tile candidate assembly')

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
        raise SystemExit('VC4Tile bundles must use kernels[].qasm_path, not a root-level singleton QASM artifact')
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

validate_scheduled_vc4_file() {
  local path="$1"
  require_file "$path"
  python3 - "$path" <<'PY_VALIDATE_VC4'
from pathlib import Path
import re
import sys
path = Path(sys.argv[1])
text = path.read_text(encoding='utf-8', errors='replace')
count = len(re.findall(r'(?<![A-Za-z0-9_.])(?:"vc4\.module"|vc4\.module)(?![A-Za-z0-9_.])', text))
if count != 1:
    raise SystemExit(f'{path}: expected exactly one scheduled vc4.module, found {count}')
if 'ssavc4.' in text:
    raise SystemExit(f'{path}: still contains ssavc4 operations after --convert-ssavc4-to-vc4')
if 'vc4tile.' in text:
    raise SystemExit(f'{path}: still contains vc4tile operations after lowering')
if 'vc4.qpu.' not in text:
    raise SystemExit(f'{path}: scheduled VC4 output contains no vc4.qpu.* operations')
PY_VALIDATE_VC4
}

check_generated_bundle() {
  require_file "$GENERATED_DIR/manifest.json"
  require_file "$GENERATED_DIR/kernel_launch.c"
  require_file "$GENERATED_DIR/kernel_launch.h"
  manifest_kernel_records >/dev/null
}

run_vc4_codegen() {
  local vc4_opt vc4_codegen lowered_dir lowered_tmp_dir lowered_ssavc4 scheduled_vc4 stable_ssavc4 stable_vc4
  vc4_opt="$(find_tool vc4-opt)"
  vc4_codegen="$(find_tool vc4-codegen)"
  rm -rf "$GENERATED_DIR"
  mkdir -p "$GENERATED_DIR"
  lowered_dir="$AUTO_ROOT/lowered"
  mkdir -p "$lowered_dir"
  lowered_tmp_dir="$(mktemp -d "$lowered_dir/${TEST_NAME}.tmp.XXXXXX")"
  lowered_ssavc4="$lowered_tmp_dir/${TEST_NAME}.ssavc4.mlir"
  scheduled_vc4="$lowered_tmp_dir/${TEST_NAME}.vc4.mlir"
  stable_ssavc4="$lowered_dir/${TEST_NAME}.ssavc4.mlir"
  stable_vc4="$lowered_dir/${TEST_NAME}.vc4.mlir"
  log "lowering $(relpath "$INPUT_MLIR") to SSAVC4 at $(relpath "$lowered_ssavc4")"
  "$vc4_opt" "$INPUT_MLIR" --convert-vc4tile-to-ssavc4 -o "$lowered_ssavc4"
  require_file "$lowered_ssavc4"
  log "lowering $(relpath "$lowered_ssavc4") to scheduled VC4 at $(relpath "$scheduled_vc4")"
  "$vc4_opt" "$lowered_ssavc4" \
    --convert-ssavc4-to-vc4 \
    --vc4-verify-emit-contract \
    --vc4-verify-scheduled-hardware-rules \
    --vc4-verify-scheduled-adjacent-hazards \
    --vc4-verify-scheduled-io-spacing \
    --vc4-verify-scheduled-peripheral-accesses \
    -o "$scheduled_vc4"
  validate_scheduled_vc4_file "$scheduled_vc4"
  log "generating $(relpath "$GENERATED_DIR") from $(relpath "$scheduled_vc4")"
  "$vc4_codegen" "$scheduled_vc4" --emit-bundle "$GENERATED_DIR"
  cp "$lowered_ssavc4" "$stable_ssavc4"
  cp "$scheduled_vc4" "$stable_vc4"
  cp "$INPUT_MLIR" "$GENERATED_DIR/input.vc4tile.mlir"
  cp "$lowered_ssavc4" "$GENERATED_DIR/lowered.ssavc4.mlir"
  cp "$scheduled_vc4" "$GENERATED_DIR/scheduled.vc4.mlir"
  rm -rf "$lowered_tmp_dir"
  check_generated_bundle
}

ensure_generated() {
  check_fixture
  if [[ "${VC4_REUSE_GENERATED_CANDIDATE:-0}" == "1" &&
        -f "$GENERATED_DIR/manifest.json" &&
        -f "$GENERATED_DIR/kernel_launch.c" &&
        -f "$GENERATED_DIR/kernel_launch.h" ]]; then
    log "VC4_REUSE_GENERATED_CANDIDATE=1; reusing existing generated candidate artifacts: $(relpath "$GENERATED_DIR")"
    check_generated_bundle
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

  if [[ "${VC4_ALLOW_REFERENCE_HARNESS_FALLBACK:-0}" != "1" ]]; then
    fail "VC4Tile candidate fixture $TEST_NAME needs $(relpath "$candidate") using kernel_launch.h and vc4Malloc/vc4Memcpy APIs; reference harness fallback is disabled"
  fi

  log "WARNING: VC4_ALLOW_REFERENCE_HARNESS_FALLBACK=1; using legacy harness fallback for $TEST_NAME. This is debug-only and forbidden for normal VC4Tile candidate verification."

  local named="$CANDIDATE_DIR/${TEST_NAME}_harness.c"
  if [[ -f "$named" ]]; then printf '%s\n' "$named"; return 0; fi
  local -a matches
  if [[ -d "$REFERENCE_DIR" ]]; then
    mapfile -t matches < <(find "$REFERENCE_DIR" -maxdepth 1 -type f -name '3-test-*.c' -print | sort)
    if [[ "${#matches[@]}" -eq 1 ]]; then printf '%s\n' "${matches[0]}"; return 0; fi
    mapfile -t matches < <(find "$REFERENCE_DIR" -maxdepth 1 -type f -name "${TEST_NAME}_harness.c" -print | sort)
    if [[ "${#matches[@]}" -eq 1 ]]; then printf '%s\n' "${matches[0]}"; return 0; fi
  fi
  fail "VC4_ALLOW_REFERENCE_HARNESS_FALLBACK=1 but no legacy fallback harness was found for $TEST_NAME"
}

write_candidate_makefile() {
  local harness_name="$1"
  shift
  local common_src="kernel_launch.c $*"
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

write_case_config_header() {
  python3 - "$WORK_DIR/vc4_case_config.h" <<'PY_CASE_CONFIG'
import json
import os
import re
import sys
from pathlib import Path

out = Path(sys.argv[1])
integer = re.compile(r"^[+-]?(?:0[xX][0-9A-Fa-f]+|[0-9]+)$")
name = re.compile(r"^(?:VC4_CASE|VC4_MATRIX)_[A-Z0-9_]+$")

lines = [
    "/* Generated by run_vc4tile_candidate_codegen_test.sh; do not commit. */",
    "#ifndef VC4_CASE_CONFIG_H",
    "#define VC4_CASE_CONFIG_H",
    "",
]

for key, value in sorted(os.environ.items()):
    if not name.match(key):
        continue
    if integer.match(value):
        literal = value
    else:
        literal = json.dumps(value)
    lines.append(f"#define {key} {literal}")

lines.extend([
    "",
    "#endif /* VC4_CASE_CONFIG_H */",
    "",
])
out.write_text("\n".join(lines), encoding="utf-8")
PY_CASE_CONFIG
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
  local elf_name="${bin_name%.bin}.elf"
  printf '%s\n' "$bin_name" > "$WORK_DIR/.vc4_candidate_bin"
  cat > "$WORK_DIR/run.sh" <<EOF_RUN
#!/usr/bin/env bash
set -euo pipefail

echo "RUNNING MAKE"
make RUN=0 ${bin_name}
EOF_RUN

  if [[ "$BUNDLE_ONLY_FIXTURE" -eq 0 ]]; then
    cat >> "$WORK_DIR/run.sh" <<EOF_RUN
if command -v arm-none-eabi-nm >/dev/null 2>&1 && [[ -f "objs/${elf_name}" ]]; then
  nm_out="\$(arm-none-eabi-nm "objs/${elf_name}")"
  if grep -Fq 'vc4_codegen_weak_mailbox_storage' <<<"\$nm_out"; then
    echo "ERROR: candidate ELF contains generated weak mailbox storage instead of libpi VC4 runtime" >&2
    exit 1
  fi
  for sym in qpu_enable mem_alloc mem_lock mem_unlock mem_free vc4LaunchKernel; do
    if ! grep -Eq "[[:space:]][TtWw][[:space:]]+\${sym}\$" <<<"\$nm_out"; then
      echo "ERROR: candidate ELF is missing VC4 runtime symbol: \${sym}" >&2
      exit 1
    fi
    if grep -Eq "[[:space:]][Ww][[:space:]]+\${sym}\$" <<<"\$nm_out"; then
      echo "ERROR: candidate ELF uses weak VC4 runtime symbol instead of libpi: \${sym}" >&2
      exit 1
    fi
  done
fi

EOF_RUN
  fi

  cat >> "$WORK_DIR/run.sh" <<EOF_RUN
echo "RUNNING PI INSTALL"
"\${VC4_PI_INSTALL_CMD:-pi-install}" "./${bin_name}"
EOF_RUN
  chmod +x "$WORK_DIR/run.sh"
}

check_candidate_runtime_symbols() {
  local bin_name="$1"
  local elf_path="$WORK_DIR/objs/${bin_name%.bin}.elf"
  [[ -f "$elf_path" ]] || return 0
  command -v arm-none-eabi-nm >/dev/null 2>&1 || return 0
  if [[ "$BUNDLE_ONLY_FIXTURE" -eq 1 ]]; then
    return 0
  fi

  local nm_out sym
  nm_out="$(arm-none-eabi-nm "$elf_path")"
  if grep -Fq 'vc4_codegen_weak_mailbox_storage' <<<"$nm_out"; then
    fail "candidate ELF contains generated weak mailbox storage instead of libpi VC4 runtime"
  fi
  for sym in qpu_enable mem_alloc mem_lock mem_unlock mem_free vc4LaunchKernel; do
    if ! grep -Eq "[[:space:]][TtWw][[:space:]]+${sym}$" <<<"$nm_out"; then
      fail "candidate ELF is missing VC4 runtime symbol: ${sym}"
    fi
    if grep -Eq "[[:space:]][Ww][[:space:]]+${sym}$" <<<"$nm_out"; then
      fail "candidate ELF uses weak VC4 runtime symbol instead of libpi: ${sym}"
    fi
  done
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

  cp "$GENERATED_DIR/manifest.json" "$WORK_DIR/manifest.json"
  cp "$GENERATED_DIR/kernel_launch.c" "$WORK_DIR/kernel_launch.c"
  cp "$GENERATED_DIR/kernel_launch.h" "$WORK_DIR/kernel_launch.h"
  write_case_config_header

  shader_sources=()
  while IFS= read -r src; do
    shader_sources+=("$src")
  done < <(copy_manifest_artifacts_to_workdir)

  write_bundle_smoke_harness "$harness_name"
  write_bundle_smoke_makefile "$harness_name" "${shader_sources[@]}"
  write_workdir_run_sh "$bin_name"

  cat > "$WORK_DIR/README.generated.md" <<EOF_README
# Generated VC4Tile program-bundle smoke workdir

Generated by compiler/test/CodeGen/VC4Tile/Support/run_vc4tile_candidate_codegen_test.sh.
This directory is under .vc4_auto and must not be committed.

Test: $TEST_NAME
Harness: $harness_name
Binary: $bin_name
EOF_README
  log "program-bundle smoke workdir ready: $(relpath "$WORK_DIR")"
}

prepare_workdir() {
  assemble_candidate
  local harness_path harness_name bin_name shader_sources
  harness_path="$(select_harness_path)"
  harness_name="$(basename "$harness_path")"
  bin_name="${harness_name%.c}.bin"

  log "preparing candidate workdir $(relpath "$WORK_DIR")"
  rm -rf "$WORK_DIR"
  mkdir -p "$WORK_DIR"

  cp "$harness_path" "$WORK_DIR/$harness_name"
  if [[ -f "$SCRIPT_DIR/vc4_m2_candidate_test_helpers.h" ]]; then
    cp "$SCRIPT_DIR/vc4_m2_candidate_test_helpers.h" "$WORK_DIR/vc4_m2_candidate_test_helpers.h"
  elif [[ -f "$REPO_ROOT/compiler/test/CodeGen/VC4/Support/vc4_m2_candidate_test_helpers.h" ]]; then
    cp "$REPO_ROOT/compiler/test/CodeGen/VC4/Support/vc4_m2_candidate_test_helpers.h" "$WORK_DIR/vc4_m2_candidate_test_helpers.h"
  fi

  cp "$GENERATED_DIR/manifest.json" "$WORK_DIR/manifest.json"
  cp "$GENERATED_DIR/kernel_launch.c" "$WORK_DIR/kernel_launch.c"
  cp "$GENERATED_DIR/kernel_launch.h" "$WORK_DIR/kernel_launch.h"
  write_case_config_header

  shader_sources=()
  while IFS=$'\t' read -r qasm_rel code_symbol public; do
    cp "$GENERATED_DIR/$qasm_rel" "$WORK_DIR/$(basename "$qasm_rel")"
    cp "$GENERATED_DIR/${code_symbol}.c" "$WORK_DIR/${code_symbol}.c"
    cp "$GENERATED_DIR/${code_symbol}.h" "$WORK_DIR/${code_symbol}.h"
    shader_sources+=("${code_symbol}.c")
  done < <(manifest_kernel_records)

  write_candidate_makefile "$harness_name" "${shader_sources[@]}"
  write_workdir_run_sh "$bin_name"

  cat > "$WORK_DIR/README.generated.md" <<EOF_README
# Generated candidate hardware workdir

Generated by compiler/test/CodeGen/VC4Tile/Support/run_vc4tile_candidate_codegen_test.sh.
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
  check_candidate_runtime_symbols "$bin_name"
}

vc4_candidate_power_cycle_if_needed() {
  local message="$1"
  if [[ "${VC4_SKIP_POWER_CYCLE:-0}" == "1" ]]; then
    log "skipping Pi power cycle: ${message}"
    return 0
  fi

  local cmd="${VC4_PI_POWER_CYCLE_CMD:-uhubctl -l 0-1 -a cycle}"
  local sleep_sec="${VC4_PI_POWER_CYCLE_SLEEP_SEC:-1}"
  local timeout_sec="${VC4_PI_POWER_CYCLE_TIMEOUT_SEC:-10}"
  local rc=0

  log "${message}: ${cmd}"
  if [[ -n "$cmd" ]]; then
    set +e
    if [[ "$timeout_sec" =~ ^[0-9]+$ ]] && [[ "$timeout_sec" -gt 0 ]]; then
      python3 - "$timeout_sec" "$cmd" <<'PY_POWER_CYCLE'
import os
import signal
import subprocess
import sys

timeout = int(sys.argv[1])
cmd = sys.argv[2]
process = subprocess.Popen(cmd, shell=True, start_new_session=True)
try:
    rc = process.wait(timeout=timeout)
except subprocess.TimeoutExpired:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
    print(f"[vc4tile-candidate] WARNING: Pi power cycle command timed out after {timeout}s", file=sys.stderr)
    sys.exit(124)
sys.exit(rc)
PY_POWER_CYCLE
      rc=$?
    else
      bash -lc "$cmd"
      rc=$?
    fi
    set -e

    if [[ "$rc" -ne 0 ]]; then
      if [[ "${VC4_REQUIRE_POWER_CYCLE:-0}" == "1" ]]; then
        fail "Pi power cycle command failed with exit code ${rc}: ${cmd}"
      fi
      log "continuing after non-fatal Pi power cycle command failure (exit=${rc})"
    fi
  fi

  if [[ "$sleep_sec" =~ ^[0-9]+$ ]] && [[ "$sleep_sec" -gt 0 ]]; then
    sleep "$sleep_sec"
  fi
}

run_candidate() {
  prepare_workdir
  log "running candidate hardware workdir $(relpath "$WORK_DIR")"

  local max_attempts="${VC4_RUN_SH_MAX_ATTEMPTS:-3}"
  if ! [[ "$max_attempts" =~ ^[0-9]+$ ]] || [[ "$max_attempts" -lt 1 ]]; then
    fail "VC4_RUN_SH_MAX_ATTEMPTS must be a positive integer, got: $max_attempts"
  fi

  local attempt=1
  local run_rc=0
  local attempt_log="$WORK_DIR/.vc4_candidate_run_attempt.log"
  local retry_reason=""
  while true; do
    if [[ "$attempt" -eq 1 ]]; then
      vc4_candidate_power_cycle_if_needed "power cycling Pi before candidate run"
    else
      log "retrying candidate run after transient pre-boot failure (attempt ${attempt}/${max_attempts}; matched: ${retry_reason})"
      vc4_candidate_power_cycle_if_needed "power cycling Pi before retry"
    fi

    rm -f "$attempt_log"
    set +e
    (
      cd "$WORK_DIR"
      bash run.sh
    ) 2>&1 | tee "$attempt_log"
    run_rc=${PIPESTATUS[0]}
    set -e

    if [[ "$run_rc" -ne 0 ]] && retry_reason="$(vc4_candidate_transient_preboot_failure_reason "$attempt_log")" && [[ "$attempt" -lt "$max_attempts" ]]; then
      log "transient pre-boot failure matched in attempt log: ${retry_reason}"
      printf '[vc4tile-candidate] transient pre-boot failure matched: %s\n' "$retry_reason" >> "$attempt_log"
      attempt=$((attempt + 1))
      continue
    fi

    break
  done

  return "$run_rc"
}

vc4_candidate_self_test_assert_true() {
  local name="$1"
  local log_path="$2"
  local reason
  if ! reason="$(vc4_candidate_transient_preboot_failure_reason "$log_path")"; then
    fail "self-test expected transient pre-boot true: $name"
  fi
  log "self-test true: ${name} (${reason})"
}

vc4_candidate_self_test_assert_false() {
  local name="$1"
  local log_path="$2"
  if vc4_candidate_is_transient_preboot_failure "$log_path"; then
    fail "self-test expected transient pre-boot false: $name"
  fi
  log "self-test false: ${name}"
}

vc4_candidate_self_test() {
  local tmp marker
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' RETURN

  printf '%s\n' 'tty-USB read() returned 0 bytes.  r/pi not responding [reboot it?]' > "$tmp/tty-read-zero.log"
  vc4_candidate_self_test_assert_true "tty-read-zero log" "$tmp/tty-read-zero.log"

  {
    printf '%s\n' 'GET_CODE op mismatch: expected 55556666, got 00000000'
    printf '%s\n' 'simple-boot.c:ck_eq32:30:PANIC:pi-boot failed'
  } > "$tmp/get-code-panic.log"
  vc4_candidate_self_test_assert_true "GET_CODE mismatch plus pi-boot panic log" "$tmp/get-code-panic.log"

  marker='VC4_KERNEL_LAUNCH_ERROR'
  printf '%s\n' "${marker} reason=wave_timeout" > "$tmp/launch-error.log"
  vc4_candidate_self_test_assert_false "kernel launch error log" "$tmp/launch-error.log"

  marker='VC4_TEST_''RESULT'
  printf '%s\n' "${marker} status=FAIL" > "$tmp/test-result-fail.log"
  vc4_candidate_self_test_assert_false "test result fail log" "$tmp/test-result-fail.log"

  marker='VC4_RUNTIME_''LAYOUT'
  {
    printf '%s\n' "${marker} heap_base=0x1000"
    printf '%s\n' 'PANIC:pi-boot failed'
  } > "$tmp/runtime-layout-crash.log"
  vc4_candidate_self_test_assert_false "runtime layout then crash log" "$tmp/runtime-layout-crash.log"

  echo "run_vc4tile_candidate_codegen_test.sh self-test PASS"
}

case "$PHASE" in
  clean) rm -rf "$GENERATED_DIR" "$HARDWARE_ROOT"; log "removed generated candidate state for $TEST_NAME" ;;
  generate) check_fixture; run_vc4_codegen ;;
  assemble) assemble_candidate ;;
  build) build_candidate ;;
  run|all) run_candidate ;;
  workdir) prepare_workdir; printf '%s\n' "$WORK_DIR" ;;
  self-test) vc4_candidate_self_test ;;
  *) usage; exit 2 ;;
esac
