#!/usr/bin/env bash
# Generate, assemble, build, or run a real TTIR codegen candidate program bundle
# for a hardware ground-truth test without mutating the checked-in reference side.
#
# M2 program-bundle behavior:
#   * VC4_CODEGEN_STATE_ROOT selects the generated-code state root.
#   * manifest-v2 kernels[] are treated as the general case, including the
#     single-kernel case.
#   * every manifest-listed qasm_path is assembled to its code_symbol .c/.h.
#   * candidate/<test>_candidate_harness.c is required for normal M2 hardware
#     candidates; reference harness fallback is debug-only and opt-in.

set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
usage:
  run_ttir_candidate_codegen_test.sh --self-test
  run_ttir_candidate_codegen_test.sh TEST_NAME [generate|assemble|build|run|all|workdir|generated-dir|clean]
  run_ttir_candidate_codegen_test.sh FIXTURE_DIR [generate|assemble|build|run|all|workdir|generated-dir|clean]
USAGE
}

if [[ $# -eq 1 && "$1" == "--self-test" ]]; then
  TEST_NAME="__self_test__"
  TEST_ARG="__self_test__"
  PHASE="self-test"
  shift
elif [[ $# -lt 1 ]]; then
  usage
  exit 2
else
  TEST_ARG="$1"
  PHASE="${2:-all}"
  TEST_NAME="$(basename "$TEST_ARG")"
  shift
  if [[ $# -gt 0 ]]; then shift; fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$REPO_ROOT" ]]; then
  REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
fi
CHECKER="$REPO_ROOT/compiler/test/CodeGen/VC4/Support/check_vc4_test_result.py"

case "$TEST_ARG" in
  /*) TEST_ROOT="$TEST_ARG" ;;
  */*) TEST_ROOT="$REPO_ROOT/$TEST_ARG" ;;
  *) TEST_ROOT="$REPO_ROOT/compiler/test/CodeGen/Triton/Hardware/Run/$TEST_NAME" ;;
esac
INPUT_TTIR="$TEST_ROOT/input.ttir.mlir"
INPUTS_DIR="$TEST_ROOT/inputs"
EXPECTED_JSON="$TEST_ROOT/expected.json"
REFERENCE_DIR="$TEST_ROOT/reference"
CANDIDATE_DIR="$TEST_ROOT/candidate"
BUNDLE_ONLY_FIXTURE=0

AUTO_ROOT_RAW="${VC4_CODEGEN_STATE_ROOT:-.vc4_auto/codegen_ttir}"
case "$AUTO_ROOT_RAW" in
  /*) AUTO_ROOT="$AUTO_ROOT_RAW" ;;
  *) AUTO_ROOT="$REPO_ROOT/$AUTO_ROOT_RAW" ;;
esac
GENERATED_DIR="$AUTO_ROOT/candidates/$TEST_NAME"
HARDWARE_ROOT="$AUTO_ROOT/hardware/$TEST_NAME"
WORK_DIR="$HARDWARE_ROOT/candidate_work"

log() { printf '[ttir-candidate] %s\n' "$*"; }
fail() { printf '[ttir-candidate] ERROR: %s\n' "$*" >&2; exit 1; }
relpath() { case "$1" in "$REPO_ROOT"/*) printf '%s\n' "${1#$REPO_ROOT/}" ;; *) printf '%s\n' "$1" ;; esac; }
require_file() { [[ -f "$1" ]] || fail "required file not found: $(relpath "$1")"; }
require_dir() { [[ -d "$1" ]] || fail "required directory not found: $(relpath "$1")"; }

ttir_input_paths() {
  if [[ -f "$INPUT_TTIR" ]]; then
    printf '%s\n' "$INPUT_TTIR"
    return 0
  fi
  if [[ -d "$INPUTS_DIR" ]]; then
    find "$INPUTS_DIR" -maxdepth 1 -type f -name '*.ttir.mlir' -print | sort
    return 0
  fi
  return 0
}

ttir_input_count() {
  ttir_input_paths | awk 'NF { count++ } END { print count + 0 }'
}

validate_ttir_input_file() {
  local input_path="$1"
  require_file "$input_path"
  python3 - "$input_path" <<'PY_CHECK_TTIR_INPUT'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8", errors="replace")
if not re.search(r'(?<![A-Za-z0-9_])"?tt\.', text):
    raise SystemExit(f"{path}: TTIR hardware input must contain tt dialect operations")
for dialect in ("vc4value", "vc4kernel", "ssavc4", "ttg", "gpu", "nvgpu", "nvvm", "rocdl", "llvm"):
    if re.search(rf'(?<![A-Za-z0-9_])"?{re.escape(dialect)}\.', text):
        raise SystemExit(f"{path}: TTIR hardware input must not contain {dialect}.*")
if re.search(r'(?<![A-Za-z0-9_])"?vc4\.', text):
    raise SystemExit(f"{path}: TTIR hardware input must not contain scheduled vc4.*")
for forbidden in ("ttgir", "ptx", "cubin", "hsaco"):
    if forbidden in text.lower():
        raise SystemExit(f"{path}: TTIR hardware input must not contain {forbidden}")
PY_CHECK_TTIR_INPUT
}

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
  local env_name env_value
  env_name="$(printf '%s\n' "$tool" | tr '[:lower:]-' '[:upper:]_')"
  env_name="VC4_${env_name}"
  env_value="${!env_name:-}"
  if [[ -n "$env_value" && -x "$env_value" ]]; then printf '%s\n' "$env_value"; return 0; fi
  if [[ -n "${VC4_ACTIVE_TOOLS_DIR:-}" && -x "$VC4_ACTIVE_TOOLS_DIR/$tool" ]]; then printf '%s\n' "$VC4_ACTIVE_TOOLS_DIR/$tool"; return 0; fi
  if command -v "$tool" >/dev/null 2>&1; then command -v "$tool"; return 0; fi
  if [[ -x "$REPO_ROOT/compiler/build-triton-llvm/bin/$tool" ]]; then printf '%s\n' "$REPO_ROOT/compiler/build-triton-llvm/bin/$tool"; return 0; fi
  if [[ -x "$REPO_ROOT/compiler/build/bin/$tool" ]]; then printf '%s\n' "$REPO_ROOT/compiler/build/bin/$tool"; return 0; fi
  fail "could not find required tool: $tool"
}

check_fixture() {
  local count input_path
  if [[ "$BUNDLE_ONLY_FIXTURE" -eq 1 ]]; then
    count="$(ttir_input_count)"
    [[ "$count" -gt 0 ]] || fail "required TTIR input not found: $(relpath "$INPUT_TTIR") or $(relpath "$INPUTS_DIR")/*.ttir.mlir"
    while IFS= read -r input_path; do
      [[ -z "$input_path" ]] && continue
      validate_ttir_input_file "$input_path"
    done < <(ttir_input_paths)
    return 0
  fi
  require_dir "$TEST_ROOT"
  count="$(ttir_input_count)"
  [[ "$count" -gt 0 ]] || fail "required TTIR input not found: $(relpath "$INPUT_TTIR") or $(relpath "$INPUTS_DIR")/*.ttir.mlir"
  require_file "$EXPECTED_JSON"
  while IFS= read -r input_path; do
    [[ -z "$input_path" ]] && continue
    validate_ttir_input_file "$input_path"
  done < <(ttir_input_paths)
}

manifest_kernel_records() {
  require_file "$GENERATED_DIR/manifest.json"
  python3 - "$GENERATED_DIR" <<'PY_RECORDS'
import json, re, sys
from pathlib import Path, PurePosixPath
bundle = Path(sys.argv[1])
data = json.loads((bundle / 'manifest.json').read_text())

if data.get('schema_version') != 2:
    raise SystemExit('manifest schema_version 2 is required for TTIR candidate assembly')

kernels = data.get('kernels')
if not isinstance(kernels, list) or not kernels:
    raise SystemExit('manifest must contain non-empty kernels[] for TTIR candidate assembly')

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
        raise SystemExit('TTIR bundles must use kernels[].qasm_path, not a root-level singleton QASM artifact')
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
if 'vc4kernel.' in text:
    raise SystemExit(f'{path}: still contains vc4kernel operations after lowering')
if 'vc4.qpu.' not in text:
    raise SystemExit(f'{path}: scheduled VC4 output contains no vc4.qpu.* operations')
PY_VALIDATE_VC4
}

validate_vc4kernel_verified_file() {
  local path="$1"
  require_file "$path"
  python3 - "$path" <<'PY_VALIDATE_VERIFIED'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8", errors="replace")
if "vc4kernel.kernel" not in text:
    raise SystemExit(f"{path}: verified VC4Kernel file contains no vc4kernel.kernel")
if "ssavc4." in text:
    raise SystemExit(f"{path}: verified VC4Kernel file unexpectedly contains ssavc4.*")
if "vc4.qpu." in text or '"vc4.module"' in text or "vc4.module" in text:
    raise SystemExit(f"{path}: verified VC4Kernel file unexpectedly contains scheduled vc4")
for dialect in ("func", "vector", "memref", "vc4value", "scf", "tensor", "linalg", "gpu", "tt", "ttg"):
    if re.search(rf'(?<![A-Za-z0-9_])"?{re.escape(dialect)}\.', text):
        raise SystemExit(f"{path}: verified VC4Kernel file contains producer dialect op {dialect}.*")
PY_VALIDATE_VERIFIED
}

validate_value_file() {
  local path="$1"
  require_file "$path"
  local ttir_paths=()
  mapfile -t ttir_paths < <(ttir_input_paths)
  python3 - "$path" "${ttir_paths[@]}" <<'PY_VALIDATE_VALUE'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8", errors="replace")
ttir_text = "\n".join(Path(arg).read_text(encoding="utf-8", errors="replace")
                      for arg in sys.argv[2:])
for required in ("func.func", "vc4value.kernel", "vc4value.program_id",
                 "vector.transfer_write"):
    if required not in text:
        raise SystemExit(f"{path}: lowered value file missing {required}")
if re.search(r'(?<![A-Za-z0-9_])"?tt\.load\b', ttir_text) and "vector.transfer_read" not in text:
    raise SystemExit(f"{path}: lowered value file missing vector.transfer_read for TTIR tt.load")
def has_masked_ttir_transfer(module_text):
    for line in module_text.splitlines():
        if not re.search(r'(?<![A-Za-z0-9_])"?tt\.(load|store)\b', line):
            continue
        operands = line.split(":", 1)[0]
        if operands.count(",") >= 2:
            return True
    return False

if has_masked_ttir_transfer(ttir_text) and "vector.create_mask" not in text:
    raise SystemExit(f"{path}: lowered value file missing vector.create_mask for masked TTIR transfer")
for dialect in ("tt", "ttg", "gpu", "nvgpu", "nvvm", "rocdl", "llvm",
                "vc4kernel", "ssavc4"):
    if re.search(rf'(?<![A-Za-z0-9_])"?{re.escape(dialect)}\.', text):
        raise SystemExit(f"{path}: lowered value file contains forbidden {dialect}.*")
if re.search(r'(?<![A-Za-z0-9_])"?vc4\.', text):
    raise SystemExit(f"{path}: lowered value file contains scheduled vc4.*")
PY_VALIDATE_VALUE
}

append_value_module_body() {
  local out_path="$1"
  local module_path="$2"
  python3 - "$out_path" "$module_path" <<'PY_APPEND_VALUE_MODULE'
from pathlib import Path
import sys

out_path = Path(sys.argv[1])
module_path = Path(sys.argv[2])
lines = module_path.read_text(encoding="utf-8").splitlines()
while lines and not lines[0].strip():
    lines.pop(0)
while lines and not lines[-1].strip():
    lines.pop()
if len(lines) < 2 or lines[0].strip() != "module {" or lines[-1].strip() != "}":
    raise SystemExit(f"{module_path}: expected a single top-level 'module {{ ... }}' value artifact")
body = lines[1:-1]
with out_path.open("a", encoding="utf-8") as out:
    for line in body:
        out.write(line)
        out.write("\n")
PY_APPEND_VALUE_MODULE
}

validate_core_ssavc4_file() {
  local path="$1"
  require_file "$path"
  python3 - "$path" <<'PY_VALIDATE_CORE'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8", errors="replace")
if "scf." in text:
    raise SystemExit(f"{path}: generated SSAVC4 core still contains scf.*")
if re.search(r"(?<![A-Za-z0-9_!])index(?![A-Za-z0-9_])", text):
    raise SystemExit(f"{path}: generated SSAVC4 core still contains index type")
for dialect in (
    "affine", "gpu", "triton", "tt", "ttg", "nvgpu", "iree", "stablehlo",
    "tosa", "linalg", "tensor", "memref", "vector", "func", "vc4value",
    "spirv", "nvvm", "rocdl",
):
    if re.search(rf'(?<![A-Za-z0-9_])"?{re.escape(dialect)}\.', text):
        raise SystemExit(f"{path}: generated SSAVC4 core contains producer dialect op {dialect}.*")
if "ssavc4.func" not in text:
    raise SystemExit(f"{path}: generated SSAVC4 core contains no ssavc4.func")
if "vc4kernel." in text:
    raise SystemExit(f"{path}: generated SSAVC4 core still contains vc4kernel.*")
if re.search(r'(?<![A-Za-z0-9_.])(?:"vc4\.module"|vc4\.module)(?![A-Za-z0-9_.])', text):
    raise SystemExit(f"{path}: generated SSAVC4 core already contains scheduled vc4.module")
if "vc4.qpu." in text:
    raise SystemExit(f"{path}: generated SSAVC4 core already contains scheduled vc4.qpu.*")
PY_VALIDATE_CORE
}

check_generated_bundle() {
  require_file "$GENERATED_DIR/manifest.json"
  require_file "$GENERATED_DIR/kernel_launch.c"
  require_file "$GENERATED_DIR/kernel_launch.h"
  require_file "$GENERATED_DIR/input.ttir.mlir"
  require_file "$GENERATED_DIR/input.vc4value.mlir"
  require_file "$GENERATED_DIR/input.vc4kernel.mlir"
  require_file "$GENERATED_DIR/input.ssavc4.mlir"
  require_file "$GENERATED_DIR/input.vc4.mlir"
  require_file "$GENERATED_DIR/lowered.value.mlir"
  require_file "$GENERATED_DIR/verified.vc4kernel.mlir"
  require_file "$GENERATED_DIR/lowered.ssavc4.mlir"
  require_file "$GENERATED_DIR/scheduled.vc4.mlir"
  manifest_kernel_records >/dev/null
}

run_vc4_codegen() {
  local vc4_opt vc4_codegen vc4_triton_opt lowered_dir lowered_tmp_dir input_ttir input_value value_verified value_cf verified_vc4kernel lowered_ssavc4 scheduled_vc4 stable_ttir stable_value stable_value_cf stable_vc4kernel stable_ssavc4 stable_vc4 stable_legacy_value stable_legacy_vc4kernel stable_legacy_ssavc4 stable_legacy_vc4 input_count input_path input_base per_input_ttir per_input_value
  vc4_opt="$(find_tool vc4-opt)"
  vc4_codegen="$(find_tool vc4-codegen)"
  vc4_triton_opt="$(find_tool vc4-triton-opt)"
  rm -rf "$GENERATED_DIR"
  mkdir -p "$GENERATED_DIR"
  lowered_dir="$AUTO_ROOT/lowered"
  mkdir -p "$lowered_dir"
  lowered_tmp_dir="$(mktemp -d "$lowered_dir/${TEST_NAME}.tmp.XXXXXX")"
  input_ttir="$lowered_tmp_dir/input.ttir.mlir"
  input_value="$lowered_tmp_dir/lowered.value.mlir"
  value_verified="$lowered_tmp_dir/lowered.value.verified.mlir"
  value_cf="$lowered_tmp_dir/lowered.value.cf.mlir"
  verified_vc4kernel="$lowered_tmp_dir/verified.vc4kernel.mlir"
  lowered_ssavc4="$lowered_tmp_dir/lowered.ssavc4.mlir"
  scheduled_vc4="$lowered_tmp_dir/scheduled.vc4.mlir"
  stable_ttir="$lowered_dir/${TEST_NAME}.input.ttir.mlir"
  stable_value="$lowered_dir/${TEST_NAME}.input.vc4value.mlir"
  stable_value_cf="$lowered_dir/${TEST_NAME}.input.vc4value.cf.mlir"
  stable_vc4kernel="$lowered_dir/${TEST_NAME}.input.vc4kernel.mlir"
  stable_ssavc4="$lowered_dir/${TEST_NAME}.input.ssavc4.mlir"
  stable_vc4="$lowered_dir/${TEST_NAME}.input.vc4.mlir"
  stable_legacy_value="$lowered_dir/${TEST_NAME}.lowered.value.mlir"
  stable_legacy_vc4kernel="$lowered_dir/${TEST_NAME}.verified.vc4kernel.mlir"
  stable_legacy_ssavc4="$lowered_dir/${TEST_NAME}.lowered.ssavc4.mlir"
  stable_legacy_vc4="$lowered_dir/${TEST_NAME}.scheduled.vc4.mlir"

  input_count="$(ttir_input_count)"
  [[ "$input_count" -gt 0 ]] || fail "no TTIR inputs found for $TEST_NAME"
  if [[ "$input_count" -eq 1 ]]; then
    : > "$input_value"
    input_path="$(ttir_input_paths | head -n 1)"
    cp "$input_path" "$input_ttir"
    log "importing TTIR input $(relpath "$input_path") to VC4 value IR at $(relpath "$input_value")"
    "$vc4_triton_opt" "$input_ttir" \
      --convert-triton-to-vc4-value \
      -o "$input_value"
    validate_value_file "$input_value"
  else
    mkdir -p "$lowered_tmp_dir/inputs" "$lowered_tmp_dir/imported"
    : > "$input_ttir"
    printf 'module {\n' > "$input_value"
    while IFS= read -r input_path; do
      [[ -z "$input_path" ]] && continue
      input_base="$(basename "$input_path" .ttir.mlir)"
      per_input_ttir="$lowered_tmp_dir/inputs/${input_base}.ttir.mlir"
      per_input_value="$lowered_tmp_dir/imported/${input_base}.value.mlir"
      cp "$input_path" "$per_input_ttir"
      {
        printf '// BEGIN_TTIR_INPUT %s\n' "$(relpath "$input_path")"
        cat "$per_input_ttir"
        printf '\n// END_TTIR_INPUT %s\n\n' "$(relpath "$input_path")"
      } >> "$input_ttir"
      log "importing TTIR input $(relpath "$input_path") to VC4 value IR at $(relpath "$per_input_value")"
      "$vc4_triton_opt" "$per_input_ttir" \
        --convert-triton-to-vc4-value \
        -o "$per_input_value"
      validate_value_file "$per_input_value"
      append_value_module_body "$input_value" "$per_input_value"
    done < <(ttir_input_paths)
    printf '}\n' >> "$input_value"
    validate_value_file "$input_value"
  fi

  log "verifying imported VC4 value IR at $(relpath "$value_verified")"
  "$vc4_opt" "$input_value" --vc4-verify-value-surface -o "$value_verified"
  validate_value_file "$value_verified"

  log "lowering imported VC4 value SCF to CF at $(relpath "$value_cf")"
  "$vc4_opt" "$value_verified" --convert-scf-to-cf -o "$value_cf"
  validate_value_file "$value_cf"

  log "converting imported VC4 value IR to VC4Kernel at $(relpath "$verified_vc4kernel")"
  "$vc4_opt" "$value_cf" \
    --vc4-verify-value-surface \
    --convert-vc4-value-to-vc4kernel \
    --verify-vc4kernel \
    -o "$verified_vc4kernel"
  validate_vc4kernel_verified_file "$verified_vc4kernel"

  log "lowering $(relpath "$verified_vc4kernel") to SSAVC4 at $(relpath "$lowered_ssavc4")"
  "$vc4_opt" "$verified_vc4kernel" --verify-vc4kernel --convert-vc4kernel-to-ssavc4 -o "$lowered_ssavc4"
  require_file "$lowered_ssavc4"
  validate_core_ssavc4_file "$lowered_ssavc4"

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

  cp "$input_ttir" "$stable_ttir"
  cp "$input_value" "$stable_value"
  cp "$value_cf" "$stable_value_cf"
  cp "$verified_vc4kernel" "$stable_vc4kernel"
  cp "$lowered_ssavc4" "$stable_ssavc4"
  cp "$scheduled_vc4" "$stable_vc4"
  cp "$input_value" "$stable_legacy_value"
  cp "$verified_vc4kernel" "$stable_legacy_vc4kernel"
  cp "$lowered_ssavc4" "$stable_legacy_ssavc4"
  cp "$scheduled_vc4" "$stable_legacy_vc4"
  cp "$input_ttir" "$GENERATED_DIR/input.ttir.mlir"
  cp "$input_value" "$GENERATED_DIR/input.vc4value.mlir"
  cp "$value_cf" "$GENERATED_DIR/input.vc4value.cf.mlir"
  cp "$verified_vc4kernel" "$GENERATED_DIR/input.vc4kernel.mlir"
  cp "$lowered_ssavc4" "$GENERATED_DIR/input.ssavc4.mlir"
  cp "$scheduled_vc4" "$GENERATED_DIR/input.vc4.mlir"
  cp "$input_value" "$GENERATED_DIR/lowered.value.mlir"
  cp "$value_cf" "$GENERATED_DIR/lowered.value.cf.mlir"
  cp "$verified_vc4kernel" "$GENERATED_DIR/verified.vc4kernel.mlir"
  cp "$lowered_ssavc4" "$GENERATED_DIR/lowered.ssavc4.mlir"
  cp "$scheduled_vc4" "$GENERATED_DIR/scheduled.vc4.mlir"
  if [[ "$input_count" -gt 1 ]]; then
    mkdir -p "$GENERATED_DIR/inputs"
    while IFS= read -r input_path; do
      [[ -z "$input_path" ]] && continue
      cp "$input_path" "$GENERATED_DIR/inputs/$(basename "$input_path")"
    done < <(ttir_input_paths)
  fi
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
  local -a records
  local qasm_rel code_symbol public qasm_path out_c out_h out record
  records=()
  while IFS= read -r record; do
    records+=("$record")
  done < <(manifest_kernel_records)
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
    fail "TTIR candidate fixture $TEST_NAME needs $(relpath "$candidate") using kernel_launch.h and vc4Malloc/vc4Memcpy APIs; reference harness fallback is disabled"
  fi

  log "WARNING: VC4_ALLOW_REFERENCE_HARNESS_FALLBACK=1; using legacy harness fallback for $TEST_NAME. This is debug-only and FORBIDDEN for normal TTIR candidate verification."

  local named="$CANDIDATE_DIR/${TEST_NAME}_harness.c"
  if [[ -f "$named" ]]; then printf '%s\n' "$named"; return 0; fi
  local -a matches
  if [[ -d "$REFERENCE_DIR" ]]; then
    matches=()
    while IFS= read -r match; do
      matches+=("$match")
    done < <(find "$REFERENCE_DIR" -maxdepth 1 -type f -name '3-test-*.c' -print | sort)
    if [[ "${#matches[@]}" -eq 1 ]]; then printf '%s\n' "${matches[0]}"; return 0; fi
    matches=()
    while IFS= read -r match; do
      matches+=("$match")
    done < <(find "$REFERENCE_DIR" -maxdepth 1 -type f -name "${TEST_NAME}_harness.c" -print | sort)
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
    "/* Generated by run_ttir_candidate_codegen_test.sh; do not commit. */",
    "#ifndef VC4_CASE_CONFIG_H",
    "#define VC4_CASE_CONFIG_H",
    "",
    "#define VC4_CASE_SAW_CPP_TTIR_IMPORTER 1",
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
# Generated TTIR program-bundle smoke workdir

Generated by compiler/test/CodeGen/Triton/Support/run_ttir_candidate_codegen_test.sh.
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

Generated by compiler/test/CodeGen/Triton/Support/run_ttir_candidate_codegen_test.sh.
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
    print(f"[ttir-candidate] WARNING: Pi power cycle command timed out after {timeout}s", file=sys.stderr)
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
  local attempt_timeout="${VC4_HW_ATTEMPT_TIMEOUT_SEC:-60}"
  local retry_reason=""
  if ! [[ "$attempt_timeout" =~ ^[0-9]+$ ]] || [[ "$attempt_timeout" -lt 1 ]]; then
    fail "VC4_HW_ATTEMPT_TIMEOUT_SEC must be a positive integer, got: $attempt_timeout"
  fi
  while true; do
    if [[ "$attempt" -eq 1 ]]; then
      vc4_candidate_power_cycle_if_needed "power cycling Pi before candidate run"
    else
      log "retrying candidate run after transient pre-boot failure (attempt ${attempt}/${max_attempts}; matched: ${retry_reason})"
      vc4_candidate_power_cycle_if_needed "power cycling Pi before retry"
    fi

    rm -f "$attempt_log"
    set +e
    vc4_candidate_run_sh_with_timeout "$WORK_DIR" "$attempt_timeout" 2>&1 | tee "$attempt_log"
    run_rc=${PIPESTATUS[0]}
    set -e

    if [[ "$run_rc" -eq 0 ]]; then
      python3 "$CHECKER" "$EXPECTED_JSON" "$attempt_log"
      return 0
    fi

    if [[ "$run_rc" -ne 0 ]] && retry_reason="$(vc4_candidate_transient_preboot_failure_reason "$attempt_log")" && [[ "$attempt" -lt "$max_attempts" ]]; then
      log "transient pre-boot failure matched in attempt log: ${retry_reason}"
      printf '[ttir-candidate] transient pre-boot failure matched: %s\n' "$retry_reason" >> "$attempt_log"
      attempt=$((attempt + 1))
      continue
    fi

    break
  done

  return "$run_rc"
}

vc4_candidate_run_sh_with_timeout() {
  local work_dir="$1"
  local timeout_sec="$2"
  python3 - "$work_dir" "$timeout_sec" <<'PY_RUN_TIMEOUT'
import os
import signal
import subprocess
import sys

work_dir, timeout_s = sys.argv[1:]
timeout = float(timeout_s)
process = subprocess.Popen(["bash", "run.sh"], cwd=work_dir, start_new_session=True)
try:
    sys.exit(process.wait(timeout=timeout))
except subprocess.TimeoutExpired:
    print(
        f"[ttir-candidate] ERROR: bash run.sh timed out after {int(timeout)} seconds; terminating hardware process group",
        file=sys.stderr,
    )
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        print(
            "[ttir-candidate] ERROR: bash run.sh did not exit after SIGTERM; sending SIGKILL",
            file=sys.stderr,
        )
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
    sys.exit(124)
PY_RUN_TIMEOUT
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

  cat > "$tmp/expected.json" <<'EOF'
{"name":"runner_self_test","status":"PASS","required":{"mismatches":0,"count":1},"float_max":{"max_abs_diff":0.001}}
EOF
  printf '%s\n' \
    'noise before result' \
    'VC4_TEST_RESULT name=runner_self_test status=PASS mismatches=0 count=1 max_abs_diff=0.0005' > "$tmp/pass.log"
  python3 "$CHECKER" "$tmp/expected.json" "$tmp/pass.log"
  log "self-test checker pass: synthetic PASS log"

  printf '%s\n' 'VC4_TEST_RESULT name=runner_self_test status=FAIL mismatches=0 count=1 max_abs_diff=0.0005' > "$tmp/fail.log"
  if python3 "$CHECKER" "$tmp/expected.json" "$tmp/fail.log" >/dev/null 2>&1; then
    fail "self-test expected checker failure for synthetic FAIL log"
  fi
  log "self-test checker fail: synthetic FAIL log"

  printf '%s\n' 'noise without result' > "$tmp/missing-result.log"
  if python3 "$CHECKER" "$tmp/expected.json" "$tmp/missing-result.log" >/dev/null 2>&1; then
    fail "self-test expected checker failure for missing VC4_TEST_RESULT log"
  fi
  log "self-test checker fail: missing result log"

  mkdir -p "$tmp/timeout-workdir"
  cat > "$tmp/timeout-workdir/run.sh" <<'EOF'
#!/usr/bin/env bash
sleep 5
EOF
  chmod +x "$tmp/timeout-workdir/run.sh"
  if vc4_candidate_run_sh_with_timeout "$tmp/timeout-workdir" 1 >/dev/null 2>&1; then
    fail "self-test expected timeout wrapper to fail"
  fi
  log "self-test timeout wrapper returned nonzero"

  echo "run_ttir_candidate_codegen_test.sh self-test PASS"
}

case "$PHASE" in
  clean) rm -rf "$GENERATED_DIR" "$HARDWARE_ROOT"; log "removed generated candidate state for $TEST_NAME" ;;
  generate) check_fixture; run_vc4_codegen ;;
  assemble) assemble_candidate ;;
  build) build_candidate ;;
  run|all) run_candidate ;;
  workdir) prepare_workdir; printf '%s\n' "$WORK_DIR" ;;
  generated-dir) printf '%s\n' "$GENERATED_DIR" ;;
  self-test) vc4_candidate_self_test ;;
  *) usage; exit 2 ;;
esac
