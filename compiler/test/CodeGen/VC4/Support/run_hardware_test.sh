#!/usr/bin/env bash
# Run one VC4 hardware golden-test side and check VC4_TEST_RESULT output.

set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
usage:
  run_hardware_test.sh --self-test
  run_hardware_test.sh <test-root> [reference|candidate] [expected-json] [log-path]

environment:
  VC4_PI_POWER_CYCLE_CMD        default: uhubctl -l 0-1 -a cycle
  VC4_PI_POWER_CYCLE_SLEEP_SEC  default: 1
  VC4_SKIP_POWER_CYCLE=1        skip power cycle, for self-test/manual debug only
  VC4_RUN_SH_MAX_ATTEMPTS       default: 3
  VC4_HW_ATTEMPT_TIMEOUT_SEC   default: 60; timeout for one side run.sh attempt
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKER="$SCRIPT_DIR/check_vc4_test_result.py"

fail() {
  printf '[vc4-hw] ERROR: %s\n' "$*" >&2
  exit 1
}

vc4_hw_transient_preboot_failure_reason() {
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

vc4_hw_is_transient_preboot_failure() {
  vc4_hw_transient_preboot_failure_reason "$1" >/dev/null
}

run_power_cycle() {
  if [[ "${VC4_SKIP_POWER_CYCLE:-0}" == "1" ]]; then
    printf '[vc4-hw] skipping power cycle\n'
    return 0
  fi
  local cmd="${VC4_PI_POWER_CYCLE_CMD:-uhubctl -l 0-1 -a cycle}"
  printf '[vc4-hw] power cycle: %s\n' "$cmd"
  bash -c "$cmd"
  sleep "${VC4_PI_POWER_CYCLE_SLEEP_SEC:-1}"
}


run_side_with_timeout() {
  local side_dir="$1"
  local timeout_sec="$2"
  local test_root="$3"
  local side="$4"
  local input_mlir="$5"
  local expected_json="$6"

  python3 - "$side_dir" "$timeout_sec" "$test_root" "$side" "$input_mlir" "$expected_json" <<'PY_RUN_TIMEOUT'
import os
import signal
import subprocess
import sys

side_dir, timeout_s, test_root, side, input_mlir, expected_json = sys.argv[1:]
timeout = float(timeout_s)
env = os.environ.copy()
env["VC4_TEST_ROOT"] = test_root
env["VC4_TEST_SIDE"] = side
env["VC4_TEST_INPUT_MLIR"] = input_mlir
env["VC4_TEST_EXPECTED_JSON"] = expected_json

proc = subprocess.Popen(["bash", "run.sh"], cwd=side_dir, env=env, start_new_session=True)
try:
    sys.exit(proc.wait(timeout=timeout))
except subprocess.TimeoutExpired:
    print(f"[vc4-hw] ERROR: run.sh timed out after {int(timeout)} seconds; terminating hardware process group", file=sys.stderr)
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        print("[vc4-hw] ERROR: run.sh did not exit after SIGTERM; sending SIGKILL", file=sys.stderr)
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
    sys.exit(124)
PY_RUN_TIMEOUT
}

run_one() {
  local test_root="$1"
  local side="${2:-reference}"
  local expected_json="${3:-$test_root/expected.json}"
  local log_path="${4:-$test_root/$side/run.log}"
  local side_dir="$test_root/$side"
  local max_attempts="${VC4_RUN_SH_MAX_ATTEMPTS:-3}"
  local attempt_timeout="${VC4_HW_ATTEMPT_TIMEOUT_SEC:-60}"
  local input_mlir="$test_root/input.mlir"

  [[ "$side" == "reference" || "$side" == "candidate" ]] || fail "side must be reference or candidate"
  [[ -f "$test_root/input.mlir" ]] || fail "missing input.mlir under $test_root"
  [[ -f "$expected_json" ]] || fail "missing expected json: $expected_json"
  [[ -f "$side_dir/run.sh" ]] || fail "missing side runner: $side_dir/run.sh"
  [[ "$max_attempts" =~ ^[0-9]+$ && "$max_attempts" -ge 1 ]] || max_attempts=1
  [[ "$attempt_timeout" =~ ^[0-9]+$ && "$attempt_timeout" -ge 1 ]] || attempt_timeout=60

  mkdir -p "$(dirname "$log_path")"
  run_power_cycle

  local attempt status
  for ((attempt = 1; attempt <= max_attempts; ++attempt)); do
    printf '[vc4-hw] running %s side attempt %d/%d\n' "$side" "$attempt" "$max_attempts"
    set +e
    run_side_with_timeout "$side_dir" "$attempt_timeout" "$test_root" "$side" "$input_mlir" "$expected_json" 2>&1 | tee "$log_path"
    status=${PIPESTATUS[0]}
    set -e

    if [[ "$status" -eq 0 ]]; then
      python3 "$CHECKER" "$expected_json" "$log_path"
      return 0
    fi

    if [[ "$status" -eq 124 && "$attempt" -lt "$max_attempts" ]]; then
      printf '[vc4-hw] run.sh attempt timed out after %s seconds; power-cycling and retrying\n' "$attempt_timeout" >&2
      run_power_cycle
      continue
    fi

    local retry_reason
    if retry_reason="$(vc4_hw_transient_preboot_failure_reason "$log_path")" && [[ "$attempt" -lt "$max_attempts" ]]; then
      printf '[vc4-hw] retrying %s side after transient pre-boot failure (attempt %d/%d; matched: %s)\n' "$side" "$((attempt + 1))" "$max_attempts" "$retry_reason" >&2
      printf '[vc4-hw] transient pre-boot failure matched: %s\n' "$retry_reason" >> "$log_path"
      run_power_cycle
      continue
    fi

    return "$status"
  done
}

vc4_hw_self_test_assert_true() {
  local name="$1"
  local log_path="$2"
  local reason
  if ! reason="$(vc4_hw_transient_preboot_failure_reason "$log_path")"; then
    fail "self-test expected transient pre-boot true: $name"
  fi
  printf '[vc4-hw] self-test true: %s (%s)\n' "$name" "$reason"
}

vc4_hw_self_test_assert_false() {
  local name="$1"
  local log_path="$2"
  if vc4_hw_is_transient_preboot_failure "$log_path"; then
    fail "self-test expected transient pre-boot false: $name"
  fi
  printf '[vc4-hw] self-test false: %s\n' "$name"
}

self_test() {
  local tmp marker
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' RETURN

  printf '%s\n' 'tty-USB read() returned 0 bytes.  r/pi not responding [reboot it?]' > "$tmp/tty-read-zero.log"
  vc4_hw_self_test_assert_true "tty-read-zero log" "$tmp/tty-read-zero.log"

  {
    printf '%s\n' 'GET_CODE op mismatch: expected 55556666, got 00000000'
    printf '%s\n' 'simple-boot.c:ck_eq32:30:PANIC:pi-boot failed'
  } > "$tmp/get-code-panic.log"
  vc4_hw_self_test_assert_true "GET_CODE mismatch plus pi-boot panic log" "$tmp/get-code-panic.log"

  marker='VC4_KERNEL_LAUNCH_ERROR'
  printf '%s\n' "${marker} reason=wave_timeout" > "$tmp/launch-error.log"
  vc4_hw_self_test_assert_false "kernel launch error log" "$tmp/launch-error.log"

  marker='VC4_TEST_''RESULT'
  printf '%s\n' "${marker} status=FAIL" > "$tmp/test-result-fail.log"
  vc4_hw_self_test_assert_false "test result fail log" "$tmp/test-result-fail.log"

  marker='VC4_RUNTIME_''LAYOUT'
  {
    printf '%s\n' "${marker} heap_base=0x1000"
    printf '%s\n' 'PANIC:pi-boot failed'
  } > "$tmp/runtime-layout-crash.log"
  vc4_hw_self_test_assert_false "runtime layout then crash log" "$tmp/runtime-layout-crash.log"

  mkdir -p "$tmp/reference"
  cat > "$tmp/input.mlir" <<'EOF'
// self-test placeholder
EOF
  cat > "$tmp/expected.json" <<'EOF'
{"name":"runner_self_test","status":"PASS","required":{"mismatches":0,"count":1},"float_max":{"max_abs_diff":0.001}}
EOF
  cat > "$tmp/reference/run.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
echo "noise before result"
echo "VC4_TEST_RESULT name=runner_self_test status=PASS mismatches=0 count=1 max_abs_diff=0.0005"
EOF
  VC4_SKIP_POWER_CYCLE=1 run_one "$tmp" reference
  echo "run_hardware_test.sh self-test PASS"
}

if [[ $# -eq 1 && "$1" == "--self-test" ]]; then
  self_test
  exit 0
fi

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

run_one "$@"
