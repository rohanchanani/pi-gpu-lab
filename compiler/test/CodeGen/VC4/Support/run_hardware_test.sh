#!/usr/bin/env bash
# Run one VC4 hardware golden test side.
#
# Normal usage:
#   compiler/test/CodeGen/VC4/Support/run_hardware_test.sh \
#     compiler/test/CodeGen/VC4/Hardware/Run/<test-name> \
#     reference
#
# Future generated-code usage:
#   compiler/test/CodeGen/VC4/Support/run_hardware_test.sh \
#     compiler/test/CodeGen/VC4/Hardware/Run/<test-name> \
#     candidate
#
# Test root contract:
#   input.mlir
#   expected.json
#   reference/run.sh       # for reference-side runs
#   candidate/run.sh       # for future candidate-side runs
#
# This wrapper power-cycles the Pi by default, runs `bash run.sh` inside the
# selected side directory, captures <side>/run.log, and checks the final
# VC4_TEST_RESULT line against the root-level expected.json.

set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
usage:
  run_hardware_test.sh --self-test
  run_hardware_test.sh <test-root> [reference|candidate] [expected-json] [log-path]

environment:
  VC4_PI_POWER_CYCLE_CMD        default: uhubctl -l 0-1 -a cycle
  VC4_PI_POWER_CYCLE_SLEEP_SEC  default: 4
  VC4_SKIP_POWER_CYCLE=1        skip power cycle, for self-test/manual debug only
USAGE
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CHECKER="$SCRIPT_DIR/check_vc4_test_result.py"

run_one() {
  local test_root=$1
  local side=${2:-reference}
  local expected_arg=${3:-}
  local log_arg=${4:-}

  if [[ "$side" != "reference" && "$side" != "candidate" ]]; then
    echo "error: side must be 'reference' or 'candidate', got: $side" >&2
    return 2
  fi

  if [[ ! -d "$test_root" ]]; then
    echo "error: test root does not exist: $test_root" >&2
    return 1
  fi

  test_root=$(cd "$test_root" && pwd)
  local input_mlir="$test_root/input.mlir"
  local expected=${expected_arg:-"$test_root/expected.json"}
  local side_dir="$test_root/$side"
  local log_path=${log_arg:-"$side_dir/run.log"}

  if [[ ! -f "$input_mlir" ]]; then
    echo "error: missing input MLIR: $input_mlir" >&2
    return 1
  fi
  if [[ ! -f "$expected" ]]; then
    echo "error: missing expected JSON: $expected" >&2
    return 1
  fi
  if [[ ! -d "$side_dir" ]]; then
    echo "error: missing side directory: $side_dir" >&2
    return 1
  fi
  if [[ ! -f "$side_dir/run.sh" ]]; then
    echo "error: missing run.sh in $side_dir" >&2
    return 1
  fi

  mkdir -p "$(dirname "$log_path")"

  local power_cmd=${VC4_PI_POWER_CYCLE_CMD:-"uhubctl -l 0-1 -a cycle"}
  local power_sleep=${VC4_PI_POWER_CYCLE_SLEEP_SEC:-4}

  if [[ "${VC4_SKIP_POWER_CYCLE:-0}" != "1" ]]; then
    echo "[vc4-hw] power cycling Pi: $power_cmd"
    # shellcheck disable=SC2086
    $power_cmd
    echo "[vc4-hw] sleeping ${power_sleep}s after power cycle"
    sleep "$power_sleep"
  else
    echo "[vc4-hw] skipping Pi power cycle because VC4_SKIP_POWER_CYCLE=1"
  fi

  echo "[vc4-hw] test root: $test_root"
  echo "[vc4-hw] side: $side"
  echo "[vc4-hw] input MLIR: $input_mlir"
  echo "[vc4-hw] expected JSON: $expected"
  echo "[vc4-hw] log: $log_path"
  echo "[vc4-hw] running hardware test in $side_dir"

  rm -f "$log_path"
  set +e
  (
    cd "$side_dir"
    export VC4_TEST_ROOT="$test_root"
    export VC4_TEST_SIDE="$side"
    export VC4_TEST_INPUT_MLIR="$input_mlir"
    export VC4_TEST_EXPECTED_JSON="$expected"
    bash run.sh
  ) 2>&1 | tee "$log_path"
  local run_rc=${PIPESTATUS[0]}
  set -e

  if [[ $run_rc -ne 0 ]]; then
    echo "[vc4-hw] run.sh exited with status $run_rc; checking log for diagnostics anyway" >&2
  fi

  set +e
  python3 "$CHECKER" "$expected" "$log_path"
  local check_rc=$?
  set -e

  if [[ $check_rc -ne 0 ]]; then
    return "$check_rc"
  fi
  if [[ $run_rc -ne 0 ]]; then
    return "$run_rc"
  fi

  echo "[vc4-hw] PASS: $test_root ($side)"
  return 0
}

self_test() {
  local tmp
  tmp=$(mktemp -d "${TMPDIR:-/tmp}/vc4_hw_runner_selftest.XXXXXX")
  trap 'rm -rf "$tmp"' RETURN

  mkdir -p "$tmp/reference"
  cat >"$tmp/input.mlir" <<'MLIR'
vc4.module @runner_self_test {
}
MLIR

  cat >"$tmp/expected.json" <<'JSON'
{
  "name": "runner_self_test",
  "status": "PASS",
  "required": {
    "mismatches": 0,
    "count": 1
  },
  "float_max": {
    "max_abs_diff": 0.001
  }
}
JSON

  cat >"$tmp/reference/run.sh" <<'RUN'
#!/usr/bin/env bash
set -euo pipefail
echo "noise before result"
echo "VC4_TEST_RESULT name=runner_self_test status=PASS mismatches=0 count=1 max_abs_diff=0.0005"
RUN
  chmod +x "$tmp/reference/run.sh"

  VC4_SKIP_POWER_CYCLE=1 run_one "$tmp" reference
  echo "run_hardware_test.sh self-test PASS"
}

if [[ $# -eq 1 && "$1" == "--self-test" ]]; then
  self_test
  exit $?
fi

if [[ $# -lt 1 || $# -gt 4 ]]; then
  usage
  exit 2
fi

run_one "$@"
