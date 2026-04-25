#!/usr/bin/env bash
# Run one self-contained VC4 hardware golden test directory.
#
# Usage:
#   compiler/test/CodeGen/VC4/Support/run_hardware_test.sh \
#     compiler/test/CodeGen/VC4/Hardware/Run/<test-name>
#
# The test directory must contain:
#   run.sh
#   expected.json
#
# This wrapper power-cycles the Pi by default, runs `bash run.sh` in the test
# directory, captures run.log, and checks the final VC4_TEST_RESULT line.

set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "usage: $0 <test-dir> [expected-json] [log-path]" >&2
  exit 2
fi

TEST_DIR=$1
if [[ ! -d "$TEST_DIR" ]]; then
  echo "error: test directory does not exist: $TEST_DIR" >&2
  exit 1
fi

TEST_DIR=$(cd "$TEST_DIR" && pwd)
EXPECTED=${2:-"$TEST_DIR/expected.json"}
LOG_PATH=${3:-"$TEST_DIR/run.log"}

if [[ ! -f "$TEST_DIR/run.sh" ]]; then
  echo "error: missing run.sh in $TEST_DIR" >&2
  exit 1
fi
if [[ ! -f "$EXPECTED" ]]; then
  echo "error: missing expected JSON: $EXPECTED" >&2
  exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CHECKER="$SCRIPT_DIR/check_vc4_test_result.py"

POWER_CMD=${VC4_PI_POWER_CYCLE_CMD:-"uhubctl -l 0-1 -a cycle"}
POWER_SLEEP=${VC4_PI_POWER_CYCLE_SLEEP_SEC:-4}

if [[ "${VC4_SKIP_POWER_CYCLE:-0}" != "1" ]]; then
  echo "[vc4-hw] power cycling Pi: $POWER_CMD"
  # shellcheck disable=SC2086
  $POWER_CMD
  echo "[vc4-hw] sleeping ${POWER_SLEEP}s after power cycle"
  sleep "$POWER_SLEEP"
else
  echo "[vc4-hw] skipping Pi power cycle because VC4_SKIP_POWER_CYCLE=1"
fi

echo "[vc4-hw] running hardware test in $TEST_DIR"
rm -f "$LOG_PATH"
set +e
(
  cd "$TEST_DIR"
  bash run.sh
) 2>&1 | tee "$LOG_PATH"
RUN_RC=${PIPESTATUS[0]}
set -e

if [[ $RUN_RC -ne 0 ]]; then
  echo "[vc4-hw] run.sh exited with status $RUN_RC; checking log for diagnostics anyway" >&2
fi

python3 "$CHECKER" "$EXPECTED" "$LOG_PATH"
CHECK_RC=$?

if [[ $RUN_RC -ne 0 ]]; then
  exit "$RUN_RC"
fi
exit "$CHECK_RC"
