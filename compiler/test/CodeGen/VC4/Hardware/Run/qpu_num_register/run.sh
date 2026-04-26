#!/bin/bash
set -euo pipefail

test_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$test_dir/../../../../../../.." && pwd)"

exec "$repo_root/compiler/test/CodeGen/VC4/Support/run_hardware_test.sh" \
  "$test_dir" \
  reference
