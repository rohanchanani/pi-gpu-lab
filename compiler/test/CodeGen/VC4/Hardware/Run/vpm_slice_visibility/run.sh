#!/bin/bash
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
repo_root=$(cd "$here/../../../../../../.." && pwd)

"$repo_root/compiler/test/CodeGen/VC4/Support/run_hardware_test.sh" \
  "$here" \
  reference
