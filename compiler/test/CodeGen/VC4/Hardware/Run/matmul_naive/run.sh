#!/bin/bash
set -euo pipefail

compiler/test/CodeGen/VC4/Support/run_hardware_test.sh \
  compiler/test/CodeGen/VC4/Hardware/Run/matmul_naive \
  reference
