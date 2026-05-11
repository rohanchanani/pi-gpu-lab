#!/usr/bin/env bash
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || cd ../../../../../../.. && pwd)"
export VC4_CODEGEN_STATE_ROOT="${VC4_CODEGEN_STATE_ROOT:-.vc4_auto/codegen_m2}"
bash "$ROOT/compiler/test/CodeGen/VC4/Support/run_candidate_codegen_test.sh" vector_store_smoke all
