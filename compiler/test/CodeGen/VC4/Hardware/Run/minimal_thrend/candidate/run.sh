#!/usr/bin/env bash
# Candidate-side hardware run hook for minimal_thrend.
# The support runner generates, assembles, and prepares the candidate workdir;
# this script only delegates and does not mutate reference/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$REPO_ROOT" ]]; then
  REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../../../../.." && pwd)"
fi

exec bash "$REPO_ROOT/compiler/test/CodeGen/VC4/Support/run_candidate_codegen_test.sh" minimal_thrend run
