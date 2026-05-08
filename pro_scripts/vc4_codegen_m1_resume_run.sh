#!/usr/bin/env bash
# Resumable VC4 codegen Milestone 1 runner.
#
# Drop this file into pro_scripts/ and run from anywhere inside the repo:
#   bash pro_scripts/vc4_codegen_m1_resume_run.sh
#
# It uses the typed verifier as the resume oracle:
#   1. Verify each slice first.
#   2. If verification passes, skip autorun for that slice.
#   3. If verification fails, run autorun for exactly that slice.
#   4. Verify the slice again after autorun.
#   5. Stop immediately on any autorun or post-verify failure.
#
# This deliberately allows redundant verifier runs, but avoids redundant GPT
# calls and repo edits for slices that are already good.

set -euo pipefail

DEFAULT_CHAT_URL="https://chatgpt.com/c/69fd396d-6dcc-83e8-9be0-343809bb4005"
DEFAULT_CHAT_TIMEOUT_SEC=5400
DEFAULT_GATE_TIMEOUT_SEC=7200
DEFAULT_GPT_MODE="current_tab"

usage() {
  cat >&2 <<'USAGE'
usage: vc4_codegen_m1_resume_run.sh [REPO_ROOT] [options]

Options:
  --from SLICE_OR_INDEX       Start at this slice/index/alias. Examples: m0, 0, m1-00-preflight.
  --to SLICE_OR_INDEX         Stop after this slice/index/alias. Examples: m11, 11, m1-11-simple-memory-output.
  --only SLICE_OR_INDEX       Run/check only one slice.
  --chat-url URL              GPT_WEB_CURRENT_CHAT_URL to export before autorun.
  --gpt-mode MODE             Passed to autorun --gpt-mode. Default: current_tab.
  --chat-timeout-sec N        Passed to autorun. Default: 5400.
  --gate-timeout-sec N        Passed to autorun and verifier. Default: 7200.
  --python PYTHON             Python executable. Default: python3.
  --allow-dirty               Pass --allow-dirty to autorun when a slice needs work.
  --verify-only               Never call autorun; stop at first verifier failure.
  --no-hardware-verify        Pass --no-hardware to typed verifier. Autorun gates are unchanged.
  --dry-run                   Print actions without executing verifier or autorun.
  -h, --help                  Show this help.

Environment:
  GPT_WEB_CURRENT_CHAT_URL    Used if --chat-url is not provided; otherwise defaults to the active URL
                              embedded in this script.
  VC4_M1_RESUME_EXTRA_AUTORUN_ARGS
                              Optional extra arguments appended to autorun invocations.
USAGE
}

log() {
  printf '[vc4-m1-resume] %s\n' "$*" | tee -a "$LOG"
}

fail() {
  log "ERROR: $*"
  exit 1
}

# Resolve repo root from first positional arg or current directory.
REPO_ARG=""
if [[ $# -gt 0 && "${1:-}" != --* && "${1:-}" != "-h" ]]; then
  REPO_ARG="$1"
  shift
fi

FROM_SLICE="m1-00-preflight"
TO_SLICE="m1-11-simple-memory-output"
ONLY_SLICE=""
CHAT_URL="${GPT_WEB_CURRENT_CHAT_URL:-$DEFAULT_CHAT_URL}"
GPT_MODE="$DEFAULT_GPT_MODE"
CHAT_TIMEOUT_SEC="$DEFAULT_CHAT_TIMEOUT_SEC"
GATE_TIMEOUT_SEC="$DEFAULT_GATE_TIMEOUT_SEC"
PYTHON="python3"
ALLOW_DIRTY=0
VERIFY_ONLY=0
NO_HARDWARE_VERIFY=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --from)
      [[ $# -ge 2 ]] || fail "--from requires an argument"
      FROM_SLICE="$2"; shift 2 ;;
    --to)
      [[ $# -ge 2 ]] || fail "--to requires an argument"
      TO_SLICE="$2"; shift 2 ;;
    --only)
      [[ $# -ge 2 ]] || fail "--only requires an argument"
      ONLY_SLICE="$2"; shift 2 ;;
    --chat-url)
      [[ $# -ge 2 ]] || fail "--chat-url requires an argument"
      CHAT_URL="$2"; shift 2 ;;
    --gpt-mode)
      [[ $# -ge 2 ]] || fail "--gpt-mode requires an argument"
      GPT_MODE="$2"; shift 2 ;;
    --chat-timeout-sec)
      [[ $# -ge 2 ]] || fail "--chat-timeout-sec requires an argument"
      CHAT_TIMEOUT_SEC="$2"; shift 2 ;;
    --gate-timeout-sec)
      [[ $# -ge 2 ]] || fail "--gate-timeout-sec requires an argument"
      GATE_TIMEOUT_SEC="$2"; shift 2 ;;
    --python)
      [[ $# -ge 2 ]] || fail "--python requires an argument"
      PYTHON="$2"; shift 2 ;;
    --allow-dirty)
      ALLOW_DIRTY=1; shift ;;
    --verify-only)
      VERIFY_ONLY=1; shift ;;
    --no-hardware-verify)
      NO_HARDWARE_VERIFY=1; shift ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      fail "unknown argument: $1" ;;
  esac
done

if [[ -n "$REPO_ARG" ]]; then
  REPO_ROOT="$(cd "$REPO_ARG" && pwd)"
else
  REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
fi
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"

AUTORUN="$REPO_ROOT/pro_scripts/vc4_codegen_m1_autorun.py"
VERIFIER="$REPO_ROOT/pro_scripts/vc4_codegen_m1_verifier.py"
SPEC="$REPO_ROOT/pro_scripts/vc4_codegen_m1_verifications.json"
WORKLIST="$REPO_ROOT/pro_scripts/vc4_codegen_m1_worklist.json"
STATE_ROOT="$REPO_ROOT/.vc4_auto/codegen_m1"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="$STATE_ROOT/resume"
REPORT_DIR="$STATE_ROOT/verifier/resume_reports/$RUN_ID"
mkdir -p "$LOG_DIR" "$REPORT_DIR"
LOG="$LOG_DIR/run_${RUN_ID}.log"
: > "$LOG"

SLICES=(
  "m1-00-preflight"
  "m1-01-artifact-tool-skeleton"
  "m1-02-emit-contract-local-tests"
  "m1-03-minimal-thrend-qasm"
  "m1-04-qpu-bundle-basic"
  "m1-05-qpu-ldi-sema"
  "m1-06-qpu-branch-delay-slots"
  "m1-07-launch-abi-model-header"
  "m1-08-launcher-c-uniform-packing"
  "m1-09-candidate-bundle-runner"
  "m1-10-hardware-smoke-minimal"
  "m1-11-simple-memory-output"
)

normalize_slice() {
  local raw="$1"
  local value="${raw#m}"
  value="${value#M}"
  if [[ "$raw" =~ ^m1-[0-9][0-9]- ]]; then
    printf '%s\n' "$raw"
    return 0
  fi
  if [[ "$raw" =~ ^m[0-9]+$ || "$raw" =~ ^M[0-9]+$ || "$raw" =~ ^[0-9]+$ ]]; then
    local n="$value"
    if [[ "$raw" =~ ^[0-9]+$ ]]; then n="$raw"; fi
    case "$n" in
      0|00) printf '%s\n' "m1-00-preflight" ;;
      1|01) printf '%s\n' "m1-01-artifact-tool-skeleton" ;;
      2|02) printf '%s\n' "m1-02-emit-contract-local-tests" ;;
      3|03) printf '%s\n' "m1-03-minimal-thrend-qasm" ;;
      4|04) printf '%s\n' "m1-04-qpu-bundle-basic" ;;
      5|05) printf '%s\n' "m1-05-qpu-ldi-sema" ;;
      6|06) printf '%s\n' "m1-06-qpu-branch-delay-slots" ;;
      7|07) printf '%s\n' "m1-07-launch-abi-model-header" ;;
      8|08) printf '%s\n' "m1-08-launcher-c-uniform-packing" ;;
      9|09) printf '%s\n' "m1-09-candidate-bundle-runner" ;;
      10) printf '%s\n' "m1-10-hardware-smoke-minimal" ;;
      11) printf '%s\n' "m1-11-simple-memory-output" ;;
      *) return 1 ;;
    esac
    return 0
  fi
  return 1
}

slice_index() {
  local needle="$1"
  local i
  for i in "${!SLICES[@]}"; do
    if [[ "${SLICES[$i]}" == "$needle" ]]; then
      printf '%s\n' "$i"
      return 0
    fi
  done
  return 1
}

FROM_SLICE="$(normalize_slice "$FROM_SLICE")" || fail "unknown --from slice: $FROM_SLICE"
TO_SLICE="$(normalize_slice "$TO_SLICE")" || fail "unknown --to slice: $TO_SLICE"
if [[ -n "$ONLY_SLICE" ]]; then
  ONLY_SLICE="$(normalize_slice "$ONLY_SLICE")" || fail "unknown --only slice: $ONLY_SLICE"
  FROM_SLICE="$ONLY_SLICE"
  TO_SLICE="$ONLY_SLICE"
fi
FROM_IDX="$(slice_index "$FROM_SLICE")" || fail "internal error: from slice not indexed: $FROM_SLICE"
TO_IDX="$(slice_index "$TO_SLICE")" || fail "internal error: to slice not indexed: $TO_SLICE"
if (( FROM_IDX > TO_IDX )); then
  fail "--from comes after --to: $FROM_SLICE > $TO_SLICE"
fi

require_file() {
  [[ -f "$1" ]] || fail "missing required file: $1"
}

non_auto_dirty_status() {
  git -C "$REPO_ROOT" status --porcelain | awk '$2 !~ /^\.vc4_auto\// && $1 !~ /^\?\?$/ { print; next } $1 == "??" && $2 !~ /^\.vc4_auto\// { print }'
}

ensure_clean_before_autorun() {
  if (( ALLOW_DIRTY )); then
    return 0
  fi
  local dirty
  dirty="$(non_auto_dirty_status || true)"
  if [[ -n "$dirty" ]]; then
    log "non-.vc4_auto dirty repo state before autorun:"
    printf '%s\n' "$dirty" | tee -a "$LOG"
    fail "commit/stash non-.vc4_auto changes or rerun with --allow-dirty"
  fi
}

run_logged() {
  local label="$1"
  shift
  log "RUN $label: $*"
  if (( DRY_RUN )); then
    log "DRY-RUN skip $label"
    return 0
  fi
  set +e
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  set -e
  if (( rc == 0 )); then
    log "OK $label"
  else
    log "FAIL $label rc=$rc"
  fi
  return "$rc"
}

verify_slice() {
  local slice="$1"
  local phase="$2"
  local report="$REPORT_DIR/${slice}.${phase}.json"
  local -a cmd=(
    "$PYTHON" "$VERIFIER" verify
    --repo "$REPO_ROOT"
    --spec "$SPEC"
    --worklist "$WORKLIST"
    --slice "$slice"
    --out "$report"
    --timeout-sec "$GATE_TIMEOUT_SEC"
    --keep-going
  )
  if (( NO_HARDWARE_VERIFY )); then
    cmd+=(--no-hardware)
  fi
  run_logged "typed-verifier:$slice:$phase" "${cmd[@]}"
}

run_autorun_slice() {
  local slice="$1"
  ensure_clean_before_autorun
  export GPT_WEB_CURRENT_CHAT_URL="$CHAT_URL"
  local -a cmd=(
    "$PYTHON" "$AUTORUN" run
    --slice "$slice"
    --verbose
    --gpt-mode "$GPT_MODE"
    --chat-timeout-sec "$CHAT_TIMEOUT_SEC"
    --gate-timeout-sec "$GATE_TIMEOUT_SEC"
  )
  if (( ALLOW_DIRTY )); then
    cmd+=(--allow-dirty)
  fi
  if [[ -n "${VC4_M1_RESUME_EXTRA_AUTORUN_ARGS:-}" ]]; then
    # Intentional shell splitting for explicit user-provided extra flags.
    # shellcheck disable=SC2206
    local extra=( ${VC4_M1_RESUME_EXTRA_AUTORUN_ARGS} )
    cmd+=("${extra[@]}")
  fi
  run_logged "autorun:$slice" "${cmd[@]}"
}

require_file "$AUTORUN"
require_file "$VERIFIER"
require_file "$SPEC"
require_file "$WORKLIST"

log "repo: $REPO_ROOT"
log "log:  ${LOG#$REPO_ROOT/}"
log "reports: ${REPORT_DIR#$REPO_ROOT/}"
log "chat: $CHAT_URL"
log "range: ${SLICES[$FROM_IDX]} .. ${SLICES[$TO_IDX]}"
log "policy: verify-first; autorun only when verifier fails; verify again after autorun"
if (( VERIFY_ONLY )); then log "mode: verify-only"; fi
if (( NO_HARDWARE_VERIFY )); then log "verifier flag: --no-hardware"; fi

for (( idx = FROM_IDX; idx <= TO_IDX; idx++ )); do
  slice="${SLICES[$idx]}"
  log ""
  log "=== START $slice $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

  if verify_slice "$slice" "pre"; then
    log "SKIP autorun for $slice: typed verifier already passes"
    log "=== PASS $slice $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    git -C "$REPO_ROOT" log -1 --oneline 2>/dev/null | tee -a "$LOG" || true
    continue
  fi

  if (( VERIFY_ONLY )); then
    fail "typed verifier failed for $slice in --verify-only mode"
  fi

  log "typed verifier failed for $slice; invoking autorun once for this slice"
  if ! run_autorun_slice "$slice"; then
    log "autorun failed for $slice; stopping"
    log "last commit:"
    git -C "$REPO_ROOT" log -1 --oneline 2>/dev/null | tee -a "$LOG" || true
    log "git status:"
    git -C "$REPO_ROOT" status --short | tee -a "$LOG" || true
    exit 1
  fi

  if ! verify_slice "$slice" "post"; then
    log "post-autorun typed verifier failed for $slice; stopping"
    log "last commit:"
    git -C "$REPO_ROOT" log -1 --oneline 2>/dev/null | tee -a "$LOG" || true
    log "git status:"
    git -C "$REPO_ROOT" status --short | tee -a "$LOG" || true
    exit 1
  fi

  log "=== PASS $slice $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  git -C "$REPO_ROOT" log -1 --oneline 2>/dev/null | tee -a "$LOG" || true
done

log ""
log "ALL REQUESTED SLICES VERIFIED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
git -C "$REPO_ROOT" log --oneline -n 16 2>/dev/null | tee -a "$LOG" || true
