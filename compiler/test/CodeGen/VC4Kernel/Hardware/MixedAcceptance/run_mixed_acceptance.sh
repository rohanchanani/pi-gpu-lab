#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
manifest="${script_dir}/mixed_acceptance_manifest.json"
repo_root="$(cd "${script_dir}/../../../../../../" && pwd)"
state_root="${VC4_CODEGEN_STATE_ROOT:-}"
list_only=0
fixtures=()

usage() {
  cat <<'EOF'
usage: run_mixed_acceptance.sh [--manifest PATH] [--repo-root PATH]
                               [--state-root PATH] [--list]
                               [--fixture NAME ...]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest)
      manifest="$2"
      shift 2
      ;;
    --repo-root)
      repo_root="$2"
      shift 2
      ;;
    --state-root)
      state_root="$2"
      shift 2
      ;;
    --fixture)
      fixtures+=("$2")
      shift 2
      ;;
    --list)
      list_only=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unexpected argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${state_root}" ]]; then
  timestamp="$(date +%Y%m%d_%H%M%S)"
  state_root="${repo_root}/.vc4_auto/codegen_mixed_acceptance_${timestamp}"
fi

fixture_filter_json="$(
  python3 - "$@" <<'PY' "${fixtures[@]}"
import json
import sys
print(json.dumps(sys.argv[1:]))
PY
)"

entries="$(
  python3 - "${manifest}" "${fixture_filter_json}" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
selected = set(json.loads(sys.argv[2]))
seen = set()
for fixture in manifest["fixtures"]:
    name = fixture["name"]
    if selected and name not in selected:
        continue
    seen.add(name)
    print("\t".join([
        name,
        fixture["status"],
        fixture["runner_kind"],
        fixture["path"],
        fixture["dialect"],
    ]))
missing = selected - seen
if missing:
    print("missing requested fixture(s): " + ", ".join(sorted(missing)), file=sys.stderr)
    raise SystemExit(1)
PY
)"

if [[ "${list_only}" -eq 1 ]]; then
  if [[ -z "${entries}" ]]; then
    echo "no mixed acceptance fixtures selected"
    exit 0
  fi
  while IFS=$'\t' read -r name status runner path dialect; do
    [[ -z "${name}" ]] && continue
    printf '%s\t%s\t%s\t%s\t%s\n' "${name}" "${status}" "${runner}" "${dialect}" "${path}"
  done <<< "${entries}"
  exit 0
fi

mkdir -p "${state_root}/logs"

if [[ -z "${entries}" ]]; then
  echo "no mixed acceptance fixtures selected"
  exit 0
fi

while IFS=$'\t' read -r name status runner path dialect; do
  [[ -z "${name}" ]] && continue
  if [[ "${status}" == "planned" ]]; then
    echo "SKIP planned mixed acceptance fixture: ${name}"
    continue
  fi

  fixture_dir="${repo_root}/${path}"
  if [[ ! -d "${fixture_dir}" ]]; then
    echo "missing mixed acceptance fixture directory: ${fixture_dir}" >&2
    exit 1
  fi
  log_path="${state_root}/logs/${name}.log"
  echo "RUN mixed acceptance fixture: ${name}"
  case "${runner}" in
    vc4kernel_candidate)
      VC4_CODEGEN_STATE_ROOT="${state_root}" \
        "${repo_root}/compiler/test/CodeGen/VC4Kernel/Support/run_vc4kernel_candidate_codegen_test.sh" \
        "${name}" all 2>&1 | tee "${log_path}"
      ;;
    ssavc4_candidate)
      VC4_CODEGEN_STATE_ROOT="${state_root}" \
        "${repo_root}/compiler/test/CodeGen/SSAVC4/Support/run_ssavc4_candidate_codegen_test.sh" \
        "${name}" all 2>&1 | tee "${log_path}"
      ;;
    *)
      echo "unsupported runner_kind for ${name}: ${runner}" >&2
      exit 1
      ;;
  esac
  if ! grep -q 'VC4_TEST_RESULT' "${log_path}"; then
    echo "missing VC4_TEST_RESULT for ${name}" >&2
    exit 1
  fi
  grep 'VC4_TEST_RESULT' "${log_path}"
  if ! grep 'VC4_TEST_RESULT' "${log_path}" | grep -q 'status=PASS'; then
    echo "mixed acceptance fixture failed after boot or result validation: ${name}" >&2
    exit 1
  fi
done <<< "${entries}"
