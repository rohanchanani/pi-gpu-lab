#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
manifest="${script_dir}/mixed_acceptance_manifest.json"
repo_root="$(cd "${script_dir}/../../../../../../" && pwd)"
state_root="${VC4_CODEGEN_STATE_ROOT:-.vc4_auto/codegen_ttir_mixed}"
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
    --manifest) manifest="$2"; shift 2 ;;
    --repo-root) repo_root="$2"; shift 2 ;;
    --state-root) state_root="$2"; shift 2 ;;
    --fixture) fixtures+=("$2"); shift 2 ;;
    --list) list_only=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unexpected argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "${state_root}" in
  /*) ;;
  *) state_root="${repo_root}/${state_root}" ;;
esac

if [[ ${#fixtures[@]} -eq 0 ]]; then
  fixture_filter_json="[]"
else
  fixture_filter_json="$(python3 - "${fixtures[@]}" <<'PY'
import json
import sys
print(json.dumps(sys.argv[1:]))
PY
)"
fi

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
    print("\t".join([name, fixture["status"], fixture["runner_kind"], fixture["path"], fixture["dialect"]]))
missing = selected - seen
if missing:
    print("missing requested fixture(s): " + ", ".join(sorted(missing)), file=sys.stderr)
    raise SystemExit(1)
PY
)"

if [[ "${list_only}" -eq 1 ]]; then
  if [[ -z "${entries}" ]]; then
    echo "no TTIR mixed acceptance fixtures selected"
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
  echo "no TTIR mixed acceptance fixtures selected"
  exit 0
fi

while IFS=$'\t' read -r name status runner path dialect; do
  [[ -z "${name}" ]] && continue
  if [[ "${status}" != "implemented" ]]; then
    echo "unsupported TTIR mixed acceptance fixture status for ${name}: ${status}" >&2
    exit 1
  fi
  if [[ "${runner}" != "ttir_candidate" ]]; then
    echo "unsupported runner_kind for ${name}: ${runner}" >&2
    exit 1
  fi
  if [[ "${dialect}" != "ttir" ]]; then
    echo "unsupported dialect for ${name}: ${dialect}" >&2
    exit 1
  fi

  fixture_dir="${repo_root}/${path}"
  if [[ ! -d "${fixture_dir}" ]]; then
    echo "missing TTIR mixed acceptance fixture directory: ${fixture_dir}" >&2
    exit 1
  fi

  log_path="${state_root}/logs/${name}.log"
  echo "RUN TTIR mixed acceptance fixture: ${name}"
  VC4_CODEGEN_STATE_ROOT="${state_root}" \
    "${repo_root}/compiler/test/CodeGen/Triton/Support/run_ttir_candidate_codegen_test.sh" \
    "${fixture_dir}" all 2>&1 | tee "${log_path}"

  result_line="$(grep 'VC4_TEST_RESULT' "${log_path}" | tail -n 1 || true)"
  if [[ -z "${result_line}" ]]; then
    echo "missing VC4_TEST_RESULT for ${name}" >&2
    exit 1
  fi
  echo "${result_line}"
  if [[ "${result_line}" != *"status=PASS"* ]]; then
    echo "TTIR mixed acceptance fixture failed: ${name}" >&2
    exit 1
  fi
  for zero_field in total_mismatches sentinel_mismatches launch_failures; do
    if [[ "${result_line}" == *"${zero_field}="* && "${result_line}" != *"${zero_field}=0"* ]]; then
      echo "TTIR mixed acceptance fixture has nonzero ${zero_field}: ${name}" >&2
      exit 1
    fi
  done
  for required_field in active_qpus=12 lanes=16; do
    if [[ "${result_line}" != *"${required_field}"* ]]; then
      echo "TTIR mixed acceptance fixture missing ${required_field}: ${name}" >&2
      exit 1
    fi
  done
  if [[ "${result_line}" == *"output_hash=0"* || "${result_line}" != *"output_hash="* ]]; then
    echo "TTIR mixed acceptance fixture missing nonzero output_hash: ${name}" >&2
    exit 1
  fi
done <<< "${entries}"
