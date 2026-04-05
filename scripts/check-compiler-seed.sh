#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
expected_file="$repo_root/compiler/test/vc4-scaffold.expected"
tmp_output="$(mktemp)"
trap 'rm -f "$tmp_output"' EXIT

while IFS= read -r check; do
  [[ -z "$check" ]] && continue
  file="${check%%: *}"
  needle="${check#*: }"

  if ! grep -Fq "$needle" "$repo_root/$file"; then
    printf 'missing expected text in %s: %s\n' "$file" "$needle" >&2
    exit 1
  fi
done < "$expected_file"

bash "$repo_root/scripts/lower-saxpy-toy.sh" \
  "$repo_root/compiler/test/saxpy-toy.mlir" > "$tmp_output"

diff -u "$repo_root/compiler/test/saxpy-toy-lowered.vc4" "$tmp_output" >/dev/null

printf 'compiler seed scaffold check passed\n'
