#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
input_file="${1:-$repo_root/compiler/test/saxpy-toy.mlir}"
output_file="$repo_root/compiler/test/saxpy-toy-lowered.vc4"

grep -Fq 'func.func @saxpy_vec16' "$input_file"
grep -Fq 'vector.transfer_read' "$input_file"
grep -Fq 'arith.mulf' "$input_file"
grep -Fq 'arith.addf' "$input_file"
grep -Fq 'vector.transfer_write' "$input_file"

cat "$output_file"
