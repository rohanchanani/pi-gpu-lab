#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_FILE="${1:-$ROOT_DIR/compile_commands.json}"

if ! command -v bear >/dev/null 2>&1; then
  echo "error: bear is required. Install it with: brew install bear" >&2
  exit 1
fi

if ! command -v vc4asm >/dev/null 2>&1; then
  echo "error: vc4asm is required and must be on PATH" >&2
  exit 1
fi

if [[ -z "${CS240LX_2025_PATH:-}" ]]; then
  echo "error: CS240LX_2025_PATH is not set" >&2
  exit 1
fi

TARGET_DIRS=(
  "code/0-deadbeef"
  "code/0-index"
  "code/1-parallel-add"
  "code/2-mandelbrot"
  "code/3-saxpy"
  "code/matmul"
)

rm -f "$OUT_FILE"

for dir in "${TARGET_DIRS[@]}"; do
  echo "==> Capturing compile commands for $dir"
  (
    cd "$ROOT_DIR/$dir"
    bear --append --output "$OUT_FILE" -- env RUN=0 bash run.sh
  )
done

echo "Wrote $OUT_FILE"
