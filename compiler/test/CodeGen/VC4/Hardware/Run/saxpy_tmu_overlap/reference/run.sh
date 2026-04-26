#!/bin/bash
set -euo pipefail

echo "ASSEMBLING QASM"
out=$(vc4asm -c saxpy_tmu_overlapshader.c -h saxpy_tmu_overlapshader.h saxpy_tmu_overlap.qasm 2>&1)

if [[ -n $out ]]; then
  echo "▶ ASSEMBLY FAILED WITH OUTPUT:"
  printf '%s\n' "$out"
  exit 1
else
  echo "RUNNING MAKE"
  make
fi
