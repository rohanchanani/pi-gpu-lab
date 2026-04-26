#!/bin/bash
set -euo pipefail

echo "ASSEMBLING QASM"
out=$(vc4asm -c saxpy_basicshader.c -h saxpy_basicshader.h saxpy_basic.qasm 2>&1)

if [[ -n $out ]]; then
  echo "▶ ASSEMBLY FAILED WITH OUTPUT:"
  printf '%s\n' "$out"
  exit 1
else
  echo "RUNNING MAKE"
  make
fi
