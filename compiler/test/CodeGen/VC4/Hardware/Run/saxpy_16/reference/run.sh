#!/bin/bash
set -euo pipefail

echo "ASSEMBLING QASM"
if ! out=$(vc4asm -c saxpy_16shader.c -h saxpy_16shader.h saxpy_16.qasm 2>&1); then
  echo "▶ ASSEMBLY FAILED WITH OUTPUT:"
  printf '%s\n' "$out"
  exit 1
fi

if [[ -n $out ]]; then
  echo "▶ ASSEMBLY PRODUCED OUTPUT:"
  printf '%s\n' "$out"
fi

echo "RUNNING MAKE"
make
