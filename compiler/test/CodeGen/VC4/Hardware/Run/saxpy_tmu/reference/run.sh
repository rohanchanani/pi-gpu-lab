#!/bin/bash
set -euo pipefail

echo "ASSEMBLING QASM"
if ! out=$(vc4asm -c saxpy_tmushader.c -h saxpy_tmushader.h saxpy_tmu.qasm 2>&1); then
  echo "▶ ASSEMBLY FAILED WITH OUTPUT:"
  printf '%s\n' "$out"
  exit 1
fi

if [[ -n "$out" ]]; then
  echo "▶ ASSEMBLY PRODUCED UNEXPECTED OUTPUT:"
  printf '%s\n' "$out"
  exit 1
fi

echo "RUNNING MAKE"
make
