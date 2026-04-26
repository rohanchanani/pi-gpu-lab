#!/bin/bash
set -euo pipefail

echo "ASSEMBLING QASM"
if ! out=$(vc4asm -c minimal_threndshader.c -h minimal_threndshader.h minimal_thrend.qasm 2>&1); then
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
