#!/bin/bash
set -uo pipefail

echo "ASSEMBLING QASM"
out=$(vc4asm -c matmul_blockedshader.c -h matmul_blockedshader.h matmul_blocked.qasm 2>&1)
asm_rc=$?

if [[ $asm_rc -ne 0 ]]; then
  echo "▶ ASSEMBLY FAILED WITH OUTPUT:"
  printf '%s\n' "$out"
  exit "$asm_rc"
fi

if [[ -n "$out" ]]; then
  echo "▶ ASSEMBLY PRODUCED UNEXPECTED OUTPUT:"
  printf '%s\n' "$out"
  exit 1
fi

echo "RUNNING MAKE"
make
