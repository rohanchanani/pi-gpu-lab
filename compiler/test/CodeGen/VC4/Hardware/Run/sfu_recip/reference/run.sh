#!/bin/bash
set -uo pipefail

echo "ASSEMBLING QASM"
out=$(vc4asm -c sfu_recipshader.c -h sfu_recipshader.h sfu_recip.qasm 2>&1)
asm_rc=$?

if [[ $asm_rc -ne 0 || -n "$out" ]]; then
  echo "▶ ASSEMBLY FAILED WITH OUTPUT:"
  printf '%s\n' "$out"
  exit "$asm_rc"
else
  echo "RUNNING MAKE"
  make
fi
