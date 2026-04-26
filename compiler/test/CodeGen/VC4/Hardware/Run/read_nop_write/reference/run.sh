#!/bin/bash
set -euo pipefail

echo "ASSEMBLING QASM"
if ! out=$(vc4asm -c read_nop_writeshader.c -h read_nop_writeshader.h read_nop_write.qasm 2>&1); then
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
