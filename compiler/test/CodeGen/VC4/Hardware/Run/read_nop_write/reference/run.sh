#!/bin/bash
set -euo pipefail

echo "ASSEMBLING QASM"
set +e
out=$(vc4asm -c read_nop_writeshader.c -h read_nop_writeshader.h read_nop_write.qasm 2>&1)
rc=$?
set -e

if [[ $rc -ne 0 || -n "$out" ]]; then
  echo "▶ ASSEMBLY FAILED WITH OUTPUT:"
  printf '%s\n' "$out"
  exit 1
else
  echo "RUNNING MAKE"
  make
fi
