#!/bin/bash
set -euo pipefail

echo "ASSEMBLING QASM"
out=$(vc4asm -c tmu_read_nop_writeshader.c -h tmu_read_nop_writeshader.h tmu_read_nop_write.qasm 2>&1)

if [[ -n $out ]]; then
  echo "▶ ASSEMBLY FAILED WITH OUTPUT:"
  printf '%s\n' "$out"
  exit 1
else
  echo "RUNNING MAKE"
  make
fi
