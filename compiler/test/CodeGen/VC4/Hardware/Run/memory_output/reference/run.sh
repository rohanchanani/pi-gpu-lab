#!/bin/bash
set -euo pipefail

echo "ASSEMBLING QASM"
set +e
out=$(vc4asm -c memory_outputshader.c -h memory_outputshader.h memory_output.qasm 2>&1)
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
