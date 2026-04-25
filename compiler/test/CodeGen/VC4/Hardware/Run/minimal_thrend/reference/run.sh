#!/bin/bash
set -euo pipefail

echo "ASSEMBLING QASM"
set +e
out=$(vc4asm -c minimal_threndshader.c -h minimal_threndshader.h minimal_thrend.qasm 2>&1)
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
