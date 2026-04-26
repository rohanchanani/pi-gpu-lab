#!/bin/bash
set -euo pipefail

echo "ASSEMBLING QASM"
if ! out=$(vc4asm -c qpu_num_registershader.c -h qpu_num_registershader.h qpu_num_register.qasm 2>&1); then
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
