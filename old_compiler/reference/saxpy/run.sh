#!/bin/bash

echo "ASSEMBLING QASM"
out=$(vc4asm -c saxpyshader.c -h saxpyshader.h saxpy.qasm 2>&1)

if [[ -n $out ]]; then
  echo "▶ ASSEMBLY FAILED WITH OUTPUT:"
  printf '%s\n' "$out"
else
  echo "RUNNING MAKE"
  make
fi
