#!/usr/bin/env bash
set -euo pipefail
vc4asm -c minimal_threndshader.c -h minimal_threndshader.h minimal_thrend.qasm
vc4asm -c memory_outputshader.c -h memory_outputshader.h memory_output.qasm
make RUN=0 3-test-multi_kernel_minimal.bin
pi-install ./3-test-multi_kernel_minimal.bin
