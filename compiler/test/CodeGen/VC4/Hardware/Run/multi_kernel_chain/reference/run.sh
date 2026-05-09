#!/usr/bin/env bash
set -euo pipefail
vc4asm -c memory_outputshader.c -h memory_outputshader.h memory_output.qasm
vc4asm -c read_nop_writeshader.c -h read_nop_writeshader.h read_nop_write.qasm
make RUN=0 3-test-multi_kernel_chain.bin
pi-install ./3-test-multi_kernel_chain.bin
