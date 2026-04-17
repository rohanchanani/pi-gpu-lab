# Full SAXPY Harness

This directory is the current end-to-end bridge from pipeline-consumed MLIR to
a runnable Pi-side test harness.

What is handwritten here:

- `3-test-saxpy.c`
- `Makefile`
- `mailbox.c`
- `mailbox.h`
- `run.sh`

What the compiler path is expected to generate into this directory:

- `saxpy.qasm`
- `saxpy_launch.c`
- `saxpy_launch.h`

What `vc4asm` is expected to generate after that:

- `saxpyshader.c`
- `saxpyshader.h`

Bundled assembler support kept here:

- `share/vc4inc/vc4.qinc`

Input:

- `saxpy-full.mlir` is the minimal supported kernel-side MLIR input for the
  current compiler slice.

Current workflow:

1. lower and emit artifacts from `saxpy-full.mlir`
2. rename/copy emitted `kernel.qasm`, `kernel_launch.c`, and `kernel_launch.h`
   to the `saxpy.*` names expected by this harness
3. run `bash run.sh`

The host-side sample app remains handwritten for now. This directory keeps only
the kernel-side MLIR that the current pipeline actually consumes.
