# Compiler-Owned SAXPY Reference Target

This directory is the compiler-owned runnable reference target for the current
narrow SAXPY slice defined by [backend_spec.md](../../docs/backend_spec.md).

What remains the handwritten baseline:
- `code/3-saxpy/` is preserved unchanged as the original handwritten working
  reference and fallback baseline.

What this directory is:
- the compiler-owned reference artifact set rooted in:
  - `saxpy.qasm`
  - `saxpy_launch.c`
  - `saxpy_launch.h`
- plus the minimal supporting handwritten runtime/build files needed so the
  reference can still be built and run from this directory with `bash run.sh`

How this matches the spec target:
- public launcher interface is:
  `int saxpy_launch(struct vc4_runtime *rt, float *x, float *y, float a, uint32_t n);`
- the launcher exposes only semantic kernel arguments plus a runtime handle
- builtins stay internal to the launcher and device artifact
- lane width is a fixed runtime/backend constant: `vc4_runtime_lane_width() == 16`
- active QPU count is runtime launch policy, not a kernel-local constant
- the current runtime default is `vc4_runtime_active_qpus(&rt) == 12`
- physical uniform stream order per QPU is:
  `[0] x, [1] y, [2] a, [3] n, [4] qpu_id, [5] num_qpus`
- `saxpy.qasm` consumes that stream sequentially with repeated `mov ..., unif`
- the device-side execution model is:
  `base = qpu_id * 16`
  `stride = num_qpus * 16`
  explicit loop over 16-wide chunks
- alpha remains scalar in the launcher/runtime ABI even though the lowered VC4
  IR treats it as vector-typed due to lane-broadcast semantics

What was reused from the handwritten baseline:
- `mailbox.c` / `mailbox.h` generic VC4 services
- `run.sh`
- `Makefile` structure
- the general VPM/DMA/SAXPY machine structure in the handwritten qasm

What is reference-specific here:
- `saxpy.qasm` now matches the compiler ABI and lowered execution model
- `saxpy_launch.c/h` replace the old handwritten kernel-specific `saxpy.c/h`
  interface
- `saxpy_launch.c` still performs semantic marshaling and uniform packing, but
  it now gets the active QPU count from the generic runtime boundary
- `3-test-saxpy.c` now acts like a sample application that calls the launcher
  through the public generated-style interface instead of manipulating a
  benchmark-specific GPU struct

`bash run.sh` flow in this directory:
1. assemble `saxpy.qasm` into `saxpyshader.c` and `saxpyshader.h` with `vc4asm`
2. build the launcher, runtime, sample app, and shader wrapper with `make`
3. use the existing Pi install/run path from the copied handwritten build

On a Pi-configured setup matching the handwritten baseline environment, this is
intended to be the concrete runnable compiler reference target for the current
slice.

For off-Pi build validation only, the same entrypoint can be used as:
`RUN=0 bash run.sh`
