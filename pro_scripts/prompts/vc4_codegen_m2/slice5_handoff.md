# VC4 Codegen M2 slice 5 fresh-chat handoff

Slices m2-00 through m2-04 are expected to be complete before this slice runs.
The current backend already emits M2 program bundles: manifest schema v2 with
`kernels[]`, per-kernel qasm under `kernels/<public_name>.qasm`, `kernel_launch.c`,
`kernel_launch.h`, `layout.json`, and a persistent program image with a heap-backed
`vc4Malloc`/`vc4Free`/`vc4Memcpy*`/`vc4MemsetD8` API.

Slice m2-05 must finish the public CUDA-like launch ABI:

- Public launch prototypes are `int <public_name>_launch(struct vc4_program *program, vc4_dim3 grid, vc4_dim3 block, ...)`.
- Buffer kernel arguments are `vc4_deviceptr_t`; scalar kernel arguments are passed by value.
- Launch functions pack uniforms and enqueue V3D/QPU work. They must not allocate, free, copy host buffers, or expose a host-copy convenience wrapper.
- Allocation and host/device copies are explicit API calls made by callers/harnesses through `vc4Malloc`, `vc4MemcpyHtoD`, `vc4MemcpyDtoH`, `vc4MemcpyDtoD`, `vc4MemsetD8`, and `vc4Free`.
- The slice owns stale launch-wrapper tests: `emit-launch-abi-header.mlir`, `emit-launcher-uniform-packing.mlir`, and `emit-saxpy-full-qasm-artifact-regression.mlir`. Update them to the M2 program API; do not resurrect legacy `*_prepare(struct vc4_runtime*, ...)` expectations.

Known pitfalls from previous failed attempts:

- The verifier regexes are now valid Python regex strings. Satisfy them; do not weaken the spec.
- Do not reference undefined cache-maintenance symbols (`clean_dcache`, `flush_dcache`, `invalidate_dcache`) unless the generated source also defines or includes valid implementations available to the bare-metal harness.
- Do not delete existing launch-ABI helper functions from `VC4ArtifactEmitter.cpp`; prior failures came from removing helpers and causing compile errors.
- Hardware failure is judged by `VC4_TEST_RESULT` in the candidate hardware log; preserve the harness/result contract.
