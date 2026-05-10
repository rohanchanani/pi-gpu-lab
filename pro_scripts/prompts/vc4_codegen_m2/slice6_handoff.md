# M2 slice 6 handoff: multi-kernel-chain hardware

This slice starts after m2-05 has passed. The public launch ABI is now CUDA-like: `struct vc4_program *`, `vc4_dim3 grid`, `vc4_dim3 block`, `vc4_deviceptr_t` buffer parameters, and by-value scalars. Do not resurrect `*_prepare`, `*_release`, `struct vc4_runtime`, host pointer launch parameters, or root `kernel.qasm` as the M2 canonical artifact.

Goal: make `compiler/test/CodeGen/VC4/Hardware/Run/multi_kernel_chain` run as one resident program bundle containing two kernels. The candidate harness must create one program, allocate/copy one heap-backed buffer, launch `memory_output`, then launch `read_nop_write`, and emit a final `VC4_TEST_RESULT` line matching `expected.json`.

Verifier plumbing note: the `reference_hardware` phase may generate files under the checked-in reference tree while it runs. The verifier cleans those byproducts afterward. Do not commit generated `reference/*shader.c`, `reference/*shader.h`, `reference/objs/**`, or `reference/run.log`.

Expected stable runtime log lines for this slice include `VC4_RUNTIME_LAYOUT`, one `VC4_KERNEL_LAUNCH name=memory_output`, one `VC4_KERNEL_LAUNCH name=read_nop_write`, and final counters including `program_allocations=1`, `code_uploads=2`, `runtime_launches>=2`, and `launch_failures=0`.
