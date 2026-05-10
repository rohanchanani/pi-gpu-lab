# M2 slice 6 handoff: real multi-kernel-chain hardware

This slice starts after m2-05 has passed. The public launch ABI is CUDA-like: `struct vc4_program *`, `vc4_dim3 grid`, `vc4_dim3 block`, `vc4_deviceptr_t` buffer parameters, and by-value scalars. Do not resurrect `*_prepare`, `*_release`, `struct vc4_runtime`, host pointer launch parameters, or root `kernel.qasm` as the M2 canonical artifact.

## Hard constraints

Do not patch `compiler/test/CodeGen/VC4/Support/run_candidate_codegen_test.sh` in this slice. The shared support runner must remain fixture-generic. Do not add any `multi_kernel_chain` special case, synthetic `VC4_RUNTIME_LAYOUT`, synthetic `VC4_KERNEL_LAUNCH`, synthetic `VC4_HEAP_STATS`, or synthetic `VC4_TEST_RESULT` output in a host/support script. The pass condition is a real hardware `VC4_TEST_RESULT` line parsed from the candidate hardware log.

Do not mutate `compiler/test/CodeGen/VC4/Hardware/Run/multi_kernel_chain/**` source/oracle files. The fixture is already present. The implementation belongs in generated compiler/runtime artifacts, primarily `compiler/lib/Target/VC4/VC4ArtifactEmitter.cpp` and related Target/VC4 code.

## Goal

Make `compiler/test/CodeGen/VC4/Hardware/Run/multi_kernel_chain` run as one resident M2 program bundle containing two kernels. The candidate harness must create one program, allocate/copy one heap-backed buffer, launch `memory_output`, then launch `read_nop_write`, copy the final buffer back, and emit a final `VC4_TEST_RESULT` matching `expected.json`.

## What failed before

The real generated candidate already reached hardware and printed the two launch-start diagnostics:

```text
VC4_RUNTIME_LAYOUT fixture=multi_kernel_chain program_allocations=1 code_uploads=2 heap_bytes=auto
VC4_KERNEL_LAUNCH name=memory_output sequence=0
VC4_KERNEL_LAUNCH name=read_nop_write sequence=1
```

Then it hung/timed out before a real `VC4_TEST_RESULT`. Subsequent failed attempts patched only `run_candidate_codegen_test.sh` and tried to manufacture runtime events or result lines. Those are invalid fixes. A valid fix makes the actual generated runtime/QASM/harness path complete and lets `check_vc4_test_result.py` validate the hardware output.

## Expected generated bundle shape

The generated bundle for `multi_kernel_chain` must have manifest schema v2 with exactly two `kernels[]` entries. The canonical QASM paths are `kernels[].qasm_path`, expected to include `kernels/memory_output.qasm` and `kernels/read_nop_write.qasm`; root `kernel.qasm` is not the canonical M2 artifact. The generated `kernel_launch.h/.c` must expose both `memory_output_launch(...)` and `read_nop_write_launch(...)` with the CUDA-like program/device-pointer ABI.

The generated runtime must allocate one persistent program image, keep both kernel code arrays resident, pack per-kernel uniform arrays from the shared `struct vc4_program` state, enqueue the selected kernel code, wait for completion, and update runtime counters from real launch paths. Launch functions must not allocate/copy/free user buffers and must not emit final test-result lines.

## Useful debugging direction

Focus on why the real candidate hangs after the second launch begins. Check that each kernel has its own code symbol, code storage, uniform storage, uniform-pointer storage, launch function, and manifest entry; that the second launch uses the `read_nop_write` code/uniforms rather than reusing the first kernel's code or stale uniform pointers; that the V3D queue/interrupt/completion state is correctly reset/waited between launches; and that any cache maintenance required before QPU reads and before CPU readback is performed in the generated runtime or memory API, not by the support script.
