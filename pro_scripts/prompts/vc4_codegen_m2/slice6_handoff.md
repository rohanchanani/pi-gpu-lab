# M2 slice 6 handoff: real multi-kernel-chain hardware

This slice starts after m2-05 has passed. The public launch ABI is CUDA-like: `struct vc4_program *`, `vc4_dim3 grid`, `vc4_dim3 block`, `vc4_deviceptr_t` buffer parameters, and by-value scalars. Do not resurrect `*_prepare`, `*_release`, `struct vc4_runtime`, host pointer launch parameters, or root `kernel.qasm` as the M2 canonical artifact.

## Context availability

The m2-06 context profile intentionally includes the full `compiler/lib/Target/VC4/VC4ArtifactEmitter.cpp` file, not just a focused excerpt. Use that full emitter source when changing manifest, layout, generated `kernel_launch.h/.c`, QASM path/code-symbol handling, cache/runtime synchronization, launch waiting, or multi-kernel state layout. Do not infer missing implementation details from snippets if the full file is present in the prompt.

Read the full emitter before editing. Prefer a small, local patch over a broad rewrite: preserve the M2 slice-0 through slice-5 manifest/layout/heap/launch-ABI behavior unless the failing m2-06 evidence directly requires a change. In particular, keep `layout.json` heap accounting positive (`heap_size_bytes` must remain present and > 0), keep exactly two layout kernel records for `multi_kernel_chain`, and keep all public launch functions on the CUDA-like program/device-pointer ABI.

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

Then it hung/timed out before a real `VC4_TEST_RESULT`. Subsequent failed attempts patched only `run_candidate_codegen_test.sh` and tried to manufacture runtime events or result lines. Those are invalid fixes. A valid fix makes the actual generated runtime/QASM/harness path complete and lets the result checker validate the hardware output.

Two recent implementation traps are now explicitly forbidden by the verifier and should not be repeated:

* Do not make `layout.json` lose its persistent heap. A failed candidate built but produced `heap_size_bytes <= 0`, causing the `multi-kernel-chain-layout` verifier to fail before hardware. Preserve or repair the existing positive heap-size computation while adding multi-kernel support.
* Do not introduce C++ const-correctness compile failures when using `KernelRecord` values. In this checkout `mlir::vc4::FuncOp::getOperation()` is non-const; if iterating `const KernelRecord &kernel`, do not call `kernel.func.getOperation()` directly. Use a mutable local/copy where needed, pass an existing non-const `Operation *`, or keep writer helpers structured so they receive a non-const diagnostic operation. The build must pass before any hardware gate runs.

## Expected generated bundle shape

The generated bundle for `multi_kernel_chain` must have manifest schema v2 with exactly two `kernels[]` entries. The canonical QASM paths are `kernels[].qasm_path`, expected to include `kernels/memory_output.qasm` and `kernels/read_nop_write.qasm`; root `kernel.qasm` is not the canonical M2 artifact. The generated `kernel_launch.h/.c` must expose both `memory_output_launch(...)` and `read_nop_write_launch(...)` with the CUDA-like program/device-pointer ABI.

The generated runtime must allocate one persistent program image, keep both kernel code arrays resident, pack per-kernel uniform arrays from the shared `struct vc4_program` state, enqueue the selected kernel code, wait for completion, and update runtime counters from real launch paths. Launch functions must not allocate/copy/free user buffers and must not emit final test-result lines.


### Additional failure evidence from attempts 7 and 8

A later attempt rewrote a large part of `VC4ArtifactEmitter.cpp` but the static generated-runtime contract caught that both public launch functions were not obviously enqueueing real V3D work: `memory_output_launch` and `read_nop_write_launch` were missing launch-body writes to `V3D_SRQUA` and `V3D_SRQPC`. Do not hide enqueue operations in comments or support scripts; each public launch path must pack its own uniforms and enqueue the selected kernel code for real.

Another attempt reached hardware and printed `VC4_KERNEL_LAUNCH name=memory_output ... runtime_launches=1`, then hung before launching/completing `read_nop_write` and before `VC4_TEST_RESULT`. This is a generated-runtime bug, not a support-runner problem. Make launch waits bounded: do not spin forever on V3D/QPU completion. If a launch cannot complete, increment/return through a real launch-failure path so the harness can emit a hardware result/failure quickly instead of timing out at the verifier level. The final passing result still requires `launch_failures=0`.

The verifier now treats the layout/runtime static checks as stop-on-failure pre-hardware gates. If `layout.json` shrinks `heap_size_bytes` below the established M2 persistent-heap scale, if launch functions lack real V3D queue writes, or if generated code lacks a bounded launch wait/failure path, the slice will fail before hardware to produce a focused failure packet.

## Useful debugging direction

Focus on why the real candidate hangs after the second launch begins. Check that each kernel has its own code symbol, code storage, uniform storage, uniform-pointer storage, launch function, and manifest entry; that the second launch uses the `read_nop_write` code/uniforms rather than reusing the first kernel's code or stale uniform pointers; that the V3D queue/interrupt/completion state is correctly reset/waited between launches; and that any cache maintenance required before QPU reads and before CPU readback is performed in the generated runtime or memory API, not by the support script.
