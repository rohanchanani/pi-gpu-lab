# VC4 Codegen M2 post-slice-6 handoff

Slice 6 established the first real two-kernel resident hardware program. Carry these constraints forward into slices 7-10.

## Preserve the landed shape

- `VC4ArtifactEmitter.cpp` must keep emitting QASM from the scheduled VC4 instruction stream. Do not add fixture-name or `public_name` branches such as `memory_output`, `read_nop_write`, `saxpy_full`, or later matrix fixture names.
- Single-kernel programs remain one-kernel program bundles. Do not restore top-level `kernel.qasm` singleton assumptions or legacy prepare/release wrappers.
- Public launch APIs stay CUDA-like: `struct vc4_program *`, `vc4_dim3` grid/block, device pointers for buffers, and by-value scalars. Launch bodies pack uniforms, enqueue hardware, have bounded wait/failure paths, and do not allocate/copy/free user buffers.
- Hardware harnesses own `VC4_TEST_RESULT` lines. Generated runtime may emit real runtime diagnostics such as `VC4_RUNTIME_LAYOUT` and `VC4_KERNEL_LAUNCH`, but must not synthesize pass/fail result lines.

## Do not shortcut hardware failures

- A pi timeout or empty pre-runtime log can be transient. Use the workflow retry/power-cycle path, but do not skip hardware verification or weaken expected JSON.
- If static generated-runtime/resource/layout contracts fail, fix those before spending hardware time.
- If hardware emits a real `VC4_TEST_RESULT` with wrong values, treat it as a semantic compiler/runtime failure, not a transient harness issue.

## Keep the prefix green

- A later slice may legitimately require updating an earlier slice's lit expectation when the generated runtime ABI evolves, but only when the compiler behavior is correct and the updated check is stricter or equally strict.
- Cumulative prefix rechecks are intentional. If slice N passes and N-1 fails, fix the regression or test-expectation mismatch before continuing.

## Known bad patterns from slice 6

- Do not patch `run_candidate_codegen_test.sh` to inject fixture-specific `VC4_TEST_RESULT` or synthetic runtime lines.
- Do not mutate checked-in hardware fixture reference trees, expected JSON, `.vc4_auto/**`, lit `Output/**`, `.lit_test_times.txt`, or `run.log` artifacts.
- Do not replace generic QASM emission with hardcoded fixture QASM emitters.
