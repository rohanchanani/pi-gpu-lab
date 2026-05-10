# VC4 Codegen M2 slice-7 workflow handoff

This handoff exists because the first slice-7 run made progress but burned attempts on missing/underused context and known-bad workaround patterns. Use it for slice 7 and carry the lessons into slices 8-10.

## Current slice-7 target

Slice 7 is `m2-07-independent-vector-scheduler`. Its verifier matrix is `independent_vector`:

- `saxpy_16`
- `saxpy_basic`
- `saxpy_full`
- `global_store_coalesced_multi`
- `gemv_naive_tail`

The prompt must include the full text fixture inputs and trusted reference files for those fixtures. Do not infer the contents of `input.mlir`, qasm, launchers, harnesses, run scripts, or expected JSON from filenames or logs.

## What happened in the first failed slice-7 run

- Initial state failed the resource contract because the generated `saxpy_full` manifest did not expose the expected `saxpy_full` public name.
- A later attempt reached the fixture matrix. `saxpy_16`, `saxpy_basic`, `saxpy_full`, and `global_store_coalesced_multi` progressed, but `gemv_naive_tail` failed in generate/assemble with the existing scheduled input. The useful failures are in the verifier logs, not just in the summary tails.
- Another attempt added broad legacy-wrapper compatibility machinery and test rewrites. That is the wrong direction for M2. Slice 7 should implement the current CUDA-like resident program/runtime shape, not restore M1-style prepare/release or host-copy wrappers.

## Known-bad patterns to avoid

- Do not add or restore `struct vc4_runtime`, `*_prepare(...)`, `*_release(...)`, or host-wrapper launch entry points.
- Do not patch `run_candidate_codegen_test.sh` to inject synthetic `VC4_TEST_RESULT`, `VC4_KERNEL_LAUNCH`, or runtime counter lines.
- Do not weaken expected JSON, mutate fixture reference trees, or add fixture-name/public-name codegen special cases.
- Do not special-case `gemv_naive_tail`, `saxpy_16`, or any matrix fixture name in `VC4ArtifactEmitter.cpp`.
- Do not rewrite earlier slice tests just to match a guessed runtime shape. If a lit expectation must change, it must reflect a real stricter/equally strict runtime ABI evolution and the cumulative prefix must remain green.

## Implementation direction that is in scope

- Fix generic scheduled-stream QASM emission for the instruction forms present in the full matrix fixtures.
- Fix generic launch/runtime scheduling for independent-vector kernels: launch logical work requests in waves, pack logical request IDs and tail metadata via uniforms, and avoid assigning work by physical QPU index.
- Use the full fixture source context and full failure logs. If a needed fixture/source/log is absent from the prompt, stop and ask for it rather than reconstructing missing functions or guessing fixture contents.
