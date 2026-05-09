# VC4 Codegen M1 -> M2 handoff

This handoff is intentionally short and is safe to include in every fresh M2 GPT prompt.

## M1 baseline

- M1 implemented `vc4-codegen` for already-scheduled VC4 QPU input.
- The current artifact emitter is legacy/single-kernel shaped: it accepts one `vc4.module` with exactly one eligible `vc4.func` where `domain = qpu`, `form = scheduled`, `kernel`, and `vc4.launch_abi` are present.
- The current bundle has legacy top-level manifest fields such as `kind`, `bundle_format`, `kernel`, `symbol_name`, `public_name`, `scheduled_sink_ops`, `uniform_words_per_qpu`, and `kernel_info`.
- The current emitter rejects multi-kernel programs with the diagnostic fragment `expected exactly one eligible VC4 QPU kernel`.

## M2-01 manifest-v2 target

- M2 must treat every accepted program as a program artifact, including the one-kernel case.
- A single kernel is represented as `kernels.length == 1`; do not keep a special legacy manifest path for one-kernel inputs.
- Manifest v2 must include top-level `schema_version`, `kind`, `program_name`, `target`, and `kernels`.
- Collect all eligible scheduled QPU kernel functions within the one VC4 module for M2 program emission.
- Each kernel entry must carry stable per-kernel fields such as `kernel_id`, `symbol_name`, `public_name`, `code_symbol`, `qasm_path`, and launch/resource metadata required by the tests.
- Write distinct per-kernel artifact paths/names for multi-kernel bundles rather than reusing `kernel.qasm` as a singleton-only assumption.
- Reject duplicate `public_name` and duplicate `code_symbol` with diagnostics containing both `duplicate` and the field name before falling back to generic collection errors.

## Do not regress

- Do not weaken verifier checks or M2 lit tests to make the slice pass.
- Do not mutate existing hardware reference/oracle files, `.vc4_auto/**`, generated `Output/**`, or `.lit_test_times.txt`.
- Keep existing M1 single-kernel behavior working through manifest-v2 one-element program artifacts.
