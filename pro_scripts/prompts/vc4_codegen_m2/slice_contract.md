# VC4 Codegen M2 Slice Contract

Each slice has a natural-language expected end state plus typed verifications. The implementation must satisfy both. Source products use exact paths; do not substitute nearby names.

M2 verifier specs may use these additional mechanisms:

- `manifest_schema`
- `program_artifact_bundle`
- `all_qasm_assemble`
- `program_layout_contract`
- `generated_runtime_contract`
- `heap_api_unit`
- `launch_abi_contract`
- `runtime_event_log`
- `fixture_matrix`
- `resource_contract`
- `support_script_contract`
- `negative_diagnostic`

Reference/oracle policy: existing references and expected JSON are immutable. Slices that create new fixtures may add their own reference/oracle files only when those paths are explicitly allowlisted and covered by source-product checks.

Scheduling policy: M2 may implement runtime scheduling for already-scheduled kernels. M2 must not implement `gpu` dialect lowering, register allocation, or structured VC4 instruction scheduling.

Bundle layout policy: root-level `kernel.qasm` is an M1 artifact spelling and is not canonical for M2. Single-kernel M2 bundles must use `manifest.json` schema v2 with `kernels.length == 1`, and all QASM checks must use `kernels[].qasm_path`.
