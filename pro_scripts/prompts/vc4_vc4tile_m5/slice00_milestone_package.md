# m5-00-milestone-package: M5 milestone package

## Intent
Install and validate the M5 descriptor, worklist, verification spec, context profiles, prompt package, design documents, and mechanism smoke. No compiler implementation source is allowed in this slice.

## Required context to use

Read `pro_scripts/MISTAKES.md`, `compiler/docs/codegen/vc4tile-m5-full-design.md`, and the relevant companion M5 design documents before editing. Use the slice context profile; do not guess missing compiler details.

## Hard constraints

- Preserve the required M5 path.
- Keep executable semantics 32-bit-only.
- Add targeted tests for every addition/update.
- Use real hardware verification for executable behavior.
- Do not satisfy verifier scans using comments or dummy strings.

## Exact expected GPT output / source products

The implementation bundle for this slice must include exactly the repo-relative files needed to satisfy these source-product expectations. Do not rename these files without updating the worklist, verifier spec, and this prompt in the same bundle.

```text
pro_scripts/milestones/vc4-vc4tile-m5.json
pro_scripts/vc4_vc4tile_m5_worklist.json
pro_scripts/vc4_vc4tile_m5_verifications.json
pro_scripts/vc4_vc4tile_m5_context_profiles.json
pro_scripts/vc4_vc4tile_m5_mechanism_smoke.json
pro_scripts/m5_mechanism_smoke/surface_core_ok.mlir
pro_scripts/m5_mechanism_smoke/copy_plan_core_ok.mlir
pro_scripts/m5_mechanism_smoke/copy-plan.json
pro_scripts/m5_mechanism_smoke/precision_policy_ok.txt
pro_scripts/m5_mechanism_smoke/run_vc4tile_candidate_codegen_test_smoke.sh
pro_scripts/MISTAKES.md
pro_scripts/prompts/vc4_vc4tile_m5/README.md
pro_scripts/prompts/vc4_vc4tile_m5/constitution.md
pro_scripts/prompts/vc4_vc4tile_m5/output_contract.md
pro_scripts/prompts/vc4_vc4tile_m5/slice_contract.md
pro_scripts/prompts/vc4_vc4tile_m5/m5_handoff.md
pro_scripts/prompts/vc4_vc4tile_m5/codex_contract.md
pro_scripts/prompts/vc4_vc4tile_m5/implementation_integrity_audit_prompt.md
pro_scripts/prompts/vc4_vc4tile_m5/gpt_slice_prompt.md.j2
pro_scripts/prompts/vc4_vc4tile_m5/gpt_failure_prompt.md.j2
pro_scripts/prompts/vc4_vc4tile_m5/gpt_diagnosis_prompt.md.j2
pro_scripts/prompts/vc4_vc4tile_m5/codex_mechanical_prompt.md.j2
pro_scripts/prompts/vc4_vc4tile_m5/slice00_milestone_package.md
pro_scripts/prompts/vc4_vc4tile_m5/slice01_surface_core_pipeline_runner.md
pro_scripts/prompts/vc4_vc4tile_m5/slice02_tile_layout_precision.md
pro_scripts/prompts/vc4_vc4tile_m5/slice03_vdr_vcd_global_to_vpm.md
pro_scripts/prompts/vc4_vc4tile_m5/slice04_copy_planner_v1.md
pro_scripts/prompts/vc4_vc4tile_m5/slice05_global_register_tile_load_store.md
pro_scripts/prompts/vc4_vc4tile_m5/slice06_shared_vpm_copy_transpose.md
pro_scripts/prompts/vc4_vc4tile_m5/slice07_scf_composition.md
pro_scripts/prompts/vc4_vc4tile_m5/slice08_boundary_resource_role_metadata.md
pro_scripts/prompts/vc4_vc4tile_m5/slice09_elementwise_tile_compute.md
pro_scripts/prompts/vc4_vc4tile_m5/slice10_tile_block_reductions.md
pro_scripts/prompts/vc4_vc4tile_m5/slice11_tile_contract_dot_matmul.md
pro_scripts/prompts/vc4_vc4tile_m5/slice12_final_acceptance.md
compiler/docs/codegen/vc4tile-m5-copy-planner-design.md
compiler/docs/codegen/vc4tile-m5-compute-primitives-design.md
compiler/docs/codegen/vc4tile-m5-precision-roadmap.md
compiler/docs/codegen/vc4tile-m5-full-design.md
compiler/docs/codegen/vc4tile-m5-plan.md
```

## Verification expectations

Run this slice's typed verifier gate. For executable features, run candidate generate/assemble/build/run and expected-json checks. `ninja -C compiler/build check-vc4` must remain green.

## Final response format

Return only the downloadable artifact bundle required by the current prompt's output contract.
