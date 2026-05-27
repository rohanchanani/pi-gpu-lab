# VC4Tile M5 Milestone Plan

M5 is ergonomic VC4Tile only. It adds surface tile movement, layout vocabulary, copy planning, elementwise tile compute, reductions, and small 32-bit tile contraction/dot/matmul contracts while preserving this path:

```text
vc4tile ergonomic surface
  -> --canonicalize-vc4tile-surface
  -> --plan-vc4tile-copies
  -> --legalize-vc4tile-core-cfg
  -> --verify-vc4tile-core
  -> --convert-vc4tile-to-ssavc4
  -> --convert-ssavc4-to-vc4
  -> vc4-codegen --emit-bundle
  -> VC4 hardware
```

M5 is not Triton lowering, IREE lowering, StableHLO lowering, JAX/PyTorch/Torch-MLIR lowering, MLIR `gpu` lowering, or any producer-to-SSAVC4 shortcut. M5 does not implement executable sub-32 precision. Executable M5 semantics are 32-bit only.

The authoritative design documents are:

- `compiler/docs/codegen/vc4tile-m5-full-design.md`
- `compiler/docs/codegen/vc4tile-m5-copy-planner-design.md`
- `compiler/docs/codegen/vc4tile-m5-compute-primitives-design.md`
- `compiler/docs/codegen/vc4tile-m5-precision-roadmap.md`
- `pro_scripts/MISTAKES.md`

Slice order:

1. `m5-00-milestone-package`
2. `m5-01-surface-core-pipeline-and-runner`
3. `m5-02-tile-layout-types-and-precision-markers`
4. `m5-03-vdr-vcd-global-to-vpm-load-support`
5. `m5-04-copy-planner-v1`
6. `m5-05-global-register-tile-load-store`
7. `m5-06-shared-vpm-copy-and-transpose`
8. `m5-07-scf-composition-with-ergonomic-ops`
9. `m5-08-boundary-resource-role-metadata`
10. `m5-09-elementwise-tile-compute`
11. `m5-10-tile-and-block-reductions`
12. `m5-11-tile-contract-dot-matmul`
13. `m5-12-final-acceptance`

Hardware is the gold standard. Every executable ergonomic feature must have fresh candidate artifacts, VC4 hardware execution, device-to-host copyback, CPU oracle comparison, sentinel checking, and `VC4_TEST_RESULT` status derived from real mismatches and launch failures.

---

## M5 final acceptance gate

The `m5-12-final-acceptance` slice is the cumulative acceptance boundary for the M5 milestone. It does not add a producer lowering path or a direct VC4Tile-to-VC4 path. It records the acceptance contract and relies on the typed verifier to prove that the already-implemented ergonomic VC4Tile surface remains scoped, candidate-first, hardware-backed, and regression-clean.

The final gate is expected to cover the full M5 path:

```text
vc4tile ergonomic surface
  -> --canonicalize-vc4tile-surface
  -> --plan-vc4tile-copies
  -> --legalize-vc4tile-core-cfg
  -> --verify-vc4tile-core
  -> --convert-vc4tile-to-ssavc4
  -> --convert-ssavc4-to-vc4
  -> vc4-codegen --emit-bundle
  -> QASM / C / H artifacts / vc4_runtime
  -> VC4 hardware
```

Required final checks:

- `build-vc4_opt`
- `build-vc4_codegen`
- `build-check_vc4`
- all M5 targeted VC4Tile dialect, conversion, and emit lit subsets
- all M5 required candidate-first hardware fixtures
- M4 final acceptance / regression
- M3 final and cumulative regression as required by the M4 contract
- M2 final and cumulative regression as required by the M3/M4 contracts
- implementation integrity audit
- copy planner integrity scan
- precision scope scan
- producer-lowering absence scan
- surface/core ordering scan
- candidate harness evidence scan

### Required M5 hardware matrix

The final hardware matrix groups required executable M5 evidence by feature family. Every listed fixture must use freshly generated candidate artifacts by default, invoke the generated launch wrapper, copy device outputs back to the host, compare against a CPU oracle, preserve sentinel checks where applicable, and derive the result line from real launch failures and mismatches.

```json
{
  "m5_copy_planner_smoke": [
    "tile_load_store_1d_vc4tile",
    "register_shared_roundtrip_vc4tile",
    "shared_register_roundtrip_vc4tile"
  ],
  "m5_copy_global_register": [
    "tile_load_1d_tail_vc4tile",
    "tile_store_1d_tail_vc4tile",
    "tile_load_store_2d_row_major_vc4tile",
    "tile_load_store_affine_stride_vc4tile"
  ],
  "m5_copy_shared_vpm": [
    "shared_tile_roundtrip_vc4tile",
    "global_to_shared_to_global_2d_vc4tile",
    "shared_transpose_16x16_ergonomic_vc4tile",
    "shared_transpose_store_global_vc4tile"
  ],
  "m5_scf_ergonomic": [
    "scf_tiled_copy_loop_vc4tile",
    "scf_tiled_copy_zero_trip_vc4tile",
    "scf_tiled_copy_non_divisible_trip_vc4tile",
    "scf_tiled_transpose_loop_vc4tile",
    "scf_tiled_saxpy_loop_vc4tile"
  ],
  "m5_metadata_boundary": [
    "boundary_tail_predicated_load_store_vc4tile"
  ],
  "m5_elementwise": [
    "tile_elementwise_add_store_vc4tile",
    "tile_mul_store_vc4tile",
    "tile_masked_select_tail_vc4tile"
  ],
  "m5_reductions": [
    "tile_reduce_sum_i32_vc4tile",
    "warp_reduce_sum_ergonomic_vc4tile",
    "block_reduce_sum_ergonomic_vc4tile",
    "block_reduce_tail_sum_vc4tile"
  ],
  "m5_contracts": [
    "tile_dot_1x16_i32_vc4tile",
    "tile_matmul_4x4_i32_vc4tile",
    "tile_contract_shared_rhs_vc4tile"
  ]
}
```

### Anti-shortcut requirements

M5 final acceptance must reject the following shortcuts:

- compiler source branching on fixture names, public kernel names, or generated candidate names
- direct VC4Tile-to-VC4 conversion, direct VC4Tile-to-QASM emission, or any producer-to-SSAVC4/VC4 lowering path
- surviving surface ops, raw `scf.*`, `index` values, or producer dialect ops after verified VC4Tile core preparation
- executable sub-32 precision behavior; f16, bf16, fp8, fp4, int8, uint8, int4, uint4, packed, quantized, and mixed-precision forms remain metadata/diagnostic-only in M5
- fixed success result lines, stale `.vc4_auto` candidate reuse without explicit opt-in, reference-bundle edits as acceptance evidence, or host-computed outputs substituted for device copyback
- weakened M2, M3, or M4 regression contracts

### Final expected state

After this gate passes, M5 has a complete ergonomic VC4Tile surface. Surface tile movement, layout metadata, copy planning, shared VPM ergonomics, SCF composition, elementwise tile compute, reductions, and small 32-bit dot/contract/matmul forms all lower through VC4Tile core to SSAVC4 and scheduled VC4 artifacts before hardware execution. The milestone remains explicitly scoped away from producer integration and executable sub-32 precision work, which are reserved for later milestones.

