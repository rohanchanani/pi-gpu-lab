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
