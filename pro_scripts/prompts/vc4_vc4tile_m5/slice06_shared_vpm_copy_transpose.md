# m5-06-shared-vpm-copy-and-transpose: Shared VPM copy and ergonomic transpose

## Intent
Make shared VPM copy and shared transpose ergonomic while lowering to concrete VPM/VDR/VDW/barrier paths.

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
compiler/test/Dialect/VC4Tile/shared-tile-copy-roundtrip.mlir
compiler/test/Dialect/VC4Tile/shared-tile-copy-invalid.mlir
compiler/test/Conversion/VC4TileToSSAVC4/plan-copy-global-shared-vdr.mlir
compiler/test/Conversion/VC4TileToSSAVC4/plan-shared-transpose-view.mlir
compiler/test/CodeGen/VC4Tile/Emit/shared-transpose-ergonomic-vc4tile.mlir
compiler/test/CodeGen/VC4Tile/Hardware/Run/shared_tile_roundtrip_vc4tile/input.mlir
compiler/test/CodeGen/VC4Tile/Hardware/Run/shared_tile_roundtrip_vc4tile/expected.json
compiler/test/CodeGen/VC4Tile/Hardware/Run/shared_tile_roundtrip_vc4tile/candidate/shared_tile_roundtrip_vc4tile_candidate_harness.c
compiler/test/CodeGen/VC4Tile/Hardware/Run/global_to_shared_to_global_2d_vc4tile/input.mlir
compiler/test/CodeGen/VC4Tile/Hardware/Run/global_to_shared_to_global_2d_vc4tile/expected.json
compiler/test/CodeGen/VC4Tile/Hardware/Run/global_to_shared_to_global_2d_vc4tile/candidate/global_to_shared_to_global_2d_vc4tile_candidate_harness.c
compiler/test/CodeGen/VC4Tile/Hardware/Run/shared_transpose_16x16_ergonomic_vc4tile/input.mlir
compiler/test/CodeGen/VC4Tile/Hardware/Run/shared_transpose_16x16_ergonomic_vc4tile/expected.json
compiler/test/CodeGen/VC4Tile/Hardware/Run/shared_transpose_16x16_ergonomic_vc4tile/candidate/shared_transpose_16x16_ergonomic_vc4tile_candidate_harness.c
compiler/test/CodeGen/VC4Tile/Hardware/Run/shared_transpose_store_global_vc4tile/input.mlir
compiler/test/CodeGen/VC4Tile/Hardware/Run/shared_transpose_store_global_vc4tile/expected.json
compiler/test/CodeGen/VC4Tile/Hardware/Run/shared_transpose_store_global_vc4tile/candidate/shared_transpose_store_global_vc4tile_candidate_harness.c
```

## Verification expectations

Run this slice's typed verifier gate. For executable features, run candidate generate/assemble/build/run and expected-json checks. `ninja -C compiler/build check-vc4` must remain green.

## Final response format

Return only the downloadable artifact bundle required by the current prompt's output contract.
