# m5-04-copy-planner-v1: VC4Tile copy planner v1

## Intent
Implement the first VC4Tile copy planner and ergonomic movement ops for global/register/shared movement, views, subviews, transpose views, and shared tile allocation.

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
compiler/include/vc4/Dialect/VC4Tile/IR/VC4TileOps.td
compiler/lib/Dialect/VC4Tile/IR/VC4TileOps.cpp
compiler/lib/Conversion/VC4TileToSSAVC4/VC4TileToSSAVC4.cpp
compiler/test/Dialect/VC4Tile/tile-load-store-roundtrip.mlir
compiler/test/Dialect/VC4Tile/copy-tile-roundtrip.mlir
compiler/test/Dialect/VC4Tile/tile-view-roundtrip.mlir
compiler/test/Dialect/VC4Tile/copy-planner-invalid.mlir
compiler/test/Conversion/VC4TileToSSAVC4/plan-copy-global-register.mlir
compiler/test/Conversion/VC4TileToSSAVC4/plan-copy-register-global.mlir
compiler/test/Conversion/VC4TileToSSAVC4/plan-copy-register-shared.mlir
compiler/test/Conversion/VC4TileToSSAVC4/plan-copy-shared-register.mlir
compiler/test/Conversion/VC4TileToSSAVC4/plan-transpose-view.mlir
compiler/test/Conversion/VC4TileToSSAVC4/reject-unplanned-copy-before-core.mlir
compiler/test/Conversion/VC4TileToSSAVC4/reject-sub32-copy-m5.mlir
compiler/test/CodeGen/VC4Tile/Hardware/Run/tile_load_store_1d_vc4tile/input.mlir
compiler/test/CodeGen/VC4Tile/Hardware/Run/tile_load_store_1d_vc4tile/expected.json
compiler/test/CodeGen/VC4Tile/Hardware/Run/tile_load_store_1d_vc4tile/candidate/tile_load_store_1d_vc4tile_candidate_harness.c
compiler/test/CodeGen/VC4Tile/Hardware/Run/register_shared_roundtrip_vc4tile/input.mlir
compiler/test/CodeGen/VC4Tile/Hardware/Run/register_shared_roundtrip_vc4tile/expected.json
compiler/test/CodeGen/VC4Tile/Hardware/Run/register_shared_roundtrip_vc4tile/candidate/register_shared_roundtrip_vc4tile_candidate_harness.c
compiler/test/CodeGen/VC4Tile/Hardware/Run/shared_register_roundtrip_vc4tile/input.mlir
compiler/test/CodeGen/VC4Tile/Hardware/Run/shared_register_roundtrip_vc4tile/expected.json
compiler/test/CodeGen/VC4Tile/Hardware/Run/shared_register_roundtrip_vc4tile/candidate/shared_register_roundtrip_vc4tile_candidate_harness.c
```

## Verification expectations

Run this slice's typed verifier gate. For executable features, run candidate generate/assemble/build/run and expected-json checks. `ninja -C compiler/build check-vc4` must remain green.

## Final response format

Return only the downloadable artifact bundle required by the current prompt's output contract.
