# m5-11-tile-contract-dot-matmul: Tile contract, dot, and matmul

## Intent
Add tile_contract, tile_dot, and tile_matmul as future producer-facing ergonomic surface contracts and lower small 32-bit forms to movement, elementwise, and reduction operations.

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
compiler/test/Dialect/VC4Tile/tile-contract-roundtrip.mlir
compiler/test/Dialect/VC4Tile/tile-contract-invalid.mlir
compiler/test/Conversion/VC4TileToSSAVC4/tile-dot-lowering.mlir
compiler/test/Conversion/VC4TileToSSAVC4/tile-matmul-lowering.mlir
compiler/test/Conversion/VC4TileToSSAVC4/tile-contract-lowering.mlir
compiler/test/CodeGen/VC4Tile/Emit/tile-dot-vc4tile.mlir
compiler/test/CodeGen/VC4Tile/Emit/tile-matmul-vc4tile.mlir
compiler/test/CodeGen/VC4Tile/Hardware/Run/tile_dot_1x16_i32_vc4tile/input.mlir
compiler/test/CodeGen/VC4Tile/Hardware/Run/tile_dot_1x16_i32_vc4tile/expected.json
compiler/test/CodeGen/VC4Tile/Hardware/Run/tile_dot_1x16_i32_vc4tile/candidate/tile_dot_1x16_i32_vc4tile_candidate_harness.c
compiler/test/CodeGen/VC4Tile/Hardware/Run/tile_matmul_4x4_i32_vc4tile/input.mlir
compiler/test/CodeGen/VC4Tile/Hardware/Run/tile_matmul_4x4_i32_vc4tile/expected.json
compiler/test/CodeGen/VC4Tile/Hardware/Run/tile_matmul_4x4_i32_vc4tile/candidate/tile_matmul_4x4_i32_vc4tile_candidate_harness.c
compiler/test/CodeGen/VC4Tile/Hardware/Run/tile_contract_shared_rhs_vc4tile/input.mlir
compiler/test/CodeGen/VC4Tile/Hardware/Run/tile_contract_shared_rhs_vc4tile/expected.json
compiler/test/CodeGen/VC4Tile/Hardware/Run/tile_contract_shared_rhs_vc4tile/candidate/tile_contract_shared_rhs_vc4tile_candidate_harness.c
```

## Verification expectations

Run this slice's typed verifier gate. For executable features, run candidate generate/assemble/build/run and expected-json checks. `ninja -C compiler/build check-vc4` must remain green.

## Final response format

Return only the downloadable artifact bundle required by the current prompt's output contract.
