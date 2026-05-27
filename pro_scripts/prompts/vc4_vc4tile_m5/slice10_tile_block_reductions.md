# m5-10-tile-and-block-reductions: Tile and block reductions

## Intent
Add ergonomic tile/warp/block reductions that lower to existing rotate/reduce/shared/barrier mechanisms with input-dependent hardware proofs.

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
compiler/test/Dialect/VC4Tile/tile-reductions-roundtrip.mlir
compiler/test/Dialect/VC4Tile/tile-reductions-invalid.mlir
compiler/test/Conversion/VC4TileToSSAVC4/tile-reduce-lowering.mlir
compiler/test/Conversion/VC4TileToSSAVC4/block-reduce-lowering.mlir
compiler/test/CodeGen/VC4Tile/Emit/tile-reduce-vc4tile.mlir
compiler/test/CodeGen/VC4Tile/Hardware/Run/tile_reduce_sum_i32_vc4tile/input.mlir
compiler/test/CodeGen/VC4Tile/Hardware/Run/tile_reduce_sum_i32_vc4tile/expected.json
compiler/test/CodeGen/VC4Tile/Hardware/Run/tile_reduce_sum_i32_vc4tile/candidate/tile_reduce_sum_i32_vc4tile_candidate_harness.c
compiler/test/CodeGen/VC4Tile/Hardware/Run/warp_reduce_sum_ergonomic_vc4tile/input.mlir
compiler/test/CodeGen/VC4Tile/Hardware/Run/warp_reduce_sum_ergonomic_vc4tile/expected.json
compiler/test/CodeGen/VC4Tile/Hardware/Run/warp_reduce_sum_ergonomic_vc4tile/candidate/warp_reduce_sum_ergonomic_vc4tile_candidate_harness.c
compiler/test/CodeGen/VC4Tile/Hardware/Run/block_reduce_sum_ergonomic_vc4tile/input.mlir
compiler/test/CodeGen/VC4Tile/Hardware/Run/block_reduce_sum_ergonomic_vc4tile/expected.json
compiler/test/CodeGen/VC4Tile/Hardware/Run/block_reduce_sum_ergonomic_vc4tile/candidate/block_reduce_sum_ergonomic_vc4tile_candidate_harness.c
compiler/test/CodeGen/VC4Tile/Hardware/Run/block_reduce_tail_sum_vc4tile/input.mlir
compiler/test/CodeGen/VC4Tile/Hardware/Run/block_reduce_tail_sum_vc4tile/expected.json
compiler/test/CodeGen/VC4Tile/Hardware/Run/block_reduce_tail_sum_vc4tile/candidate/block_reduce_tail_sum_vc4tile_candidate_harness.c
```

## Verification expectations

Run this slice's typed verifier gate. For executable features, run candidate generate/assemble/build/run and expected-json checks. `ninja -C compiler/build check-vc4` must remain green.

## Final response format

Return only the downloadable artifact bundle required by the current prompt's output contract.
