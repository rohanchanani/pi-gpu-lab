# m5-05-global-register-tile-load-store: Global/register tile load-store ergonomics

## Intent
Prove ergonomic global/register tile load/store patterns with tail masks, 2D row-major indexing, and supported affine stride behavior on real hardware.

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
compiler/test/Conversion/VC4TileToSSAVC4/plan-copy-1d-tail.mlir
compiler/test/Conversion/VC4TileToSSAVC4/plan-copy-2d-row-major.mlir
compiler/test/Conversion/VC4TileToSSAVC4/plan-copy-affine-stride.mlir
compiler/test/CodeGen/VC4Tile/Hardware/Run/tile_load_1d_tail_vc4tile/input.mlir
compiler/test/CodeGen/VC4Tile/Hardware/Run/tile_load_1d_tail_vc4tile/expected.json
compiler/test/CodeGen/VC4Tile/Hardware/Run/tile_load_1d_tail_vc4tile/candidate/tile_load_1d_tail_vc4tile_candidate_harness.c
compiler/test/CodeGen/VC4Tile/Hardware/Run/tile_store_1d_tail_vc4tile/input.mlir
compiler/test/CodeGen/VC4Tile/Hardware/Run/tile_store_1d_tail_vc4tile/expected.json
compiler/test/CodeGen/VC4Tile/Hardware/Run/tile_store_1d_tail_vc4tile/candidate/tile_store_1d_tail_vc4tile_candidate_harness.c
compiler/test/CodeGen/VC4Tile/Hardware/Run/tile_load_store_2d_row_major_vc4tile/input.mlir
compiler/test/CodeGen/VC4Tile/Hardware/Run/tile_load_store_2d_row_major_vc4tile/expected.json
compiler/test/CodeGen/VC4Tile/Hardware/Run/tile_load_store_2d_row_major_vc4tile/candidate/tile_load_store_2d_row_major_vc4tile_candidate_harness.c
compiler/test/CodeGen/VC4Tile/Hardware/Run/tile_load_store_affine_stride_vc4tile/input.mlir
compiler/test/CodeGen/VC4Tile/Hardware/Run/tile_load_store_affine_stride_vc4tile/expected.json
compiler/test/CodeGen/VC4Tile/Hardware/Run/tile_load_store_affine_stride_vc4tile/candidate/tile_load_store_affine_stride_vc4tile_candidate_harness.c
```

## Verification expectations

Run this slice's typed verifier gate. For executable features, run candidate generate/assemble/build/run and expected-json checks. `ninja -C compiler/build check-vc4` must remain green.

## Final response format

Return only the downloadable artifact bundle required by the current prompt's output contract.
