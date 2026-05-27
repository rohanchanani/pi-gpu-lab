# m5-07-scf-composition-with-ergonomic-ops: SCF composition with ergonomic tile ops

## Intent
Prove supported scf.for/scf.if composition with ergonomic tile ops; surface ops must plan first and SCF/index must disappear before SSAVC4 lowering.

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
compiler/test/Conversion/VC4TileToSSAVC4/scf-tiled-copy-loop.mlir
compiler/test/Conversion/VC4TileToSSAVC4/scf-tiled-copy-nondiv-trip.mlir
compiler/test/Conversion/VC4TileToSSAVC4/scf-tiled-transpose-loop.mlir
compiler/test/CodeGen/VC4Tile/Emit/scf-tiled-copy-loop-vc4tile.mlir
compiler/test/CodeGen/VC4Tile/Hardware/Run/scf_tiled_copy_loop_vc4tile/input.mlir
compiler/test/CodeGen/VC4Tile/Hardware/Run/scf_tiled_copy_loop_vc4tile/expected.json
compiler/test/CodeGen/VC4Tile/Hardware/Run/scf_tiled_copy_loop_vc4tile/candidate/scf_tiled_copy_loop_vc4tile_candidate_harness.c
compiler/test/CodeGen/VC4Tile/Hardware/Run/scf_tiled_copy_zero_trip_vc4tile/input.mlir
compiler/test/CodeGen/VC4Tile/Hardware/Run/scf_tiled_copy_zero_trip_vc4tile/expected.json
compiler/test/CodeGen/VC4Tile/Hardware/Run/scf_tiled_copy_zero_trip_vc4tile/candidate/scf_tiled_copy_zero_trip_vc4tile_candidate_harness.c
compiler/test/CodeGen/VC4Tile/Hardware/Run/scf_tiled_copy_non_divisible_trip_vc4tile/input.mlir
compiler/test/CodeGen/VC4Tile/Hardware/Run/scf_tiled_copy_non_divisible_trip_vc4tile/expected.json
compiler/test/CodeGen/VC4Tile/Hardware/Run/scf_tiled_copy_non_divisible_trip_vc4tile/candidate/scf_tiled_copy_non_divisible_trip_vc4tile_candidate_harness.c
compiler/test/CodeGen/VC4Tile/Hardware/Run/scf_tiled_transpose_loop_vc4tile/input.mlir
compiler/test/CodeGen/VC4Tile/Hardware/Run/scf_tiled_transpose_loop_vc4tile/expected.json
compiler/test/CodeGen/VC4Tile/Hardware/Run/scf_tiled_transpose_loop_vc4tile/candidate/scf_tiled_transpose_loop_vc4tile_candidate_harness.c
compiler/test/CodeGen/VC4Tile/Hardware/Run/scf_tiled_saxpy_loop_vc4tile/input.mlir
compiler/test/CodeGen/VC4Tile/Hardware/Run/scf_tiled_saxpy_loop_vc4tile/expected.json
compiler/test/CodeGen/VC4Tile/Hardware/Run/scf_tiled_saxpy_loop_vc4tile/candidate/scf_tiled_saxpy_loop_vc4tile_candidate_harness.c
```

## Verification expectations

Run this slice's typed verifier gate. For executable features, run candidate generate/assemble/build/run and expected-json checks. `ninja -C compiler/build check-vc4` must remain green.

## Final response format

Return only the downloadable artifact bundle required by the current prompt's output contract.
