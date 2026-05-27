# m5-02-tile-layout-types-and-precision-markers: Tile/layout types and 32-bit precision markers

## Intent
Define tile/layout/role/precision vocabulary for ergonomic VC4Tile, with 32-bit-only executable semantics and strict diagnostics for sub-32 forms.

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
compiler/include/vc4/Dialect/VC4Tile/IR/VC4TileTypes.td
compiler/include/vc4/Dialect/VC4Tile/IR/VC4TileAttrs.td
compiler/include/vc4/Dialect/VC4Tile/IR/VC4TileOps.td
compiler/include/vc4/Dialect/VC4Tile/IR/VC4TileTypes.h
compiler/lib/Dialect/VC4Tile/IR/VC4TileTypes.cpp
compiler/lib/Dialect/VC4Tile/IR/VC4TileOps.cpp
compiler/test/Dialect/VC4Tile/tile-types-layouts-roundtrip.mlir
compiler/test/Dialect/VC4Tile/tile-types-layouts-invalid.mlir
compiler/test/Dialect/VC4Tile/tile-precision-markers-roundtrip.mlir
compiler/test/Dialect/VC4Tile/tile-precision-markers-invalid.mlir
compiler/test/Dialect/VC4Tile/tile-layout-affine-invalid.mlir
```

## Verification expectations

Run this slice's typed verifier gate. For executable features, run candidate generate/assemble/build/run and expected-json checks. `ninja -C compiler/build check-vc4` must remain green.

## Final response format

Return only the downloadable artifact bundle required by the current prompt's output contract.
