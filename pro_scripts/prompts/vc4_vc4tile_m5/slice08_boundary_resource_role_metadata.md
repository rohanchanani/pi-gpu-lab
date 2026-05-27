# m5-08-boundary-resource-role-metadata: Boundary, resource, and role metadata

## Intent
Add semantic metadata for roles, boundary policies, resource intent, copy stages, and reuse hints; consume, preserve, or reject every accepted field deterministically.

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
compiler/test/Dialect/VC4Tile/role-metadata-roundtrip.mlir
compiler/test/Dialect/VC4Tile/role-metadata-invalid.mlir
compiler/test/Dialect/VC4Tile/boundary-policy-roundtrip.mlir
compiler/test/Dialect/VC4Tile/boundary-policy-invalid.mlir
compiler/test/Dialect/VC4Tile/resource-intent-roundtrip.mlir
compiler/test/Dialect/VC4Tile/resource-intent-invalid.mlir
compiler/test/Conversion/VC4TileToSSAVC4/role-boundary-metadata-preserve.mlir
compiler/test/CodeGen/VC4Tile/Hardware/Run/boundary_tail_predicated_load_store_vc4tile/input.mlir
compiler/test/CodeGen/VC4Tile/Hardware/Run/boundary_tail_predicated_load_store_vc4tile/expected.json
compiler/test/CodeGen/VC4Tile/Hardware/Run/boundary_tail_predicated_load_store_vc4tile/candidate/boundary_tail_predicated_load_store_vc4tile_candidate_harness.c
```

## Verification expectations

Run this slice's typed verifier gate. For executable features, run candidate generate/assemble/build/run and expected-json checks. `ninja -C compiler/build check-vc4` must remain green.

## Final response format

Return only the downloadable artifact bundle required by the current prompt's output contract.
