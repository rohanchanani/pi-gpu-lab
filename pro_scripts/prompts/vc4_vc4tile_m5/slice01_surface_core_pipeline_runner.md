# m5-01-surface-core-pipeline-and-runner: Surface/core pipeline and VC4Tile candidate runner integration

## Intent
Add/register canonical M5 surface/core passes and integrate the VC4Tile candidate runner so ergonomic inputs flow through surface canonicalization, copy planning, core CFG legalization, core verification, VC4Tile-to-SSAVC4, SSAVC4-to-VC4, and bundle emission.

## Required context to use

Read `pro_scripts/MISTAKES.md`, `compiler/docs/codegen/vc4tile-m5-full-design.md`, and the relevant companion M5 design documents before editing. Use the slice context profile; do not guess missing compiler details.

## Hard constraints

- Preserve the required M5 path.
- Keep executable semantics 32-bit-only.
- Add targeted tests for every addition/update.
- Use real hardware verification for executable behavior.
- Do not satisfy verifier scans using comments or dummy strings.


## Required implementation details for this slice

This slice is intentionally a surface/core boundary slice before real ergonomic tile-load/tile-store/copy ops exist. To make the boundary testable without implementing later M5 features, add exactly one temporary non-executable surface sentinel op:

```text
vc4tile.surface_placeholder
```

Rules for this sentinel:

- It is a surface op, not a core op.
- `--canonicalize-vc4tile-surface` must erase `vc4tile.surface_placeholder` and leave valid VC4Tile core.
- `--plan-vc4tile-copies` may be a semantic no-op in this slice, but it must be registered and ordered in the runner.
- `--verify-vc4tile-core` must reject `vc4tile.surface_placeholder` if it reaches the core verifier.
- `--convert-vc4tile-to-ssavc4` must reject `vc4tile.surface_placeholder` if it is run directly on raw surface input. The diagnostic must mention `surface operation` and `--canonicalize-vc4tile-surface`.
- Do not make `surface_placeholder` lower to SSAVC4, scheduled VC4, QASM, or runtime artifacts. It exists only to verify the surface/core ordering boundary until later slices add real surface operations.

The direct rejection test must invoke the conversion pass, not merely parse the MLIR. A bare `vc4-opt file.mlir` is not a valid rejection test.

## Exact expected GPT output / source products

The implementation bundle for this slice must include exactly the repo-relative files needed to satisfy these source-product expectations. Do not rename these files without updating the worklist, verifier spec, and this prompt in the same bundle.

```text
compiler/include/vc4/Conversion/VC4TileToSSAVC4/VC4TileToSSAVC4.h
compiler/lib/Conversion/VC4TileToSSAVC4/VC4TileToSSAVC4.cpp
compiler/tools/vc4-opt/vc4-opt.cpp
compiler/test/Conversion/VC4TileToSSAVC4/reject-surface-before-canonicalize.mlir
compiler/test/Conversion/VC4TileToSSAVC4/pass-pipeline-empty-surface.mlir
compiler/test/Dialect/VC4Tile/verify-core-reject-surface-placeholder.mlir
compiler/test/CodeGen/VC4Tile/Emit/pipeline-minimal-surface-vc4tile.mlir
compiler/test/CodeGen/VC4Tile/Support/run_vc4tile_candidate_codegen_test.sh
```

## Verification expectations

Run this slice's typed verifier gate. For executable features, run candidate generate/assemble/build/run and expected-json checks. `ninja -C compiler/build check-vc4` must remain green.

## Final response format

Return only the downloadable artifact bundle required by the current prompt's output contract.
