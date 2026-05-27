# m5-03-vdr-vcd-global-to-vpm-load-support: SSAVC4 VDR/VCD global-to-VPM load support

## Intent
Add lower-half SSAVC4 VDR/VCD global-to-VPM load support so the M5 copy planner can choose hardware-natural global-to-shared paths.

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
compiler/include/vc4/Dialect/SSAVC4/IR/SSAVC4Ops.td
compiler/lib/Dialect/SSAVC4/IR/SSAVC4Ops.cpp
compiler/lib/Conversion/SSAVC4ToVC4/SSAVC4ToVC4.cpp
compiler/test/Dialect/SSAVC4/vdr-load-roundtrip.mlir
compiler/test/Dialect/SSAVC4/vdr-load-invalid.mlir
compiler/test/Conversion/SSAVC4ToVC4/vdr-load.mlir
compiler/test/Conversion/SSAVC4ToVC4/vdr-load-invalid.mlir
compiler/test/CodeGen/SSAVC4/Emit/vdr-load-ssavc4.mlir
compiler/test/CodeGen/SSAVC4/Hardware/Run/vdr_load_roundtrip_ssavc4/input.mlir
compiler/test/CodeGen/SSAVC4/Hardware/Run/vdr_load_roundtrip_ssavc4/expected.json
compiler/test/CodeGen/SSAVC4/Hardware/Run/vdr_load_roundtrip_ssavc4/candidate/vdr_load_roundtrip_ssavc4_candidate_harness.c
```

## Verification expectations

Run this slice's typed verifier gate. For executable features, run candidate generate/assemble/build/run and expected-json checks. `ninja -C compiler/build check-vc4` must remain green.

## Final response format

Return only the downloadable artifact bundle required by the current prompt's output contract.
