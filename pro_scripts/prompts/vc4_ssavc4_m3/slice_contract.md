# VC4 SSAVC4 M3 Slice Contract

Each slice has two sources of truth:

1. The natural-language expected end state in `pro_scripts/vc4_ssavc4_m3_worklist.json` and the slice prompt.
2. The executable typed checks in `pro_scripts/vc4_ssavc4_m3_verifications.json`.

A slice is incomplete until both are satisfied. Do not satisfy a verifier by weakening the verifier, changing an oracle, bypassing lowering, or substituting reference artifacts.

## Required M3 flow

```text
ssavc4 input
  -> vc4-opt --convert-ssavc4-to-vc4
  -> scheduled vc4.module/vc4.func/vc4.qpu.* output
  -> existing scheduled VC4 verifiers
  -> existing vc4-codegen bundle emission
  -> M2-compatible manifest/layout/kernel_launch/shader artifacts
  -> existing libpi-backed runtime path
```

The active scheduled `vc4` sink must remain scheduled-only. SSAVC4 must be separate. M3 must not implement `gpu -> ssavc4`.

## Verification expectations

- Committed slices must keep `ninja -C compiler/build check-vc4` green.
- Future-slice tests must not be checked into active lit paths early. Add tests in the same slice that implements the corresponding op, type, lowering, or fixture.
- No expected-red tests may live under global `check-vc4`.
- Dialect slices must have parser/printer/roundtrip tests and invalid verifier tests.
- Pure ops must be pure; hardware-state ops must carry effects and/or explicit tokens.
- Lowering slices must produce scheduled `vc4` that passes existing scheduled verifier passes.
- Fixture slices must generate M2-compatible artifact bundles, assemble QASM, build candidates, and run hardware checks when hardware is required.
- Final M3 acceptance must run the full generic M2 verifier to prove M2 persistence.

## Failure classification

- TableGen/CMake/build failures: fix mechanically; do not rewrite scope.
- Parser/printer/verifier failures: fix the dialect contract or invalid tests.
- Scheduled verifier failures: fix lowering, scheduling, register allocation, branch layout, or hazard insertion.
- Artifact shape failures: fix the SSAVC4 wrapper/lowering so existing M2 emitter output stays canonical.
- Hardware semantic failures: inspect generated scheduled output/QASM/runtime logs; do not fake results or edit expected JSON.
- M2 regression failures: treat as a blocker; M3 is cumulative.
