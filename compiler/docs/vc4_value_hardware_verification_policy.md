# VC4 Value Hardware Verification Policy

## Purpose

This document records the verification doctrine for executable VC4 value-layer
lowering. It applies to the handwritten value path:

```text
func + tiny vc4value + vector + memref + arith
  -> vc4kernel
  -> ssavc4
  -> scheduled vc4
  -> artifacts/runtime
  -> real VC4 hardware
```

It does not implement lowering, add fixtures, or claim Triton support.

## Static Tests Are Not Enough

Static tests are required but insufficient. Parser, verifier, conversion, and
audit tests prove syntax, diagnostics, lowering shape, and reject policy. They
do not prove executable semantics.

A value feature is executable-verified only when a kernel using that feature is
lowered through VC4Kernel, SSAVC4, scheduled VC4, emitted artifacts, and the
runtime, then run on real VC4 hardware against a strict CPU oracle with sentinel
checks.

## Isolation Fixtures

Every new executable value-lowering feature band needs targeted isolation
hardware fixtures. These fixtures should be designed to break the feature they
prove.

Isolation fixtures must:

- use source value IR, not source-authored VC4Kernel or lower-half IR;
- generate fresh candidates through the value-to-VC4Kernel path;
- sweep meaningful sizes, tails, masks, values, alignments, signs, and dynamic
  parameters for the feature;
- compare device results against strict host CPU oracles;
- include sentinels before and after checked output regions;
- prove computation happened on the device, not in generated host code;
- remain in the repo for future triage.

Isolation fixtures are feature proofs and debugging tools. They are not a
replacement for routine final mixed acceptance.

## Mixed Fixtures

Routine final acceptance for executable value phases uses mixed fixtures.
Mixed fixtures should combine implemented features and look increasingly like
real kernels over time.

Early Phase 5 mixed fixtures should combine launch identity, lane/offset math,
tail masks, `vector.transfer_read`, `vector.transfer_write`, i32/f32 ALU,
compare/select, TMU safe offsets, and VDW inactive preserve. Later mixed
fixtures should evolve toward realistic patterns such as elementwise add,
saxpy, masked activation, reductions, GEMV, GEMM, subword kernels, approximate
math kernels, and cooperative/tiled kernels.

Passing isolated fixtures does not allow skipping mixed acceptance.

## Claim Manifests And Audits

Every mixed fixture `saw_*`, `no_*`, or equivalent claim must be auditable.
Claims must be backed by one of:

```text
CHECKED_OUTPUT
CHECKED_AUDIT
PHASE_GUARD
```

Claims may not be based on fixture names, public names, paths, comments, status
strings, dead IR, generated-output path checks, or unverified assumptions.

Value mixed fixtures must have a source-controlled manifest and claim audit
analogous to:

```text
compiler/test/CodeGen/VC4Kernel/Hardware/MixedAcceptance/mixed_acceptance_manifest.json
compiler/test/CodeGen/VC4Kernel/Hardware/MixedAcceptance/mixed_fixture_claims.json
compiler/test/CodeGen/VC4Kernel/Hardware/MixedAcceptance/audit_mixed_fixture_claims.py
```

## Runner Discipline

Value hardware runners must follow the existing project runner discipline:

- generate fresh candidates from source;
- do not use stale `.vc4_auto` bundles;
- lower only through `vc4kernel -> ssavc4 -> scheduled vc4`;
- use the existing strict power-cycle-before-run policy;
- use the standard 60s attempt timeout unless a fixture documents a justified
  exception;
- treat after-boot mismatches, sentinel failures, launch failures, and timeouts
  as real failures;
- require `VC4_TEST_RESULT status=PASS` and zero mismatch, sentinel, and launch
  failure fields where present.

CPU oracles, sentinels, expected JSON, result checkers, and power-cycle
discipline are part of the proof and must not be weakened.

## Debugging Failed Mixed Fixtures

When a value mixed fixture fails:

1. Identify the feature bands involved.
2. Run isolation fixtures for those feature bands.
3. If isolation passes, reduce the mixed fixture by removing or swapping one
   feature at a time while preserving the failing structure.
4. Inspect each layer: value IR, VC4Kernel, SSAVC4, scheduled VC4, emitted
   QASM/C/manifest, and runtime uniform/resource packing.
5. Instrument hypotheses when needed with temporary diagnostic stores or
   intermediate-stage reruns.
6. Revert temporary diagnostic edits before commit.

Do not reshape natural value kernels to hide compiler or lower-half bugs.

## Lower-Half Escalation

If debugging shows a real bug in VC4Kernel, SSAVC4, scheduled VC4, artifact
emission, or runtime behavior, report it as a blocker unless the current phase
explicitly authorizes fixing that layer.

The value layer must not paper over lower-half bugs by weakening kernels,
oracles, masks, dimensions, sentinels, or diagnostics.
