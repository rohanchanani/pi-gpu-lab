# VC4Value Phase 5 Mixed Acceptance

This directory defines the routine mixed hardware acceptance gate for the
Phase 5 value-layer executable slice.

The suite starts from `func` + `vc4value` + `vector` + `memref` + `arith`
inputs and runs them through:

```text
value IR -> vc4kernel -> ssavc4 -> scheduled vc4 -> generated runtime -> hardware
```

Isolation fixtures remain the first triage tool for feature-specific failures.
This mixed suite is cumulative: fixtures combine launch identity, lane math,
TMU transfer reads, VDW tail-preserve writes, f32 arithmetic, i32 predicates,
finite f32 compare/select, and strict sentinel checks.

Run:

```bash
VC4_CODEGEN_STATE_ROOT=.vc4_auto/codegen_phase5j_value_mixed \
  compiler/test/CodeGen/VC4Value/Hardware/MixedAcceptance/run_value_mixed_acceptance.sh \
  --manifest compiler/test/CodeGen/VC4Value/Hardware/MixedAcceptance/mixed_value_acceptance_manifest.json
```

Before hardware, run the coverage checker and claim audit. Every `saw_*` or
`no_*` result claim emitted by a mixed fixture must be listed in
`mixed_value_fixture_claims.json` and backed by checked output, checked audit,
or an explicit phase guard.
