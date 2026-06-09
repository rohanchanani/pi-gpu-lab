# VC4 Vector/Triton Phase 7 TTIR Elementwise Hardware Lock

VC4_TRITON_PHASE7_ELEMENTWISE_HARDWARE_LOCKED=PENDING
VC4_TRITON_PHASE7_REAL_TTIR_TO_VALUE_LOWERING=YES
VC4_TRITON_PHASE7_TTIR_ELEMENTWISE_STATIC_PIPELINE=YES
VC4_TRITON_PHASE7_TTIR_ELEMENTWISE_ISOLATION_HARDWARE=YES
VC4_TRITON_PHASE7_TTIR_ELEMENTWISE_MIXED_HARDWARE=YES
VC4_TRITON_PHASE7_MIXED_FIXTURE_CLAIMS_AUDITED=YES
READY_FOR_PHASE8_CONTROL_FLOW_AND_LOOP_BOUNDARY=PENDING
READY_FOR_TRITON=NO

## Scope

Phase 7 proves the first real TTIR executable path for the narrow elementwise
V1 profile:

```text
real Triton source
  -> real emitted TTIR snapshot
  -> vc4-triton-import --mode lower-elementwise-v1
  -> standard VC4 value IR
  -> vc4kernel
  -> ssavc4
  -> scheduled vc4
  -> generated artifacts/runtime
  -> real VC4 hardware
```

The proof source is the checked-in Phase 6 Triton 3.7.0 TTIR corpus:

- `examples/triton/phase6/generated/vector_add_b16.ttir.mlir`;
- `examples/triton/phase6/generated/saxpy_select_b16.ttir.mlir`;
- `examples/triton/phase6/generated/i32_add_select_b16.ttir.mlir`.

## Proven Gates

- Static TTIR-to-value and value-to-vc4kernel tests exist under
  `compiler/test/TritonFrontend/`.
- Static TTIR-to-scheduled-VC4 pipeline tests exist under
  `compiler/test/CodeGen/Triton/Emit/`.
- Isolation hardware fixtures exist under
  `compiler/test/CodeGen/Triton/Hardware/Run/`.
- Mixed TTIR hardware acceptance exists under
  `compiler/test/CodeGen/Triton/Hardware/MixedAcceptance/`.
- Mixed fixture claims are audited through
  `mixed_fixture_claims.json` and `audit_mixed_fixture_claims.py`.

## Non-Goals

Phase 7 does not make full Triton ready. Reductions, dot/contract,
gather/scatter, block pointers, atomics, rank-2 tensors, subword/f16/bf16/int8
memory lowering, SFU/math, control flow, `BLOCK_SIZE != 16`, and program-id
axes 1/2 remain staged until later phases.

The final Phase 7j prompt is expected to run the final acceptance gates and may
update the pending lock/readiness lines.
