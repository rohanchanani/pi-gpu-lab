PHASE15_VALUE_TO_VC4KERNEL_PLANNED_COVERAGE=YES
PHASE15_VALUE_APPROX_MATH_SFU_SOFTMAX_CONTRACT=LOCKED
VALUE_APPROX_SFU_EXP_SURFACE=ACCEPTED
VALUE_APPROX_SFU_RECIP_DIV_SURFACE=ACCEPTED
VALUE_FINITE_F32_MAX_REDUCTION_SURFACE=ACCEPTED
VALUE_SOFTMAX_V0_COMPOSITE_SURFACE=ACCEPTED
APPROX_MATH_POLICY=EXPLICIT
EXACT_DEFAULT_MATH_REJECTED=YES
ZERO_ACTIVE_SOFTMAX_STATUS=STAGED_OR_EXPLICIT_NOOP_GUARD_REQUIRED
READY_FOR_PHASE15_4_VALUE_SFU_SOFTMAX_STATIC=YES
READY_FOR_TRITON=NO

# Phase 15 Value-To-VC4Kernel Planned Coverage

Phase 15.4 must statically prove the value lowering for the Phase 15
approximate-SFU and one-block softmax contract. This document is a planned
coverage handoff only; it does not implement executable lowering.

Required accepted static coverage:

- `math.exp` scalar/vector f32 approximate-SFU lowering.
- `arith.divf` approximate reciprocal/division lowering for finite nonzero
  denominator policy.
- Optional `math.log` and `math.rsqrt` approximate-SFU lowering, because the
  lower-half SFU modes exist.
- `vector.reduction <maxnumf>` and `<maximumf>` finite f32 max reduction
  lowering over `vector<16xf32>`.
- Scalar-to-vector f32 broadcast lowering through the locked target splat or
  accepted composite.
- One-block stable softmax v0 composite lowering:
  max reduction, broadcast/subtract, approximate exp, masked zeroing, add
  reduction, approximate reciprocal/division, multiply, and masked store.

Required staged/reject static coverage:

- exact/default math without `vc4value.math_policy = "approx_sfu"`;
- NaN/Inf math policy;
- generic division without approximate-SFU policy;
- active-count-zero softmax without explicit finite no-op guard;
- multiblock softmax;
- block-pointer softmax;
- full attention and FlashAttention;
- quantized softmax;
- `tt.dot`, `vector.contract`, and GEMM.

No hardware proof is claimed by this planned-coverage document.
