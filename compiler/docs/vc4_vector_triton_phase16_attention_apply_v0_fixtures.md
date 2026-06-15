PHASE16_CONTROLLED_TRITON_ATTENTION_APPLY_FIXTURES=YES
PHASE16_RESULT=LOCKED
FEATURE=VALUE_AND_TTIR_ATTENTION_APPLY_V0
PHASE16_VALUE_ATTENTION_APPLY_V0_CONTRACT=LOCKED
VALUE_ATTENTION_APPLY_V0_METADATA_POLICY=PASS
VALUE_ATTENTION_APPLY_V0_SURFACE=ACCEPTED
VALUE_ATTENTION_APPLY_V0_STATIC=PASS
VALUE_ATTENTION_APPLY_V0_COMPOSITE=YES
VALUE_ATTENTION_APPLY_V0_HARDWARE_ISOLATION=PASS
VALUE_ATTENTION_APPLY_V0_F32_HARDWARE=PASS
VALUE_ATTENTION_APPLY_V0_SCALED_HARDWARE=PASS
VALUE_ATTENTION_APPLY_V0_F16_STORAGE_HARDWARE=PASS
VALUE_ATTENTION_APPLY_V0_TOLERANCE_POLICY=LOCKED
VALUE_ATTENTION_APPLY_V0_MIXED_ACCEPTANCE=PASS
VALUE_MIXED_REGRESSION=PASS
REAL_TRITON_ATTENTION_APPLY_V0_SOURCES=YES
REAL_TTIR_ATTENTION_APPLY_V0_SNAPSHOTS=YES
ACCEPTED_FIXTURES_EXCLUDE_UNRELATED_STAGED_FEATURES=YES
PRECOMPUTED_SCORES_ONLY=YES
TRANSPOSED_V_LAYOUT_REQUIRED=YES
NONTRANSPOSED_V_GATHER_STAGED=YES
SCALAR_GLOBAL_LOAD_STAGED=YES
K_ZERO_ATTENTION_APPLY_STAGED_OR_GUARD_REQUIRED=YES
ATTENTION_APPLY_METADATA_POLICY_ONLY=YES
ATTENTION_APPLY_METADATA_USED_FOR_LOWERING=NO
STRUCTURAL_ATTENTION_APPLY_TESTS_REQUIRED=YES
MAGIC_METADATA_NOT_COUNTED_AS_EXECUTABLE_SUPPORT=YES
READY_FOR_PHASE16_4_VALUE_ATTENTION_APPLY_STATIC=YES
READY_FOR_PHASE16_5_VALUE_HARDWARE_ISOLATION=YES
READY_FOR_PHASE16_6_VALUE_MIXED_ACCEPTANCE=YES
READY_FOR_PHASE16_7_TTIR_IMPORTER_ATTENTION_APPLY_STATIC=YES
READY_FOR_PHASE16_3_VALUE_SURFACE_CONTRACT=YES
READY_FOR_TRITON=NO

# VC4 Vector/Triton Phase 16 Attention-Apply v0 Fixtures

Phase 16.2 locks source-controlled real Triton sources and generated TTIR
snapshots for precomputed-score attention-apply v0. These fixtures define the
TTIR acceptance contract for the next Phase 16 value-surface work.

Phase 16.3 locks the matching value-surface contract. The accepted value form
is the explicit standard-IR composite tagged with
`vc4value.attention_apply_v0 = "precomputed_transposed_v_active_1_to_16"`;
there is no new `vc4value` attention or softmax-apply operation.
The tag is metadata-only verifier policy. It is not a lowering discriminator,
not a target op, and not executable support by itself; structural value IR
tests and later static/hardware proof carry the support claims.

Phase 16.4 proves the accepted value composite statically through
`value -> vc4kernel -> ssavc4 -> scheduled vc4` for f32, scaled f32, f16
storage, and a mixed axes/mask/control-flow/f16-storage candidate. This proof
composes already locked planners; it does not add QK score generation, dot,
contract, scalar global load, non-transposed V gather, or executable hardware
claims.

Phase 16.5 proves the accepted value composite on real VC4 hardware with
targeted isolation fixtures. The f32 fixture covers the requested cartesian
cases for `q_rows`, `out_dims`, `K`, score stride, and transposed-V stride with
`active_qpus=12`. The scaled fixture proves the scalar f32 scale argument
without scalar global loads. The repeat fixture alternates K=1 and K=16
launches to check repeat-invocation behavior. The f16-storage fixture proves
f16 score/Vt storage with f32 softmax and weighted-sum compute. All fixtures
use strict CPU oracles, sentinels, expected JSON checks, nonzero output hashes,
and the Phase15 natural-exp softmax tolerance policy.

Phase 16.6 adds
`mixed_value_attention_apply_v0_axes_mask_cf_f16_storage_vc4value` to the
cumulative VC4Value mixed acceptance suite. The mixed fixture combines
precomputed-score attention apply with multi-axis launch, a group-axis
control-flow guard, K tail masks, row-strided f16/f32/i32 memory, f16 score/Vt
storage loads, i32-gated f32 residual arithmetic, natural-exp softmax, weighted
f32 reduction, scalar result stores, repeated launches, and active_qpus=12. The
full VC4Value mixed suite passes with zero mismatches, zero sentinel
mismatches, and zero launch failures.

Accepted attention-apply v0 is intentionally narrow:

- precomputed score pointer `S`;
- transposed value pointer `VT`;
- row-contiguous score and Vt loads over `BLOCK_SIZE = 16`;
- runtime active count `K` with accepted domain 1..16;
- optional scalar f32 scale argument, represented as splat plus multiply;
- finite masked stable softmax using public natural `tl.exp`;
- weighted sum using elementwise multiply and add reduction;
- scalar store to `O + q * LDO + d`.

The accepted contract excludes QK score generation, `tl.dot`, `tt.dot`,
`vector.contract`, non-transposed V gather, lane-varying stride, scalar global
scale loads, block pointers, multiblock softmax, full attention, and
FlashAttention.

The source-controlled fixture package lives under:

```text
examples/triton/phase16_attention_apply_v0/
```

Accepted sources and snapshots:

- `sources/ttir_attention_apply_v0_f32_b16.py`
- `generated/ttir_attention_apply_v0_f32_b16.ttir.mlir`
- `sources/ttir_attention_apply_v0_scaled_f32_b16.py`
- `generated/ttir_attention_apply_v0_scaled_f32_b16.ttir.mlir`
- `sources/mixed_ttir_attention_apply_v0_axes_mask_cf_f16_storage_b16.py`
- `generated/mixed_ttir_attention_apply_v0_axes_mask_cf_f16_storage_b16.ttir.mlir`

Staged/reject sources and snapshots:

- `ttir_attention_apply_nontransposed_v_reject_b16`: staged
  `STAGED_LANE_VARYING_STRIDE`.
- `ttir_attention_apply_scalar_scale_load_reject_b16`: staged
  `STAGED_SCALAR_TT_LOAD`.
- `ttir_attention_apply_k_zero_reject_b16`: staged
  `STAGED_ZERO_ACTIVE_ATTENTION_APPLY`.
- `ttir_attention_apply_multiblock_reject_b16`: staged
  `STAGED_MULTIBLOCK_SOFTMAX`.
- `ttir_attention_apply_tl_dot_reject_b16`: staged `STAGED_TT_DOT`.

The emitted snapshots were generated from real `@triton.jit` sources with the
existing zero-setup generator wrapper recorded by prior phases. TTIR was not
hand-written.

Natural `tl.exp` remains public natural exponentiation. On VC4 it is governed
by the Phase15B base-2 SFU semantic repair and the Phase15 explicit
approximate-SFU finite tolerance policy. Exact/default math remains rejected.

`READY_FOR_TRITON` remains `NO`.

## Phase 16.7 TTIR Importer Static Lock

TTIR_ATTENTION_APPLY_V0_IMPORTER_STATIC=PASS
TTIR_ATTENTION_APPLY_V0_LOWERING=YES
TTIR_ATTENTION_APPLY_V0_PRECOMPUTED_SCORES=YES
TTIR_ATTENTION_APPLY_V0_TRANSPOSED_V_LAYOUT=YES
TTIR_ATTENTION_APPLY_V0_NATURAL_EXP_SOFTMAX=YES
TTIR_ATTENTION_APPLY_V0_WEIGHTED_SUM=YES
TTIR_NONTRANSPOSED_V_GATHER_REJECT=PASS
TTIR_SCALAR_GLOBAL_LOAD_REJECT=PASS
FRONTEND_ROBUSTNESS_AUDIT=PASS
READY_FOR_PHASE16_8_TTIR_HARDWARE_ISOLATION=YES
READY_FOR_TRITON=NO

The controlled Phase 16 accepted TTIR snapshots lower through the C++ importer
by composition of previously locked structural planners: program-id axes,
row-contiguous score/Vt pointer slices, canonical tail masks, finite
max/add reductions, natural `math.exp` with explicit approximate-SFU policy,
scalar f32 scale splat where present, weighted elementwise multiply, and scalar
`memref.store` output. The imported IR remains in the standard value layer and
does not emit VC4Kernel, SSAVC4, or scheduled VC4 directly.

The staged snapshots remain rejected structurally: non-transposed V uses a
lane-varying stride/gather pointer expression, scalar scale memory uses scalar
`tt.load`, K=0 uses a non-canonical memory mask, multiblock softmax uses staged
K accumulation, and `tl.dot` emits staged `tt.dot`/contract form.

## Phase 16.8 TTIR Hardware Isolation Lock

TTIR_ATTENTION_APPLY_V0_HARDWARE_ISOLATION=PASS
TTIR_ATTENTION_APPLY_V0_MIXED_ACCEPTANCE=PASS
TTIR_MIXED_REGRESSION=PASS
LAYERED_REGRESSION_POLICY_APPLIED=YES
NO_TEMPORARY_ATTENTION_APPLY_WORKAROUNDS=YES
TTIR_ATTENTION_APPLY_V0_F32_HARDWARE=PASS
TTIR_ATTENTION_APPLY_V0_SCALED_HARDWARE=PASS
TTIR_ATTENTION_APPLY_V0_TTIR_HARNESS_BLOCK_X=16
READY_FOR_PHASE16_9_TTIR_MIXED_FINAL_LOCK=YES
READY_FOR_PHASE17_ONLINE_MULTIBLOCK_SOFTMAX_STATE=YES
READY_FOR_TRITON=NO

Phase 16.8 proves the accepted controlled TTIR attention-apply v0 snapshots on
real VC4 hardware through the full TTIR importer path. The f32 and scaled-f32
fixtures use exact source-controlled TTIR snapshot copies as hardware inputs,
lower through the C++ importer, and run with `active_qpus=12`, `lanes=16`, and
TTIR harness `block.x=16`. Each fixture covers q_rows 1/2/7, out_dims 1/2/5,
K 1/2/7/15/16, score strides 16/17/23, transposed-V strides 16/19, padded
output strides with sentinels, strict natural-exp CPU oracle checks, and
nonzero output hashes.

The staged scalar-load, non-transposed V, K=0, multiblock, and dot snapshots
remain static rejects and were not run on hardware.

## Phase 16.9 Final Mixed Lock

PHASE16_RESULT=LOCKED
FEATURE=VALUE_AND_TTIR_ATTENTION_APPLY_V0
VALUE_ATTENTION_APPLY_V0_CONTRACT=LOCKED
VALUE_ATTENTION_APPLY_V0_STATIC=PASS
VALUE_ATTENTION_APPLY_V0_HARDWARE_ISOLATION=PASS
VALUE_ATTENTION_APPLY_V0_MIXED_ACCEPTANCE=PASS
REAL_TRITON_ATTENTION_APPLY_V0_SOURCES=YES
REAL_TTIR_ATTENTION_APPLY_V0_SNAPSHOTS=YES
TTIR_ATTENTION_APPLY_V0_IMPORTER_STATIC=PASS
TTIR_ATTENTION_APPLY_V0_HARDWARE_ISOLATION=PASS
TTIR_ATTENTION_APPLY_V0_MIXED_ACCEPTANCE=PASS
PRECOMPUTED_SCORES_ONLY=YES
TRANSPOSED_V_LAYOUT_REQUIRED=YES
NONTRANSPOSED_V_GATHER_STAGED=YES
SCALAR_GLOBAL_LOAD_STAGED=YES
K_ZERO_ATTENTION_APPLY_STAGED_OR_GUARD_REQUIRED=YES
MULTIBLOCK_SOFTMAX_STAGED=YES
QK_SCORE_GENERATION_STAGED=YES
TL_DOT_TT_DOT_STAGED=YES
FULL_ATTENTION_STAGED=YES
VALUE_MIXED_REGRESSION=PASS
TTIR_MIXED_REGRESSION=PASS
LAYERED_REGRESSION_POLICY_APPLIED=YES
NO_TEMPORARY_ATTENTION_APPLY_WORKAROUNDS=YES
READY_FOR_PHASE17_ONLINE_MULTIBLOCK_SOFTMAX_STATE=YES
READY_FOR_TRITON=NO

Phase 16.9 adds the TTIR mixed acceptance fixture
`mixed_ttir_attention_apply_v0_axes_mask_cf_f16_storage_b16_vc4triton` and
runs the layered value and TTIR mixed regressions. The mixed TTIR fixture uses
the exact source-controlled Phase 16 controlled TTIR snapshot as hardware
input, lowers through the C++ importer, and proves attention-apply v0 together
with axes, masks, scalar control flow, row-strided f16 storage, natural-exp
softmax, weighted reduction, scalar result store, sentinels, and
`active_qpus=12` with TTIR harness `block.x=16`.
