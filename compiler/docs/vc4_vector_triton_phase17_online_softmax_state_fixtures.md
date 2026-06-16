PHASE17_CONTROLLED_TRITON_ONLINE_SOFTMAX_FIXTURES=YES
PHASE17_VALUE_ONLINE_SOFTMAX_STATE_CONTRACT=LOCKED
REAL_TRITON_ONLINE_SOFTMAX_SOURCES=YES
REAL_TTIR_ONLINE_SOFTMAX_SNAPSHOTS=YES
ACCEPTED_FIXTURES_EXCLUDE_UNRELATED_STAGED_FEATURES=YES
VALUE_ONLINE_SOFTMAX_STATE_SURFACE=ACCEPTED
VALUE_ONLINE_ATTENTION_APPLY_SURFACE=ACCEPTED
VALUE_ONLINE_SOFTMAX_STATE_STATIC=PASS
VALUE_ONLINE_ATTENTION_APPLY_STATIC=PASS
VALUE_ONLINE_SOFTMAX_COMPOSITE=YES
LOOP_CARRIED_F32_STATE_STATIC=PASS
VALUE_ONLINE_SOFTMAX_STATE_HARDWARE_ISOLATION=PASS
VALUE_ONLINE_ATTENTION_APPLY_HARDWARE_ISOLATION=PASS
VALUE_ONLINE_ATTENTION_APPLY_F32_HARDWARE=PASS
VALUE_ONLINE_ATTENTION_APPLY_SCALED_HARDWARE=PASS
VALUE_ONLINE_ATTENTION_APPLY_F16_STORAGE_HARDWARE=NOT_REQUIRED_FOR_PHASE17_CORE
VALUE_ONLINE_SOFTMAX_TOLERANCE_POLICY=LOCKED
VALUE_ONLINE_SOFTMAX_MIXED_ACCEPTANCE=PASS
VALUE_MIXED_REGRESSION=PASS
PRECOMPUTED_SCORES_ONLY=YES
TRANSPOSED_V_LAYOUT_REQUIRED=YES
K_RANGE_ACCEPTED=1_TO_64
K_ZERO_STAGED=YES
NONTRANSPOSED_V_GATHER_STAGED=YES
SCALAR_GLOBAL_LOAD_STAGED=YES
K_ZERO_ONLINE_SOFTMAX_STAGED=YES
READY_FOR_PHASE17_3_VALUE_SURFACE_CONTRACT=YES
READY_FOR_PHASE17_4_VALUE_ONLINE_SOFTMAX_STATIC=YES
READY_FOR_PHASE17_5_VALUE_HARDWARE_ISOLATION=YES
READY_FOR_PHASE17_6_VALUE_MIXED_ACCEPTANCE=YES
READY_FOR_PHASE17_7_TTIR_IMPORTER_ONLINE_SOFTMAX_STATIC=YES
READY_FOR_TRITON=NO

# VC4 Vector/Triton Phase 17 Online Softmax State Fixtures

Phase 17.2 locks source-controlled real Triton sources and generated TTIR
snapshots for online/multiblock softmax state over K blocks of 16. These
fixtures define the Phase 17 TTIR acceptance contract. They are controlled
fixtures, not exploratory probes.

Phase 17.3 locks the matching value-surface contract. Online softmax state is
accepted as a structural standard value IR composite with loop-carried scalar
f32 `m/l/acc` state; no `vc4value.online_softmax` or `vc4value.attention`
operation is introduced. No Phase 17 metadata attribute is introduced.

Phase 17.4 statically proves the value composite through
`value -> vc4kernel -> ssavc4 -> scheduled vc4`. The proof composes the locked
control-flow, dynamic row-slice memory, scalar-store, reduction, f16-storage,
and natural-exp SFU planners, with a narrow finite scalar f32 max helper for
the online recurrence. It does not run hardware and does not broaden the
accepted TTIR scope.

Phase 17.5 proves the accepted value online softmax state and online
attention-apply composites on real VC4 hardware with targeted isolation
fixtures. The fixtures run with `active_qpus=12`, strict sentinels, nonzero
output hashes, host CPU oracles, and checked expected JSON. The proof covers
normalizer `m/l` state, attention `m/l/acc` state, K values greater than one
block, precomputed scores, transposed V layout, scalar result stores, scalar
f32 scale arguments, and repeat launches. The f16-storage variant remains not
required for the Phase 17 core hardware-isolation proof.

Phase 17.6 adds the cumulative value mixed fixture
`mixed_value_online_attention_apply_axes_mask_cf_f16_storage_vc4value`. The
fixture combines multi-axis launch, control flow, masks, row-strided f16
storage, i32/f32 value interaction, finite reductions, natural-exp softmax,
online `m/l/acc` attention state over K blocks greater than 16, transposed V
layout, weighted reduction, scalar result stores, repeated launches, and
`active_qpus=12`. The full VC4Value mixed hardware suite passes with strict
CPU oracles, sentinels, nonzero output hashes, and checked claims. Scalar
global loads, non-transposed V gather, K=0, QK score generation, dot/contract,
full attention, and FlashAttention remain staged.

Accepted Phase 17 scope is intentionally narrow:

- real `@triton.jit` sources;
- `BLOCK_SIZE = 16`;
- runtime scalar i32 `K`, with accepted hardware domain `1..64`;
- precomputed score pointer `S`;
- transposed value pointer `VT`;
- output pointer `O`;
- one program computes one `(q, d)` output where applicable;
- loop over K in blocks of 16 using `tl.range`;
- loop-carried scalar f32 `m`, `l`, and optional `acc` online-softmax state;
- masked score and Vt loads with `other=0.0`;
- inactive scores handled by `tl.where(mask, score, finite_low)`;
- public natural `tl.exp` in the online recurrence;
- scalar result stores.

Accepted fixtures exclude QK score generation, `tl.dot`, `tt.dot`,
`vector.contract`, scalar global loads, non-transposed V gather, lane-varying
stride, block pointers, tensor descriptors, atomics, cooperative/VPM tiling,
full attention, and FlashAttention.

The source-controlled fixture package lives under:

```text
examples/triton/phase17_online_softmax_state/
```

Accepted sources and snapshots:

- `sources/ttir_online_softmax_normalizer_f32_b16.py`
- `generated/ttir_online_softmax_normalizer_f32_b16.ttir.mlir`
- `sources/ttir_online_attention_apply_v0_f32_b16.py`
- `generated/ttir_online_attention_apply_v0_f32_b16.ttir.mlir`
- `sources/ttir_online_attention_apply_v0_scaled_f32_b16.py`
- `generated/ttir_online_attention_apply_v0_scaled_f32_b16.ttir.mlir`
- `sources/mixed_ttir_online_attention_apply_axes_mask_cf_f16_storage_b16.py`
- `generated/mixed_ttir_online_attention_apply_axes_mask_cf_f16_storage_b16.ttir.mlir`

Staged/reject sources and snapshots:

- `ttir_online_attention_nontransposed_v_reject_b16`: staged
  `STAGED_LANE_VARYING_STRIDE`.
- `ttir_online_attention_scalar_scale_load_reject_b16`: staged
  `STAGED_SCALAR_TT_LOAD`.
- `ttir_online_softmax_k_zero_reject_b16`: staged
  `STAGED_K_ZERO_ONLINE_SOFTMAX`.
- `ttir_online_attention_qk_score_generation_reject_b16`: staged
  `STAGED_QK_SCORE_GENERATION`.
- `ttir_online_attention_tl_dot_reject_b16`: staged `STAGED_TT_DOT`.

The accepted normalizer snapshot emits `scf.for` from `tl.range(0, K, 16)` with
loop-carried scalar f32 `m/l` state. The accepted attention-apply snapshots
emit `scf.for` with loop-carried scalar f32 `m/l/acc` state, `arith.maxnumf`
for scalar max, `tt.reduce` helpers for block max/sum forms, natural
`math.exp`, and scalar result stores. The scaled fixture uses a scalar f32
argument emitted as `tt.splat` plus vector `arith.mulf`, not a scalar `tt.load`.

The mixed accepted fixture combines multi-axis program ids, structured
control-flow, masked f16 storage loads promoted to f32 compute, scalar f32
scale, online `m/l/acc` state, transposed V layout, and scalar output/audit
stores.

The emitted snapshots were generated from real Triton sources with the existing
zero-setup generator wrapper recorded by prior phases. TTIR was not
hand-written and is not fake acceptance IR.

Natural `tl.exp` remains public natural exponentiation. On VC4 it is governed
by the Phase15B base-2 SFU semantic repair and the Phase15 explicit
approximate-SFU finite tolerance policy. Future hardware acceptance for these
fixtures must use the appropriate approximate-SFU tolerance caveat. Exact or
default math remains rejected unless a later phase explicitly changes that
policy.

Phase 17.5 locks the online softmax tolerance policy for these isolation
fixtures. The normalizer denominator fixture uses the Phase15 natural-exp SFU
policy with absolute/relative caps `0.12/0.025`; the hardware proof observed
`max_abs_diff=0.001731` and `max_rel_diff=0.000093`. The attention fixtures use
absolute cap `0.012` and relative cap `0.08` to account for near-zero weighted
outputs while preserving a strict absolute bound; the hardware proof observed
`max_abs_diff <= 0.000071` across f32 and scaled attention. This tolerance does
not mask wrong reductions or an incorrect exp base because the result checker
still requires zero mismatches, zero sentinel mismatches, nonzero output hashes,
natural-exp oracle agreement, and checked staged-feature guards.

`READY_FOR_TRITON` remains `NO`.
