PHASE16_CONTROLLED_TRITON_ATTENTION_APPLY_FIXTURES=YES
PHASE16_VALUE_ATTENTION_APPLY_V0_CONTRACT=LOCKED
VALUE_ATTENTION_APPLY_V0_SURFACE=ACCEPTED
VALUE_ATTENTION_APPLY_V0_STATIC=PASS
VALUE_ATTENTION_APPLY_V0_COMPOSITE=YES
REAL_TRITON_ATTENTION_APPLY_V0_SOURCES=YES
REAL_TTIR_ATTENTION_APPLY_V0_SNAPSHOTS=YES
ACCEPTED_FIXTURES_EXCLUDE_UNRELATED_STAGED_FEATURES=YES
PRECOMPUTED_SCORES_ONLY=YES
TRANSPOSED_V_LAYOUT_REQUIRED=YES
NONTRANSPOSED_V_GATHER_STAGED=YES
SCALAR_GLOBAL_LOAD_STAGED=YES
K_ZERO_ATTENTION_APPLY_STAGED_OR_GUARD_REQUIRED=YES
READY_FOR_PHASE16_4_VALUE_ATTENTION_APPLY_STATIC=YES
READY_FOR_PHASE16_5_VALUE_HARDWARE_ISOLATION=YES
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

Phase 16.4 proves the accepted value composite statically through
`value -> vc4kernel -> ssavc4 -> scheduled vc4` for f32, scaled f32, f16
storage, and a mixed axes/mask/control-flow/f16-storage candidate. This proof
composes already locked planners; it does not add QK score generation, dot,
contract, scalar global load, non-transposed V gather, or executable hardware
claims.

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
