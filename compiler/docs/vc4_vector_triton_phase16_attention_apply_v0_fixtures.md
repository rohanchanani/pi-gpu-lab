PHASE16_CONTROLLED_TRITON_ATTENTION_APPLY_FIXTURES=YES
REAL_TRITON_ATTENTION_APPLY_V0_SOURCES=YES
REAL_TTIR_ATTENTION_APPLY_V0_SNAPSHOTS=YES
ACCEPTED_FIXTURES_EXCLUDE_UNRELATED_STAGED_FEATURES=YES
PRECOMPUTED_SCORES_ONLY=YES
TRANSPOSED_V_LAYOUT_REQUIRED=YES
NONTRANSPOSED_V_GATHER_STAGED=YES
SCALAR_GLOBAL_LOAD_STAGED=YES
K_ZERO_ATTENTION_APPLY_STAGED_OR_GUARD_REQUIRED=YES
READY_FOR_PHASE16_3_VALUE_SURFACE_CONTRACT=YES
READY_FOR_TRITON=NO

# VC4 Vector/Triton Phase 16 Attention-Apply v0 Fixtures

Phase 16.2 locks source-controlled real Triton sources and generated TTIR
snapshots for precomputed-score attention-apply v0. These fixtures define the
TTIR acceptance contract for the next Phase 16 value-surface work.

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
