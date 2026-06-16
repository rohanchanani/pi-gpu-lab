# VC4 Value-to-VC4Kernel Planning Guide

PHASE10_VALUE_MASK_CLASSIFIER_CONTRACT=LOCKED
PHASE10_VALUE_MEMORY_LEGALITY_CONTRACT=LOCKED
PHASE12_VALUE_REDUCTION_CONTRACT=LOCKED
PHASE13_VALUE_GEMV_ROWWISE_DOT_CONTRACT=LOCKED
PHASE14_VALUE_ML_STORAGE_NUMERIC_CONTRACT=LOCKED
PHASE15_VALUE_APPROX_MATH_SFU_SOFTMAX_CONTRACT=LOCKED
PHASE16_VALUE_ATTENTION_APPLY_V0_CONTRACT=LOCKED
PHASE17_VALUE_ONLINE_SOFTMAX_STATE_CONTRACT=LOCKED
VALUE_ATTENTION_APPLY_V0_METADATA_POLICY=PASS
PHASE14_RESULT=LOCKED
PHASE15_RESULT=LOCKED
FEATURE=VALUE_AND_TTIR_ML_STORAGE_NUMERIC_CONVERSION_POLICY
FEATURE=VALUE_AND_TTIR_APPROX_MATH_SFU_SOFTMAX
FEATURE=VALUE_AND_TTIR_ATTENTION_APPLY_V0
VALUE_APPROX_MATH_SFU_CONTRACT=LOCKED
VALUE_ATTENTION_APPLY_V0_SURFACE=ACCEPTED
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
VALUE_ATTENTION_APPLY_V0_STATIC=PASS
VALUE_ATTENTION_APPLY_V0_COMPOSITE=YES
VALUE_ATTENTION_APPLY_V0_STATIC_PLANNED=YES
VALUE_ATTENTION_APPLY_V0_COMPOSITE_LOWERS_PLANNED=YES
VALUE_ATTENTION_APPLY_V0_SCALED_STATIC_PLANNED=YES
VALUE_ATTENTION_APPLY_V0_F16_STORAGE_STATIC_PLANNED=YES
VALUE_ATTENTION_APPLY_V0_SCALAR_GLOBAL_LOAD_STAGED=YES
VALUE_ATTENTION_APPLY_V0_NONTRANSPOSED_V_GATHER_STAGED=YES
VALUE_ATTENTION_APPLY_V0_K_ZERO_STAGED_OR_GUARD_REQUIRED=YES
ATTENTION_APPLY_METADATA_POLICY_ONLY=YES
ATTENTION_APPLY_METADATA_USED_FOR_LOWERING=NO
STRUCTURAL_ATTENTION_APPLY_TESTS_REQUIRED=YES
MAGIC_METADATA_NOT_COUNTED_AS_EXECUTABLE_SUPPORT=YES
VALUE_GEMV_ROWWISE_DOT_STATIC=PASS
VALUE_GEMV_F32_ROW_DOT_STATIC=PASS
VALUE_GEMV_I32_ROW_DOT_STATUS=STAGED_BY_I32_POLICY
VALUE_GEMV_PARTIAL_KBLOCK_STATIC=PASS
VALUE_GEMV_ROWWISE_DOT_HARDWARE_ISOLATION=PASS
VALUE_GEMV_F32_ROW_DOT_HARDWARE=PASS
VALUE_GEMV_I32_ROW_DOT_HARDWARE=NOT_REQUIRED_STAGED_BY_I32_POLICY
VALUE_GEMV_PARTIAL_KBLOCK_HARDWARE=PASS
VALUE_GEMV_EMPTY_REPEAT_HARDWARE=PASS
VALUE_GEMV_ROWWISE_DOT_MIXED_ACCEPTANCE=PASS
VALUE_MIXED_REGRESSION=PASS
VALUE_REDUCTION_TO_VC4KERNEL_STATIC=PASS
VALUE_VECTOR_REDUCTION_ADD_I32_SURFACE=ACCEPTED
VALUE_VECTOR_REDUCTION_ADD_F32_FINITE_SURFACE=ACCEPTED
VALUE_SCALAR_MEMREF_STORE_FOR_REDUCTION_SURFACE=ACCEPTED
VALUE_GEMV_ROWWISE_DOT_F32_SURFACE=ACCEPTED
VALUE_GEMV_ROWWISE_DOT_I32_SURFACE=ACCEPTED
VALUE_F16_STORAGE_TO_F32_COMPUTE_SURFACE=ACCEPTED
VALUE_F32_COMPUTE_TO_F16_STORAGE_SURFACE=ACCEPTED
VALUE_APPROX_SFU_EXP_SURFACE=ACCEPTED
VALUE_APPROX_SFU_RECIP_DIV_SURFACE=ACCEPTED
VALUE_FINITE_F32_MAX_REDUCTION_SURFACE=ACCEPTED
VALUE_SOFTMAX_V0_COMPOSITE_SURFACE=ACCEPTED
VALUE_ATTENTION_APPLY_V0_COMPOSITE_SURFACE=ACCEPTED
VALUE_ATTENTION_APPLY_V0_STATIC=PASS
VALUE_ATTENTION_APPLY_V0_COMPOSITE=YES
VALUE_APPROX_SFU_STATIC=PASS
VALUE_APPROX_SFU_EXP_STATIC=PASS
VALUE_APPROX_SFU_RECIP_DIV_STATIC=PASS
VALUE_FINITE_F32_MAX_REDUCTION_STATIC=PASS
VALUE_SOFTMAX_V0_STATIC=PASS
VALUE_APPROX_SFU_HARDWARE_ISOLATION=PASS
VALUE_APPROX_SFU_EXP_HARDWARE=PASS
VALUE_APPROX_SFU_RECIP_DIV_HARDWARE=PASS
VALUE_FINITE_F32_MAX_REDUCTION_HARDWARE=PASS
VALUE_SOFTMAX_V0_HARDWARE_ISOLATION=PASS
VALUE_APPROX_SFU_TOLERANCE_POLICY=LOCKED
PHASE15_BASE2_SFU_SEMANTICS_CONTRACT=LOCKED
TARGET_SFU_EXP_IS_EXP2=YES
TARGET_SFU_LOG_IS_LOG2=YES
VALUE_MATH_EXP_IS_NATURAL_EXP=YES
VALUE_MATH_LOG_IS_NATURAL_LOG=YES
VALUE_MATH_SQRT_IS_SQRT=YES
NATURAL_EXP_LOWERING=EXP2_X_LOG2E
NATURAL_LOG_LOWERING=LOG2_X_LN2
SQRT_LOWERING=RSQRT_TIMES_X_POSITIVE_FINITE_DOMAIN
PHASE15_5_EXP2_ORACLE_IF_PRESENT_REQUIRES_REPAIR=YES
VALUE_NATURAL_EXP_STATIC=PASS
VALUE_NATURAL_LOG_STATIC=PASS
VALUE_SQRT_STATIC=PASS
VALUE_RSQRT_HARDWARE=PASS
SQRT_LOWERING=RSQRT_TIMES_X
SOFTMAX_USES_NATURAL_EXP=YES
VALUE_APPROX_SFU_SOFTMAX_MIXED_ACCEPTANCE=PASS
VALUE_MIXED_CLAIM_AUDIT=PASS
TTIR_APPROX_SFU_SOFTMAX_MIXED_ACCEPTANCE=PASS
VALUE_MIXED_REGRESSION=PASS
TTIR_MIXED_REGRESSION=PASS
LAYERED_REGRESSION_POLICY_APPLIED=YES
NO_TEMPORARY_SFU_SOFTMAX_WORKAROUNDS=YES
READY_FOR_PHASE16_ATTENTION_SCORE_SOFTMAX_APPLY=YES
READY_FOR_PHASE15_7_TTIR_IMPORTER_SFU_SOFTMAX_STATIC=YES
READY_FOR_PHASE15B_2_VALUE_HARDWARE_EXP_LOG_SQRT_REPROOF=YES
VALUE_F16_STORAGE_F32_COMPUTE_STATIC=PASS
VALUE_F16_STORE_F32_COMPUTE_STATIC=PASS
VALUE_F16_GEMV_INPUT_STORAGE_STATIC=PASS
VALUE_F16_STORAGE_F32_COMPUTE_HARDWARE_ISOLATION=PASS
VALUE_F16_LOAD_F32_COMPUTE_HARDWARE=PASS
VALUE_F32_COMPUTE_F16_STORE_HARDWARE=PASS
VALUE_F16_GEMV_INPUT_STORAGE_HARDWARE=PASS
VALUE_I32_TO_F32_CAST_HARDWARE=NOT_REQUIRED_STAGED_BY_LOWER_HALF_GAP
VALUE_F16_STORAGE_F32_COMPUTE_MIXED_ACCEPTANCE=PASS
VALUE_MIXED_REGRESSION=PASS
TTIR_MIXED_REGRESSION=PASS
LAYERED_REGRESSION_POLICY_APPLIED=YES
NO_TEMPORARY_STORAGE_NUMERIC_WORKAROUNDS=YES
VALUE_VECTOR_REDUCTION_ADD_I32_STATIC=PASS
VALUE_VECTOR_REDUCTION_ADD_F32_FINITE_STATIC=PASS
VALUE_SCALAR_STORE_FOR_REDUCTION_STATIC=PASS
F32_REDUCTION_FINITE_TREE_POLICY=YES
F32_DOT_FINITE_TREE_POLICY=YES
F16_STORAGE_FINITE_POLICY=YES
APPROX_MATH_POLICY=EXPLICIT
EXACT_DEFAULT_MATH_REJECTED=YES
PRECOMPUTED_SCORES_ONLY=YES
TRANSPOSED_V_LAYOUT_REQUIRED=YES
K_RANGE_ACCEPTED=1_TO_64
K_ZERO_STAGED=YES
SCALAR_GLOBAL_LOAD_STAGED=YES
NONTRANSPOSED_V_GATHER_STAGED=YES
K_ZERO_ATTENTION_APPLY_STAGED_OR_GUARD_REQUIRED=YES
ZERO_ACTIVE_SOFTMAX_STATUS=STAGED_OR_EXPLICIT_NOOP_GUARD_PROVEN
I32_TO_F32_CAST_STATUS=STAGED_BY_LOWER_HALF_GAP
NATIVE_F16_ARITHMETIC_STAGED=YES
BF16_FP8_STAGED=YES
INT8_INT16_QUANTIZED_STORAGE_STAGED=YES
FP_TO_INT_CASTS_STAGED=YES
EXACT_UNPOLICY_NUMERIC_CASTS_STAGED=YES
SOFTMAX_SFU_STAGED_FOR_PHASE15=YES
UNSUPPORTED_REDUCTION_VARIANTS_STAGED=YES
TL_DOT_TT_DOT_STAGED=YES
VECTOR_CONTRACT_STAGED=YES
MULTIBLOCK_K_ACCUMULATION_STAGED=YES
SPARSE_MEMORY_MASKS_STAGED=YES
NONZERO_LOAD_OTHER_STAGED=YES
RANK2_STRIDED_MEMORY_STAGED=YES
READY_FOR_PHASE10_4_VALUE_MASK_MEMORY_CLASSIFIER_STATIC=YES
READY_FOR_PHASE12_4_VALUE_REDUCTION_STATIC=YES
READY_FOR_PHASE13_4_VALUE_GEMV_STATIC=YES
READY_FOR_PHASE13_5_VALUE_HARDWARE_ISOLATION=YES
READY_FOR_PHASE13_6_VALUE_MIXED_ACCEPTANCE=YES
READY_FOR_PHASE13_7_TTIR_IMPORTER_GEMV_STATIC=YES
READY_FOR_PHASE14_4_VALUE_STORAGE_NUMERIC_STATIC=YES
READY_FOR_PHASE14_5_VALUE_HARDWARE_ISOLATION=YES
READY_FOR_PHASE14_6_VALUE_MIXED_ACCEPTANCE=YES
READY_FOR_PHASE14_7_TTIR_IMPORTER_STORAGE_NUMERIC_STATIC=YES
READY_FOR_PHASE15_APPROX_MATH_SFU_SOFTMAX=YES
READY_FOR_PHASE15_4_VALUE_SFU_SOFTMAX_STATIC=YES
READY_FOR_PHASE15_5_VALUE_HARDWARE_ISOLATION=YES
READY_FOR_PHASE15_6_VALUE_MIXED_ACCEPTANCE=YES
READY_FOR_PHASE16_4_VALUE_ATTENTION_APPLY_STATIC=YES
READY_FOR_PHASE16_5_VALUE_HARDWARE_ISOLATION=YES
READY_FOR_PHASE17_4_VALUE_ONLINE_SOFTMAX_STATIC=YES
READY_FOR_PHASE17_5_VALUE_HARDWARE_ISOLATION=YES
READY_FOR_PHASE17_6_VALUE_MIXED_ACCEPTANCE=YES
READY_FOR_PHASE12_5_VALUE_HARDWARE_ISOLATION=YES
READY_FOR_TRITON=NO

## 1. Purpose and non-goals

This document specifies how later lowering phases should plan standard
value-surface patterns into the locked VC4Kernel target surface.

It is not an implementation. It does not add accepted value features, does not
add a `vc4value` dialect, does not add a verifier, does not add conversion
passes, does not add TTIR import, does not run hardware, and does not change the
locked VC4Kernel surface.

Phase 5 implementation status:

```text
VC4_VALUE_TO_VC4KERNEL_PASS_SKELETON_PRESENT=YES
PASS_FLAG=--convert-vc4-value-to-vc4kernel
VC4_VALUE_TO_VC4KERNEL_STATIC_ELEMENTWISE_LOWERING=YES
VC4_VALUE_TRANSFER_READ_TMU_STATIC=YES
VC4_VALUE_TRANSFER_WRITE_VDW_STATIC=YES
READY_FOR_HARDWARE_PHASE5G=YES
READY_FOR_TRITON=NO
```

Phase 5g hardware note:

```text
PHASE5G_VALUE_COPY_F32_TAIL_HARDWARE_SMOKE=YES
PHASE5G_ACTIVE_QPUS_1_ONLY=YES
PHASE5G_ORIGINAL_ACTIVE_QPUS_12_FIXTURE_EXPOSED_OVERLAUNCH_TAIL_MASK_GAP=YES
PHASE5G_NARROWED_TO_ONE_LOGICAL_VECTOR16_REQUEST_PER_BLOCK=YES
PHASE5G_NOT_SUFFICIENT_FOR_FINAL_PHASE5_TAIL_MEMORY_COVERAGE=YES
PHASE5H_OR_PHASE5J_REQUIRES_MULTI_REQUEST_12_ACTIVE_QPU_VALUE_HARDWARE=YES
```

The original Phase 5g `active_qpus=12` copy/tail fixture exposed a real
overlaunch/tail-mask gap: logical requests beyond `ceil(n / 16)` could observe
negative remaining element counts and write full inactive chunks. The committed
Phase 5g fixture was intentionally narrowed to one logical `vector<16>` request
per block only to prove the initial value-to-hardware path. It must not be
treated as complete Phase 5 tail/memory coverage.

Phase 5 final hardware lock status:

```text
VC4_VALUE_TO_VC4KERNEL_PHASE5_ELEMENTWISE_LOCKED=YES
VC4_VALUE_TO_VC4KERNEL_PASS_ACCEPTED=YES
VC4_VALUE_ELEMENTWISE_VECTOR16_COMPUTE_ACCEPTED_HARDWARE=YES
VC4_VALUE_TRANSFER_READ_TMU_ACCEPTED_HARDWARE=YES
VC4_VALUE_TRANSFER_WRITE_VDW_PRESERVE_ACCEPTED_HARDWARE=YES
VC4_VALUE_ELEMENTWISE_MIXED_ACCEPTANCE_ACCEPTED=YES
READY_FOR_PHASE6_REAL_TTIR_INVENTORY_AND_IMPORTER_SKELETON=YES
READY_FOR_TRITON=NO
```

Phase 5 locks only the handwritten V1 elementwise lowering subset:
rank-1 contiguous i32/f32 `#vc4value.global` transfer reads and writes,
`vector<16>` i32/f32 compute, finite-policy f32 compare/select, i32 compare,
tail masks, TMU inactive-zero safe-offset loads, and VDW inactive-preserve
stores. Non-16 vectors, rank-2 vectors and memrefs, subword/f16 memory paths,
reductions, contracts, gather/scatter, SFU/math lowering, control-flow
lowering, and Triton/TTIR ingestion remain staged and are not made value-surface
rejects by this lock.

The guide is a planning contract for future phases. It describes the decisions
those phases must make before creating VC4Kernel IR from:

```text
func + tiny vc4value + vector + memref + arith + math + scf/cf
```

Feature classification remains in
`compiler/docs/vc4_value_ttir_feature_taxonomy.json`.
The Phase 3.5 vector abstraction boundary is recorded in
`compiler/docs/vc4_value_surface_abstraction_policy.md`.
The Phase 4 public value ABI is recorded in
`compiler/docs/vc4_value_kernel_abi.md`.
Executable value-lowering hardware proof requirements are recorded in
`compiler/docs/vc4_value_hardware_verification_policy.md`.

## 1.1 Phase 16 attention-apply v0 planned coverage

PHASE16_VALUE_ATTENTION_APPLY_V0_CONTRACT=LOCKED
VALUE_ATTENTION_APPLY_V0_METADATA_POLICY=PASS
VALUE_ATTENTION_APPLY_V0_SURFACE=ACCEPTED
VALUE_ATTENTION_APPLY_V0_STATIC=PASS
VALUE_ATTENTION_APPLY_V0_COMPOSITE=YES
PRECOMPUTED_SCORES_ONLY=YES
TRANSPOSED_V_LAYOUT_REQUIRED=YES
SCALAR_GLOBAL_LOAD_STAGED=YES
NONTRANSPOSED_V_GATHER_STAGED=YES
K_ZERO_ATTENTION_APPLY_STAGED_OR_GUARD_REQUIRED=YES
READY_FOR_PHASE16_4_VALUE_ATTENTION_APPLY_STATIC=YES
READY_FOR_PHASE16_5_VALUE_HARDWARE_ISOLATION=YES
READY_FOR_TRITON=NO

Phase 16.3 is a planning and verifier contract only. Phase 16.4 proves static
value-to-VC4Kernel lowering for the accepted forms below by composing the
already locked Phase 11 row-slice memory planner, Phase 12 scalar-store and
reduction planner, Phase 14 f16-storage planner, and Phase 15 natural-exp
softmax/SFU planner. No first-class attention op or new dot/contract lowering
is introduced.

Accepted static coverage:

- attention-apply v0 composite lowers from standard value IR;
- scaled-score variant lowers when scale is a scalar f32 argument or constexpr
  splat;
- f16 score/Vt storage lowers through Phase 14 f16 storage to f32 compute;
- row-contiguous score loads and transposed-V row-slice loads lower through the
  existing rank-1 or Phase 11 row-strided memory paths;
- scalar output store lowers through the Phase 12 scalar store path;
- natural `math.exp`, reciprocal/division, f32 max reduction, and add
  reductions lower through the Phase 15 softmax/SFU and Phase 12 reduction
  contracts.

Staged/static negative coverage:

- scalar global loads remain staged, including scale loaded from memory;
- non-transposed V gather/lane-varying stride remains staged;
- K=0 remains staged unless a later phase proves a finite no-op guard;
- exact/default math remains rejected without explicit approximate-SFU policy;
- QK score generation, `tl.dot`, `tt.dot`, `vector.contract`, block pointers,
  full attention, and FlashAttention remain out of scope.

## 1.2 Phase 17 online softmax state planned coverage

PHASE17_VALUE_ONLINE_SOFTMAX_STATE_CONTRACT=LOCKED
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
SCALAR_GLOBAL_LOAD_STAGED=YES
NONTRANSPOSED_V_GATHER_STAGED=YES
READY_FOR_PHASE17_4_VALUE_ONLINE_SOFTMAX_STATIC=YES
READY_FOR_PHASE17_5_VALUE_HARDWARE_ISOLATION=YES
READY_FOR_PHASE17_6_VALUE_MIXED_ACCEPTANCE=YES
READY_FOR_PHASE17_7_TTIR_IMPORTER_ONLINE_SOFTMAX_STATIC=YES
READY_FOR_TRITON=NO

Phase 17.3 was a planning and verifier contract only. Phase 17.4 proves static
value-to-VC4Kernel lowering for the accepted structural value forms below by
composing already locked control-flow, row-slice memory, scalar-store,
reduction, f16-storage, and natural-exp SFU planning, plus a narrow central
scalar finite f32 max helper needed by the online recurrence.

Accepted planned coverage:

- online softmax state is a standard value IR composite with `scf.for` or
  canonical `cf` loop-carried scalar f32 `m`, `l`, and `acc`;
- each loop iteration processes at most one `vector<16xf32>` block;
- score loads are precomputed-score `vector.transfer_read` operations;
- optional score scale is a scalar f32 kernel argument or constexpr splat;
- inactive scores are selected to a finite low value before block max;
- block max uses finite f32 max reduction and scalar finite max for `m_new`;
- `alpha`, `beta`, shifted-score exponentials, and final division use explicit
  approximate-SFU policy inherited from Phase 15;
- V loads require transposed V layout so lanes are contiguous over K,
  `VT[d, start + lane]`;
- output is a scalar f32 store to `O[q * LDO + d]`;
- f16 score/Vt storage may be used only through Phase 14 f16-to-f32 compute.

Staged/static negative coverage:

- K=0 remains staged unless a later phase proves an explicit finite no-op
  guard;
- scalar global loads remain staged, including scale loaded from memory;
- non-transposed V gather/lane-varying stride remains staged;
- QK score generation remains outside Phase 17 accepted scope;
- `tl.dot`, `tt.dot`, `vector.contract`, block pointers, tensor descriptors,
  cooperative/VPM tiling, full attention, and FlashAttention remain out of
  scope.

Phase 17.4 static proof covers:

- online softmax normalizer f32;
- online precomputed-score attention-apply f32;
- scalar-argument scaled online attention-apply f32;
- f16 score/Vt storage inputs promoted to f32 compute;
- loop-carried scalar f32 online state represented in VC4Kernel as f32
  fragments across CFG block arguments;
- lowering through `value -> vc4kernel -> ssavc4 -> scheduled vc4` for
  static CodeGen/VC4Value candidates.

No hardware is run in Phase 17.4. Hardware isolation is Phase 17.5.

Phase 17.5 hardware isolation proves the online/multiblock softmax value
composites on real VC4 hardware. The isolation fixtures run with
`active_qpus=12`, strict sentinels, host CPU oracles, nonzero output hashes,
and expected JSON checks. Coverage includes:

- online softmax normalizer f32 over K blocks with loop-carried `m/l` state;
- precomputed-score online attention-apply f32 with transposed V and
  loop-carried `m/l/acc` state;
- scalar f32 scale argument attention-apply;
- repeat launches with K=1, K=17, and K=64 to check invocation isolation.

The Phase 17.5 tolerance policy remains tied to the Phase15 natural-exp
approximate-SFU policy. The normalizer denominator fixture uses
absolute/relative caps `0.12/0.025`, with observed hardware
`max_abs_diff=0.001731` and `max_rel_diff=0.000093`. Attention fixtures use
absolute cap `0.012` and relative cap `0.08` because near-zero weighted outputs
can have high relative error while the absolute error remains small; observed
hardware `max_abs_diff` is at most `0.000071`. The checker still requires zero
mismatches, zero sentinel mismatches, nonzero output hashes, and structural
claim audits, so the tolerance does not hide wrong reductions or wrong exp
base. K=0, scalar global loads, non-transposed V gather, QK score generation,
dot/contract, full attention, and FlashAttention remain staged.

## 2. Planning boundary

The value surface expresses standard semantics: logical functions, memrefs,
vectors, masks, scalar arithmetic, math, and control flow.

The planner chooses target mechanisms such as TMU, VDR, VPM, and VDW. Those are
not value-surface operations. They are VC4Kernel planning outputs when the value
pattern is legal.

Verified VC4Kernel output may contain only accepted VC4Kernel operations,
attributes, and carrier types from the locked Surface v2 contract. Producer
dialect operations must not appear inside verified VC4Kernel.

VC4Kernel lowers only through SSAVC4. There is no direct VC4KernelToVC4 path.
VC4Tile is retired and must not be revived as a value-layer or planner
intermediate.

## 3. Launch identity planning

Phase 9 multi-axis launch identity contract:

```text
PHASE9_FEATURE=VALUE_AND_TTIR_MULTI_AXIS_LAUNCH_IDENTITY
VALUE_MULTI_AXIS_SURFACE_CONTRACT=LOCKED
VC4VALUE_GRID_RANK_1_2_3_EXECUTABLE_FOR_LAUNCH_IDENTITY=YES
VC4VALUE_PROGRAM_ID_AXES_0_1_2_LOWERABLE_NOW=YES
VC4VALUE_NUM_PROGRAMS_AXES_0_1_2_LOWERABLE_NOW=YES
PHASE9_MEMORY_MODEL=FLATTENED_RANK1_ONLY
PHASE9_MASK_MODEL=CANONICAL_LINEARIZED_TAIL_ONLY
MASK_CLASSIFIER_SCOPE_CREEP=NO
RANK2_MEMORY_SCOPE_CREEP=NO
READY_FOR_TRITON=NO
```

`vc4value.program_id {axis = 0|1|2}` maps to the VC4Kernel program-id operation
for the same logical axis.

`vc4value.num_programs {axis = 0|1|2}` maps to VC4Kernel num-programs behavior
and associated resource/launch metadata.

Axes are logical launch-grid axes only. They are not physical QPU IDs and must
not expose physical scheduling to the value layer.

For Phase 9, `vc4value.grid_rank` values 1, 2, and 3 are executable for launch
identity. The logical mapping from a flattened launch request to grid
coordinates is:

```text
pid0 = logical_block_id % grid.x
pid1 = (logical_block_id / grid.x) % grid.y
pid2 = logical_block_id / (grid.x * grid.y)
```

`vc4value.num_programs(axis)` returns `grid.x`, `grid.y`, or `grid.z` for axes
0, 1, and 2 respectively.

Phase 9 multi-axis launch identity does not change the memory model. Memory
remains flattened rank-1 only. Multi-axis kernels may compute a linearized
rank-1 element index from logical grid coordinates, but rank-2 memory,
rank-2 transfer planning, gather/scatter, block pointers, tensor descriptors,
and VPM tile planning remain staged.

Phase 9 also does not add a general mask classifier. The only accepted dynamic
mask form for this feature is the canonical linearized tail:

```text
idx = flattened_block_id * 16 + lane
mask = idx < n
```

Index values may remain `index` in the value surface, but the planner must prove
the target i32 conversion policy before constructing VC4Kernel scalar i32
operands.

## 4. ABI and memref lowering plan

A value memref base pointer lowers later to a raw i32 uniform for VC4Kernel.
Dynamic sizes, strides, offsets, and other ABI values lower to scalar arguments
or metadata after the value-layer ABI plan proves their type and range.

Phase 4 locks the producer-facing public memref ABI as rank-1/rank-2
`#vc4value.global` memrefs with element types:

```text
i8, i16, i32, f16, f32
```

Dynamic dimensions are named by `vc4value.shape_args`; dynamic strides are
named by `vc4value.stride_args`. There is no hidden memref descriptor ABI.
`memref.dim` is metadata-only and must resolve to explicit extent arguments or
static dimensions before target lowering.

The first Phase 5 V1 lowerable memory subset is narrower: rank-1 contiguous
i32/f32 global memrefs for elementwise kernels. i8/i16/f16 memrefs, rank-2
memrefs, and dynamic-stride layouts remain staged until the planner implements
the relevant subword, f16-storage, or VPM/tile paths.

Read-only, write-only, and inout attributes are planning hints. They can guide
TMU, VDR, and VDW path selection, but they are not permission to weaken source
memory semantics or skip required inactive-lane handling.

## 5. Lane and fragment planning

`vector.step` maps to VC4Kernel lane identity / `lane_range` equivalent in the
Phase 5 V1 subset when the vector shape is `vector<16xindex>` or can be safely
converted to the target i32 lane carrier.

`vector<16xi32>` and `vector<16xf32>` are the first lowerable VC4Kernel
fragment carrier types for executable arithmetic. `vector<16xi1>` and
`vector<16xindex>` are support forms for masks and address/program-id
arithmetic. `vector<16xT>` is not the global value-layer type limit.

Fixed rank-1 `vector<NxT>` with `N != 16` is surface-admissible and staged for
later splitting into `vector<16>` fragments plus tails or loops. Fixed rank-2
`vector<MxNxT>` is surface-admissible and staged for later tile, contract, and
VPM planning. The value planner must not invent wider VC4Kernel fragment
carriers.

`vector.splat` and constants map to VC4Kernel `splat` and `fragment_const` where
the element type and encoding are legal. Constants that are not directly
encodable must be decomposed into accepted operations or diagnosed.

## 6. Mask classifier

The planner must classify masks before memory lowering:

- full: all lanes active;
- empty: no lanes active;
- tail: active prefix;
- rect: dense rectangular active region;
- sparse: arbitrary lane pattern;
- unknown: not proven to be one of the accepted forms.

Required decisions:

- full -> accepted for compute, load, and store;
- empty -> no-op/store skip or identity path;
- tail -> accepted for TMU safe load and VDW preserve store;
- rect -> accepted for tile/VDW rect where representable;
- sparse -> compute predicate may be legal, but store rejects unless future
  hardware-proven support exists;
- unknown -> reject for stores, and reject or require canonicalization for
  loads depending on the selected path.

Sparse and unknown stores must not silently lower to sparse VDW stores.

Phase 10 refines this into the locked value mask classifier contract:

- `FULL`: no transfer mask, or `vector.create_mask` active count at least 16
  after clamping.
- `EMPTY`: `vector.create_mask` active count at most 0 after clamping.
- `TAIL_0_TO_16`: `vector.create_mask %count : vector<16xi1>`, clamped to
  `[0, 16]`, with active lanes starting at lane 0.
- `COMPUTE_MASK`: `vector<16xi1>` compare/logical masks used only by compute
  operations such as `arith.select`. These are accepted as compute masks, not
  as memory transfer predicates.
- `SPARSE_OR_UNKNOWN_MEMORY_MASK`: any transfer mask not classified as full,
  empty, or tail. Stores deterministically stage/reject; loads remain staged
  unless a later phase implements exact support.
- `RECT`: staged until rank-2/tile memory planning.

The classifier must be structural: typed operation classes, exact operation
names, attributes, and SSA use-defs decide semantics. It must not parse printed
IR or special-case fixture names, paths, public names, or generated-output
locations.

## 6.1 Phase 10 memory legality

The accepted Phase 10 value memory legality subset is:

- rank-1 memrefs in `#vc4value.global`;
- identity layout;
- element type `i32` or `f32`;
- `vector<16xi32>` or `vector<16xf32>` transfers;
- rank-1 identity transfer permutation maps;
- scalar index base;
- load padding/`other` exactly zero;
- transfer-read inactive lanes through safe-offset inactive-zero;
- transfer-write inactive lanes through inactive preserve.

The staged or rejected Phase 10 memory forms are:

- nonzero load padding/`other`;
- non-identity transfer maps;
- rank greater than 1 memrefs;
- strided layouts;
- tensor/block pointer memory;
- gather/scatter;
- sparse or unknown store masks;
- subword or f16 memory;
- boundary-check/padding-option producer semantics;
- vector rank greater than 1.

Phase 10.3 is a contract lock only. Planned Phase 10.4 static coverage must
cover full/unmasked transfers, empty masks, clamped tails, compute-mask selects
that are not memory masks, sparse transfer read/write staging, nonzero `other`
staging, and non-identity map staging before any hardware claim.

## 7. Elementwise arithmetic/cmp/select planning

Value `arith` and `vector` elementwise operations map to VC4Kernel fragment ALU
where opcode, type, and policy are accepted.

i32 add, sub, mul, bitwise operations, shifts, min/max, comparisons, constants,
splats, and selects map through the accepted fragment ALU, compare, const,
splat, and select families.

f32 arithmetic maps only for the accepted basic target arithmetic set. f32
comparisons require explicit finite policy before lowering to VC4Kernel finite
f32 compare operations.

Unsupported element type, unsupported opcode, missing finite policy, or an
unclassified mask must produce a deterministic diagnostic before VC4Kernel IR is
constructed.

## 8. Transfer-read planning

Transfer-read planning chooses among several target paths:

- simple contiguous/tail low-reuse load -> TMU safe-offset inactive-zero;
- structured rank-2, tile, or reuse load -> VDR -> VPM -> QPU VPM read;
- gather load later -> TMU direct address;
- unsupported or unclassified path -> diagnostic.

The value layer does not name TMU, VDR, VPM, or VDW. These are planner choices.

Inactive value, pad, or `other` handling must preserve source semantics. For
the initial contiguous/tail profile, inactive lanes can use the locked
VC4Kernel TMU safe-offset inactive-zero contract only when the value operation
requires zero or when the planner can legally materialize the requested
inactive value through a subsequent value operation.

No old TMU signature or inferred safe address is allowed. The planner must
provide explicit safe-offset behavior for inactive lanes.

## 9. Transfer-write planning

Transfer-write planning chooses among:

- contiguous/tail store -> VDW register-fragment inactive preserve;
- rect/tile store -> VPM/VDW path;
- sparse store -> reject;
- unknown mask -> reject until classified or emulated by a specified future
  path.

Sparse stores cannot lower to sparse VDW stores. Sparse VDW stores remain a
locked deterministic reject unless a future hardware-proven phase changes the
VC4Kernel surface.

The planner must not use branchy kernel reshaping to avoid compiler bugs. It
may use correct control-flow or masking plans, but those plans must preserve
source semantics and lower through accepted VC4Kernel forms.

## 10. Gather/scatter planning

Gather loads are supportable through future TMU planning when address formation,
coalescing assumptions, inactive-lane safety, and mask legality are specified.
They are not a Phase 1 implementation claim.

Scatter stores remain a reject/emulation candidate. They cannot be lowered to
sparse VDW. A future value-level emulation plan must prove source semantics and
must not alter the locked direct sparse VDW reject.

## 11. VPM/VDR/VDW tile planning

Value IR remains `memref`, `vector`, and `scf/cf`. There is no tile DSL and no
source-visible VPM, VDR, or VDW operation.

The planner may recognize structured patterns and emit accepted VC4Kernel
operations for:

- VPM allocation;
- QPU VPM read/write;
- VDR global-to-VPM movement;
- VDW VPM/register-fragment stores;
- barriers and semaphores where the cooperative contract requires them;
- resource metadata.

Structured tile planning must preserve memref shape, stride, layout, mask, and
element policy until a legal VC4Kernel path is selected.

## 12. P12 coordinate and selector model

P12 row, word-X, and subword selector are separate fields.

Dynamic subword selector is byte/halfword selector only. It does not mean
dynamic width, dynamic subword mode, dynamic orientation, or dynamic layout.

Setup field masks are not modulo semantics. They isolate VC4 hardware fields
and do not legalize out-of-range runtime values.

VPM QPU read/write normalized readback differs from raw DMA carrier placement.
The planner must not conflate QPU-normalized subword readback with VDR or VDW
carrier placement.

VDR and VDW selector semantics are separately proven. Never infer VDW behavior
from VDR symmetry. VDW has its own setup composition and preserve-store
constraints.

Vector or lane-varying coordinate operands are rejected for VC4Kernel dynamic
coordinate paths.

## 13. Subword and f16 storage planning

i8 and i16 storage paths must use exact mode-table entries in future
implementation. Each accepted subword path must cite its width, packing mode,
orientation, selector range, extension/truncation policy, and memory path.

f16 storage conversion plus f32 compute is the accepted f16 model. The planner
may lower f16 storage through fragment pack/unpack and f32 compute carriers
when the value policy permits it.

Native f16 arithmetic is rejected. Native bf16/fp8 arithmetic and conversion
are rejected by the locked VC4Kernel surface. Future value-level emulation
would require a separate proof and taxonomy update.

Dynamic selector support does not extend to dynamic width, dynamic mode,
dynamic orientation, or dynamic layout.

## 14. Reduction planning

`vector.reduction` maps to `vc4kernel.fragment_reduce` only after type, kind,
mask, identity, and math policy are proven.

PHASE12_VALUE_REDUCTION_CONTRACT=LOCKED
VALUE_REDUCTION_TO_VC4KERNEL_STATIC=PASS
VALUE_VECTOR_REDUCTION_ADD_I32_SURFACE=ACCEPTED
VALUE_VECTOR_REDUCTION_ADD_F32_FINITE_SURFACE=ACCEPTED
VALUE_SCALAR_MEMREF_STORE_FOR_REDUCTION_SURFACE=ACCEPTED
VALUE_VECTOR_REDUCTION_ADD_I32_STATIC=PASS
VALUE_VECTOR_REDUCTION_ADD_F32_FINITE_STATIC=PASS
VALUE_SCALAR_STORE_FOR_REDUCTION_STATIC=PASS
F32_REDUCTION_FINITE_TREE_POLICY=YES
NON_ADD_REDUCTIONS_STAGED=YES
DOT_GEMV_STAGED_FOR_PHASE13=YES
READY_FOR_PHASE12_4_VALUE_REDUCTION_STATIC=YES
READY_FOR_PHASE12_5_VALUE_HARDWARE_ISOLATION=YES
READY_FOR_TRITON=NO

Phase 12 accepts the following value-surface reduction contract for later
executable lowering:

- `vector.reduction <add>` over `vector<16xi32>` to scalar `i32`.
- `vector.reduction <add>` over `vector<16xf32>` to scalar `f32` only with
  explicit finite-input and finite-tree policy.
- tail reductions by inactive-zero input lanes from the Phase 10 transfer-read
  contract; the reduction operation itself remains unmasked.
- scalar `memref.store` of `i32`/`f32` reduction results to rank-1
  `#vc4value.global` output memrefs. The planner may implement this internally
  through one-lane VDW inactive-preserve stores when Phase 12.4 adds executable
  lowering.

i32 add reductions are exact for the accepted integer reduction policy.

f32 reductions require finite-tree or explicit approximate policy only. Strict
IEEE f32 reductions must not silently lower to finite-tree VC4Kernel reduction.
They require an exact/emulated plan or a diagnostic.

Masked reductions are not a separate Phase 12 source form. Tail reductions use
the Phase 10 inactive-zero load contract so inactive lanes already hold the add
identity before the unmasked `vector.reduction`.

The Phase 12.4 planned static coverage is:

- `vector.reduction <add>` over `vector<16xi32>` lowers.
- `vector.reduction <add>` over `vector<16xf32>` with finite-tree policy lowers.
- missing f32 finite-tree policy rejects.
- scalar `memref.store` of `i32`/`f32` reduction outputs lowers.
- tail inactive-zero reduction lowers.
- row-strided sum value patterns lower through the Phase 11 address plan.
- max, min, product, and custom reductions reject.
- rank greater than 1 reductions reject.
- `vector.multi_reduction` rejects.

Max/min/product/custom reductions, `vector.multi_reduction`, rank greater than
1 reductions, scans, atomics, f16/subword reductions, exact/default f32
reductions without finite policy, dot, GEMV, and GEMM remain staged.

Phase 12.4 implements and statically proves the accepted value executable
subset:

- `vector.reduction <add>` over `vector<16xi32>` lowers to
  `vc4kernel.fragment_reduce` with `#vc4kernel.reduce<add>`.
- `vector.reduction <add>` over `vector<16xf32>` lowers to
  `vc4kernel.fragment_reduce` with
  `#vc4kernel.fp_reduce_policy<finite_tree>` only when the explicit finite
  policy is present.
- scalar `memref.store` of `i32`/`f32` reduction-output values lowers through
  a `vc4kernel.splat` or reduction fragment, a scalar-indexed VDW byte offset,
  a lane-zero `vc4kernel.pred.tail`, and
  `vc4kernel.vdw_store_fragment` with inactive-preserve policy.
- row-strided reduction inputs reuse the Phase 11 central address planner.

No hardware proof is claimed by Phase 12.4; Phase 12.5 is the hardware
isolation phase.

## 14.1 Phase 13 GEMV / row-wise dot planning

PHASE13_VALUE_GEMV_ROWWISE_DOT_CONTRACT=LOCKED
VALUE_GEMV_ROWWISE_DOT_STATIC=PASS
VALUE_GEMV_F32_ROW_DOT_STATIC=PASS
VALUE_GEMV_I32_ROW_DOT_STATUS=STAGED_BY_I32_POLICY
VALUE_GEMV_PARTIAL_KBLOCK_STATIC=PASS
VALUE_GEMV_ROWWISE_DOT_HARDWARE_ISOLATION=PASS
VALUE_GEMV_F32_ROW_DOT_HARDWARE=PASS
VALUE_GEMV_I32_ROW_DOT_HARDWARE=NOT_REQUIRED_STAGED_BY_I32_POLICY
VALUE_GEMV_PARTIAL_KBLOCK_HARDWARE=PASS
VALUE_GEMV_EMPTY_REPEAT_HARDWARE=PASS
VALUE_GEMV_ROWWISE_DOT_MIXED_ACCEPTANCE=PASS
VALUE_MIXED_REGRESSION=PASS
VALUE_GEMV_ROWWISE_DOT_F32_SURFACE=ACCEPTED
VALUE_GEMV_ROWWISE_DOT_I32_SURFACE=ACCEPTED
VALUE_TO_VC4KERNEL_PLANNED_COVERAGE=YES
F32_DOT_FINITE_TREE_POLICY=YES
TL_DOT_TT_DOT_STAGED=YES
VECTOR_CONTRACT_STAGED=YES
MULTIBLOCK_K_ACCUMULATION_STAGED=YES
READY_FOR_PHASE13_4_VALUE_GEMV_STATIC=YES
READY_FOR_PHASE13_5_VALUE_HARDWARE_ISOLATION=YES
READY_FOR_PHASE13_6_VALUE_MIXED_ACCEPTANCE=YES
READY_FOR_PHASE13_7_TTIR_IMPORTER_GEMV_STATIC=YES
READY_FOR_TRITON=NO

Phase 13 GEMV-v0 lowers only the explicit value composite:

```text
vector<16xT> load A row slice
vector<16xT> load X vector slice
arith.mul{f,i}
vector.reduction <add>
scalar memref.store
```

There is no first-class value dot op. The planner must recognize the standard
SSA composition structurally and lower through the existing elementwise,
memory, reduction, and scalar-store planning layers. It must not special-case
fixture names, source variable names, TTIR snapshot paths, or printed IR.

The accepted f32 form is finite-input finite-tree only. `arith.mulf` feeds
`vector.reduction <add>` over `vector<16xf32>` to a scalar `f32`, with
`vc4value.fp_domain = "finite"` and
`vc4value.reduction_policy = "finite_tree"` present on the reduction or
containing kernel. The target result is not exact IEEE left-to-right summation
and not an FMA contraction.

The i32 dot composite is staged by the current integer multiply policy.
Phase 13.5 hardware isolation showed that the existing
`vc4value.i32_mul_policy = "mul24_safe"` path does not prove exact signed i32
dot results after multiply and reduction. Standalone i32 reductions and
standalone i32 mul24 policy uses remain governed by their existing contracts.

Tail dots reuse Phase 10 inactive-zero loads. The reduction itself is
unmasked; inactive lanes must already contain the add identity. Row-wise GEMV-v0
uses Phase 11 row-strided memory for A, rank-1 contiguous or scalar-strided
memory for X, K at most one `vector<16>` block for full row-dot output, and the
Phase 12 scalar reduction-output store for Y.

Partial K-block dots are accepted only as independent partial outputs: one
program computes one `(row, kblock)` product/reduction and stores one scalar
partial. Cross-block accumulation into final `y[row]`, atomics, and
cross-program accumulation remain staged.

Planned Phase 13.4 static coverage is recorded in
`compiler/docs/vc4_vector_triton_phase13_value_to_vc4kernel_planned_coverage.md`
and must cover:

- f32 vector multiply -> finite-tree f32 add reduction -> scalar store;
- i32 vector multiply -> i32 add reduction -> scalar store stages by current
  i32 mul policy;
- tail inactive-zero dot;
- row-strided dot value pattern;
- partial K-block dot value pattern;
- vector.contract staged/reject;
- exact/default f32 dot without finite-tree policy reject;
- multi-block K accumulation staged/reject when outside Phase 13 scope.

`tl.dot`, `tt.dot`, `vector.contract`, rank-2 tile values, fma policy,
multi-block K accumulation, atomics, f16/subword dot inputs, and exact/default
f32 dot remain staged.

Phase 13.4 statically proves this composite path through existing lowerers:
`arith.mulf` lowers through `vc4kernel.fragment_alu.mul`,
`vector.reduction <add>` lowers through `vc4kernel.fragment_reduce`, and the
scalar result store lowers through the Phase 12 one-lane VDW
inactive-preserve scalar-store path. No `vc4kernel.dot` or GEMV-specific target
operation is introduced. The i32 multiply-plus-add-reduction dot composite is
staged until an exact signed i32 dot policy is locked.

The i32 row-dot status is `STAGED_BY_I32_POLICY`. Missing i32 multiply policy
and present-but-insufficient i32 dot policy remain deterministic rejects.

The Phase 13.4 static reject coverage preserves:

- `vector.contract is staged for Phase 15/contract`;
- `tt.dot is staged`;
- `multi-block K accumulation is staged`;
- `exact f32 dot requires unsupported exact reduction policy`;
- `unsupported GEMV element type`.

Phase 13.5 hardware isolation proves the f32 row-dot, partial K-block, and
empty/repeat value fixtures on real hardware with `active_qpus=12`. I32 row-dot
hardware is not required because i32 dot composites are staged by the current
i32 multiply policy.

Phase 13.6 proves the value GEMV row-dot path in the cumulative VC4Value mixed
acceptance suite. The mixed fixture combines Phase 5 elementwise paths, Phase
8/8.5 control flow, Phase 9 multi-axis launch, Phase 10 masks and compute-mask
select, Phase 11 row-strided memory, Phase 12 f32 finite reductions and scalar
stores, and Phase 13 f32 row-wise dot plus partial K-block dot. The full mixed
suite passed on hardware with `active_qpus=12` where applicable and no result,
sentinel, or launch failures. `tl.dot`, `tt.dot`, `vector.contract`, and
multi-block K accumulation remain explicitly staged.

## 14.2 Phase 14 ML storage and numeric conversion planning

PHASE14_VALUE_ML_STORAGE_NUMERIC_CONTRACT=LOCKED
VALUE_F16_STORAGE_TO_F32_COMPUTE_SURFACE=ACCEPTED
VALUE_F32_COMPUTE_TO_F16_STORAGE_SURFACE=ACCEPTED
F16_STORAGE_FINITE_POLICY=YES
I32_TO_F32_CAST_STATUS=STAGED_BY_LOWER_HALF_GAP
NATIVE_F16_ARITHMETIC_STAGED=YES
BF16_FP8_STAGED=YES
INT8_INT16_QUANTIZED_STORAGE_STAGED=YES
READY_FOR_PHASE14_4_VALUE_STORAGE_NUMERIC_STATIC=YES
READY_FOR_TRITON=NO

Phase 14 accepts the following value-surface contract for planned lowering:

- f16 storage loads from Phase 10 rank-1 identity or Phase 11 row-strided
  contiguous lane transfers may produce `vector<16xf16>`, then immediately
  widen through `arith.extf` to `vector<16xf32>` for compute.
- f32 compute values may narrow through
  `arith.truncf vector<16xf32> -> vector<16xf16>` and write to f16 storage only
  when the transfer or containing kernel carries
  `vc4value.f16_storage_policy = "finite"`.
- Phase 13 row-wise dot may use f16 A/X storage inputs when both inputs widen
  to f32 before multiply and the reduction remains the existing finite-tree f32
  add reduction.
- Inactive f16 load lanes reuse the Phase 10 inactive-zero policy. The f16
  transfer `other` value must be zero.

The planner must map accepted f16 storage through the locked VC4Kernel
pack/unpack storage-conversion surface. It must not create native f16 ALU,
f16 reduction, bf16/fp8 conversion, int8/int16 quantized storage policy, SFU
math, `tt.dot`, `vector.contract`, or GEMM forms in Phase 14.

The i32/index-to-f32 cast status is `STAGED_BY_LOWER_HALF_GAP`. Phase 14.4 may
only change this status if it proves a real lower-half path. Until then,
`arith.sitofp`/`arith.uitofp` must diagnose:
`i32 to f32 numeric cast staged by lower-half gap`.

Planned Phase 14.4 static coverage is recorded in
`compiler/docs/vc4_vector_triton_phase14_value_to_vc4kernel_planned_coverage.md`
and must cover:

- f16 `vector.transfer_read` plus `arith.extf` lowers;
- f32 compute plus `arith.truncf` plus f16 `vector.transfer_write` lowers;
- f16 row-strided storage lowers;
- f16 row-dot input storage lowers through f32 compute;
- f16 store missing finite policy rejects;
- native f16 arithmetic rejects;
- f32-to-i32 casts reject;
- bf16/fp8/int8 staged forms reject;
- i32-to-f32 stages by lower-half gap unless Phase 14.4 proves support.

No hardware proof is claimed by Phase 14.3. No executable lowering is
implemented by this contract lock.

## 15. Math and SFU planning

Public `math.exp`, public `math.log`, reciprocal, and rsqrt-like value patterns
may map to approximate SFU only under explicit approximate policy.

Exact/default math must either lower through an exact sequence or reject with a
diagnostic. It must not silently lower to approximate SFU.

There is no direct SFU sqrt. Approximate sqrt may be planned only as an explicit
approximate composite, such as rsqrt plus arithmetic, when source policy allows
it.

Exp/log base semantics must be explicit. Target SFU `exp` and `log` are base-2
hardware modes. Public value `math.exp` is natural exp and public value
`math.log` is natural log; they are not interchangeable with target exp2/log2
without `LOG2E`/`LN2` scaling and explicit approximate-SFU policy.

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

Phase 15.3 locks the value-surface contract for approximate SFU math and
one-block softmax. Phase 15.4 must statically prove, without hardware:

- natural `math.exp` over scalar `f32` and `vector<16xf32>` lowers only when
  the containing function or operation carries `vc4value.math_policy =
  "approx_sfu"` and finite `vc4value.fp_domain` metadata. Phase 15B repairs the
  lowering to target exp2 with `x * log2(e)` scaling.
  `VALUE_NATURAL_EXP_STATIC=PASS`.
- `arith.divf` lowers only as approximate reciprocal/division under explicit
  approximate-SFU policy and a finite nonzero denominator domain. Exact
  division semantics are not claimed.
- natural `math.log` lowers through target log2 with `ln(2)` scaling, and
  `math.rsqrt` may lower because VC4Kernel Surface v2 already locks rsqrt.
  Both require positive finite domains. Public `math.sqrt` is a sqrt operation
  and lowers through `x * rsqrt(x)` only under explicit positive finite policy.
  `VALUE_NATURAL_LOG_STATIC=PASS` and `VALUE_SQRT_STATIC=PASS`.
- finite f32 max reductions lower for the exact value spellings
  `vector.reduction <maxnumf>` and `vector.reduction <maximumf>` over
  `vector<16xf32>` to scalar `f32`, with explicit finite input,
  finite-tree reduction, and finite max policy metadata.
- scalar-to-vector f32 broadcasts lower through `vc4kernel.splat` or the
  accepted target composite.
- softmax v0 lowers only as the value composite:
  finite max reduction, scalar broadcast, subtract, approximate exp, masked
  zeroing, finite add reduction, approximate reciprocal/division, multiply,
  and masked `vector.transfer_write`. `SOFTMAX_USES_NATURAL_EXP=YES`.

Phase 15.4 planned rejects:

- exact/default math without approximate-SFU policy;
- NaN/Inf math policy;
- generic division without approximate policy;
- active-count-zero softmax without an explicit finite no-op guard;
- multiblock softmax;
- block-pointer softmax;
- full attention and FlashAttention;
- quantized softmax;
- `tt.dot`, `vector.contract`, and GEMM.

This planning entry does not implement executable lowering and does not claim
hardware proof.

Phase 15B.2 hardware re-proves the corrected natural-math value path:

VALUE_EXP_NATURAL_HARDWARE=PASS
VALUE_LOG_NATURAL_HARDWARE=PASS
VALUE_SQRT_HARDWARE=PASS
VALUE_RSQRT_HARDWARE=PASS
VALUE_SOFTMAX_NATURAL_EXP_REPROOF=PASS
NATURAL_EXP_LOWERING=EXP2_X_LOG2E
NATURAL_LOG_LOWERING=LOG2_X_LN2
SQRT_LOWERING=RSQRT_TIMES_X
EXP2_LOG2_TARGET_ONLY_CLAIMS_HONEST=YES
READY_FOR_PHASE15B_3_TTIR_NATURAL_MATH_STATIC_BRIDGE=YES
READY_FOR_TRITON=NO

The Phase 15B.2 fixtures keep `active_qpus=12`, strict sentinels, expected JSON
checking, and natural exp/log/sqrt/rsqrt host oracles. The old Phase 15.5
target-mode exp2/log2 proof remains target-only and is not a public natural
math proof.

Phase 15.6 mixed value acceptance extends the cumulative VC4Value mixed suite
with `mixed_value_sfu_softmax_axes_mask_cf_f16_storage_vc4value`. The mixed
fixture keeps the standard value-to-VC4Kernel layering, combines axes, control
flow, masks, row-strided f16 storage, finite reductions, broadcast,
approximate reciprocal/division, and one-block softmax v0, and uses the
Phase 15B public natural-exp softmax oracle rather than target exp2 semantics.

VALUE_APPROX_SFU_SOFTMAX_MIXED_ACCEPTANCE=PASS
VALUE_MIXED_REGRESSION=PASS
VALUE_MIXED_CLAIM_AUDIT=PASS
READY_FOR_PHASE15_7_TTIR_IMPORTER_SFU_SOFTMAX_STATIC=YES
READY_FOR_TRITON=NO

## 16. Shuffle, rotate, and lane-broadcast planning

Rotate patterns map to VC4Kernel dynamic rotate only when the amount is scalar
and the modulo-16 lane semantics match the locked target rule.

Lane broadcast uses the canonical composite:

```text
lane_range -> compare lane == selected_lane -> select value or zero -> fragment_reduce add
```

For f32 bit-preserving broadcast, bitcast f32->i32, use the i32 composite, then
bitcast i32->f32.

Arbitrary direct shuffle/permutation is rejected at VC4Kernel. Value-level
emulation is future supportable, but it must be a specified composite or
emulation path, not a direct VC4Kernel arbitrary permutation.

## 17. Control-flow planning

Phase 8 is a value-layer control-flow phase, not TTIR control-flow import. The
executable value-to-VC4Kernel control boundary consumes `cf`, not raw `scf`.

PHASE8R_CF_COMPLETENESS_SCOPE=ACTIVE
SCF_WHILE_VALUE_TARGET=SUPPORT_NOW
NESTED_STRUCTURED_CF_VALUE_TARGET=SUPPORT_NOW
TL_RANGE_STYLE_LOOP_SKELETON_VALUE_TARGET=SUPPORT_NOW
PERSISTENT_LOOP_SKELETON_VALUE_TARGET=SUPPORT_NOW
VECTOR_BRANCH_CONDITION_POLICY=DETERMINISTIC_REJECT_AS_CFG_USE_MASKS
IRREDUCIBLE_CFG_POLICY=PROBE_NOT_REQUIRED_FOR_SANE_TRITON
READY_FOR_TRITON=NO

`scf` may appear in value input as a source convenience. Raw `scf` must not
survive into verified VC4Kernel. Before executable value-to-VC4Kernel lowering,
`scf.if`, `scf.for`, and `scf.while` must canonicalize through the upstream MLIR
`--convert-scf-to-cf` pass into `cf` branches and block arguments. The planner
must not grow a custom SCF lowering unless the upstream pass is proven unusable
and that blocker is recorded for user review.

QPU control flow is scalar/coherent across SIMD lanes. Per-lane control
divergence is not accepted as branch CFG; it is represented by masks/selects.
Nested structured control flow, tl.range-style loop skeletons, and
persistent-loop skeletons are value-layer targets before TTIR import when their
bodies use supported value features.

The canonical Phase 8 executable pipeline for structured value control flow is:

```text
value input with scf
  -> --convert-scf-to-cf
  -> --vc4-verify-value-surface
  -> --convert-vc4-value-to-vc4kernel
  -> --verify-vc4kernel
```

Pure `cf` inputs may enter `--vc4-verify-value-surface` directly. VC4Value
hardware/static runners that encounter `scf` input must preserve the
after-scf-to-cf intermediate IR before constructing VC4Kernel.

The Phase 8 V1 executable `cf` set is:

```text
cf.br
cf.cond_br with scalar i1 condition
func.return
```

Value block arguments are staged by type. The Phase 8 V1 candidate lowerable
set is exactly:

```text
index
i1
i32
f32
vector<16xi1>
vector<16xindex>
vector<16xi32>
vector<16xf32>
```

`index` and `vector<16xindex>` must lower to target i32 carriers before
verified VC4Kernel. `vector<16xi1>` masks must lower to accepted VC4Kernel
predicate or scalar condition forms. Memref block arguments, VPM tile block
arguments, vector-valued branch conditions, `cf.switch`, `scf.index_switch`,
`scf.parallel`, `scf.forall`, `scf.reduce`, irreducible CFG, and
exception-like control flow remain rejected or staged in Phase 8R. Irreducible
CFG is probe/classify only and is not required for sane Triton Phase 8.5
control-flow import.

Natural loops, block arguments, liveness, spills, and branch layout must be
preserved honestly. The planner must not reshape kernels to hide backend bugs.

Verified VC4Kernel output contains no producer dialect operations. It may
contain accepted VC4Kernel operations, accepted scalar `arith`, and
`cf.br`/scalar-i1 `cf.cond_br`; it must contain no raw `scf`, `vector`,
`memref`, `func`, `vc4value`, TTIR, `ssavc4`, or scheduled `vc4` operations.

## 18. Contract/GEMM planning

The first contract/GEMM planning shape is:

```text
C[pid_m, pid_n*16 + lane] += A[pid_m,k] * B[k,pid_n*16+lane]
BLOCK_M=1
BLOCK_N=16
BLOCK_K=4 first
```

`vector.contract` is the generic value input. There is no `vc4value.dot` and no
tile DSL.

The planner should lower the first row-fragment shape through vector fragments,
f32/i32 policy as appropriate, reductions, and later VPM/VDR/VDW tile planning.
No tensor-core, TF32, native f16 arithmetic, or direct TTIR-to-VC4Kernel path is
assumed.

## 19. Cooperative and resource planning

`num_warps` and `num_stages` are planner metadata. They are not SIMT source
semantics in the value layer and do not expose physical QPU identity.

Later internal VC4Kernel cooperative/resource planning may use metadata to
choose barriers, semaphores, VPM rows, staging rows, double buffering, and
runtime descriptors.

Resource metadata must remain internally consistent through VC4Kernel, SSAVC4,
scheduled VC4, artifact emission, generated C, manifests, and runtime
descriptors.

## 20. Fixture claim and verification discipline

Future value/Triton hardware fixtures must inherit the mixed fixture claim
discipline from locked VC4Kernel acceptance.

Any `saw_*`, `no_*`, or equivalent claim must be classified as one of:

```text
CHECKED_OUTPUT
CHECKED_AUDIT
PHASE_GUARD
```

Claims must be tied to checked output, checked audit evidence, or an explicit
phase guard. They must not rely on fixture names, paths, public names, status
strings, decorative operations, or generated-output special cases.

Every accepted value feature must eventually have verifier, conversion,
lower-half, hardware where required, mixed-suite, and claim-contract coverage
appropriate to its risk.

## 21. Phase-by-phase implementation hooks

Future implementation hooks:

- Phase 5 elementwise: fragment constants, splats, ALU, comparisons, and
  select;
- Phase 9 multi-axis launch identity: `vc4value.grid_rank` 1/2/3,
  `program_id` axes 0/1/2, `num_programs` axes 0/1/2, flattened rank-1 memory
  only, and canonical linearized tail masks only;
- Phase 10 masks/memory legality: mask classifier, transfer read/write
  legality, TMU safe-offset loads, and VDW preserve stores;
- later gather/strided: gather-load and affine strided memory planning;
- Phase 12 reductions: i32 add reduction, f32 finite-tree add reduction, tail
  inactive-zero reduction, and scalar reduction-output stores;
- Phase 13 GEMV / row-wise dot: elementwise multiply, finite-tree reduction,
  scalar stores, and partial K-block staging;
- Phase 14 f16 storage/numeric policy: f16 storage conversion plus f32 compute,
  explicit finite f16 store policy, and staged fp-to-int/quantized forms;
- Phase 15 base-2 SFU-backed natural math and softmax: explicit approximate
  policy, natural exp/log scaling, sqrt through rsqrt, finite max reduction,
  reciprocal/division, and one-block softmax v0;
- Phase 17 shuffle/broadcast: rotate, lane broadcast composite, and direct
  arbitrary permutation rejects;
- Phase 18 VPM tile: VPM allocation, VDR, VDW, dynamic coordinates/selectors,
  and resource metadata;
- Phase 19 contract: row-fragment GEMM/GEMV planning through `vector.contract`.

Each phase must preserve the value-surface boundary, the locked VC4Kernel
surface, and the SSAVC4 lower-half boundary.

## 22. Phase 6 TTIR-to-value planning handoff

Phase 6 pins real TTIR source to Triton 3.7.0. The compiler support matrix for
Triton is based on emitted TTIR snapshots, not on Python-level metaprogramming.
The Triton Python examples under `examples/triton/phase6/kernels/` are demo
source provenance; the generated `.ttir.mlir` files are the source of truth for
future importer support.

The Phase 7 elementwise candidate TTIR forms map conceptually into the existing
value-layer planning boundary:

- `tt.func` / `tt.return` -> `func.func` / return boundary plus value kernel
  ABI metadata;
- axis-0 `tt.get_program_id` -> `vc4value.program_id`;
- `tt.make_range` / arange -> lane range, `vector.step`, splat, and offset
  arithmetic;
- `tt.splat` -> `vector.splat`;
- tensor `arith` ops -> value `arith` and vector elementwise operations;
- masked `tt.load` with zero `other` -> contiguous/tail
  `vector.transfer_read` planning;
- masked `tt.store` -> contiguous/tail `vector.transfer_write` planning.

Future staged TTIR forms remain outside Phase 7 V1 lowering:

- `tl.sum` / reductions -> later `vector.reduction` planning;
- `tl.exp` and other math -> later explicit math/SFU policy;
- `tl.dot` -> later `vector.contract` planning;
- block pointers and tensor descriptors -> later memory descriptor planning;
- atomics, cache/eviction modifiers, and volatile -> initial rejects or future
  profile work until semantics are specified.

The path remains TTIR -> standard value layer -> VC4Kernel -> SSAVC4 ->
scheduled VC4. Phase 6 does not implement TTIR-to-value semantic lowering, and
`READY_FOR_TRITON=NO` remains true until later phases add and prove the importer
and hardware path.

## 23. Phase 7 TTIR elementwise hardware handoff

Phase 7 proves that the real Phase 6 elementwise TTIR corpus can enter the
existing Phase 5 value planner without bypassing it. Phase 7.5 locks the
accepted semantic importer as the optional C++ `vc4-triton-opt` path:

```text
real emitted TTIR
  -> vc4-triton-opt --convert-triton-to-vc4-value
  -> func/vc4value/vector/memref/arith value IR
  -> --convert-vc4-value-to-vc4kernel
  -> vc4kernel
  -> ssavc4
  -> scheduled vc4
  -> generated artifacts/runtime
  -> real VC4 hardware
```

The supported examples are:

- `vector_add_b16`;
- `saxpy_select_b16`;
- `i32_add_select_b16`.

The TTIR importer output contract remains value-only. Generated value IR may
contain `func`, `vc4value`, `vector`, `memref`, and `arith`, but not `tt`,
`ttg`, `gpu`, `vc4kernel`, `ssavc4`, or scheduled `vc4` operations. Verified
`vc4kernel` output must contain no producer dialect operations.

Phase 7 hardware covers contiguous rank-1 i32/f32 memory, zero-other masked
loads, canonical tail stores, vector<16> arithmetic, compare/select, overlaunch
tail masks, TMU load planning, and VDW inactive-preserve stores at
active_qpus=12 and lanes=16. The mixed TTIR acceptance suite provides the
cumulative Phase 7 hardware gate and claim audit.

Future value planning remains responsible for reductions, dot/contract,
gather/scatter, block pointers, subword/f16 storage, math/SFU policy, non-16
block splitting, axes 1/2, and control flow. These are staged planning gaps
unless a later proof explicitly classifies a specific form as impossible.

Phase 8 does not add TTIR control-flow import. TTIR control flow remains
staged_future_ttir_import, and `READY_FOR_TRITON=NO` remains true.
READY_FOR_TRITON remains NO.

## 24. Phase 10.4 value mask/memory classifier

Phase 10.4 implements the value-layer executable classifier for the locked
Phase 10 mask and memory legality contract. Transfer masks now pass through a
central structural classifier before either `vector.transfer_read` or
`vector.transfer_write` can lower.

Accepted transfer masks are:

- full: no transfer mask, or clean all-active canonical mask;
- empty: clean zero-active canonical mask;
- tail: `vector.create_mask` for `vector<16xi1>` with the active count clamped
  to `[0, 16]` and a lane-zero start.

Compute masks produced by vector comparisons remain valid for
`arith.select`/fragment select, but they are not accepted as memory transfer
predicates. Sparse or unknown transfer masks reject deterministically in the
value-to-VC4Kernel pass.

The central memory legality classifier accepts only rank-1 identity
`#vc4value.global` memrefs with `i32` or `f32` elements, `vector<16xi32>` or
`vector<16xf32>` transfer values, scalar base indices, rank-1 identity transfer
maps, zero `transfer_read` padding, inactive-zero TMU loads, and
inactive-preserve VDW stores. Rank-2, strided, sparse, gather/scatter,
block-pointer, nonzero-other, subword/f16, and non-identity map forms remain
staged.

VALUE_MASK_CLASSIFIER_IMPLEMENTED=YES
VALUE_MEMORY_LEGALITY_CLASSIFIER_IMPLEMENTED=YES
SPARSE_TRANSFER_WRITE_REJECTS=PASS
SPARSE_TRANSFER_READ_REJECTS=PASS
NONZERO_LOAD_OTHER_REJECTS=PASS
VALUE_MASK_MEMORY_STATIC=PASS
READY_FOR_PHASE10_5_VALUE_HARDWARE_ISOLATION=YES
READY_FOR_TRITON=NO

## 25. Phase 10.5 value mask/memory hardware isolation

Phase 10.5 proves the Phase 10.4 value mask classifier and memory legality
subset on real VC4 hardware with targeted isolation fixtures. The fixtures run
with `active_qpus=12`, strict CPU oracles, sentinel preservation checks,
expected JSON result checking, and the existing VC4Value hardware runner.

The hardware-proven subset covers full, empty, and clamped tail transfer masks,
compute compare/select masks that do not feed memory predicates, zero-other
masked loads with inactive-zero behavior, TMU safe-offset policy, and
inactive-preserve stores. Sparse transfer masks remain static rejects.

VALUE_MASK_MEMORY_HARDWARE_ISOLATION=PASS
VALUE_MASK_FULL_HARDWARE=PASS
VALUE_MASK_EMPTY_HARDWARE=PASS
VALUE_MASK_TAIL_HARDWARE=PASS
VALUE_COMPUTE_MASK_SELECT_HARDWARE=PASS
VALUE_INACTIVE_ZERO_LOAD_HARDWARE=PASS
VALUE_INACTIVE_PRESERVE_STORE_HARDWARE=PASS
READY_FOR_PHASE10_6_VALUE_MIXED_ACCEPTANCE=YES
READY_FOR_TRITON=NO

## 26. Phase 10.6 value mask/memory mixed acceptance

Phase 10.6 extends the cumulative VC4Value mixed hardware suite with
`mixed_value_mask_memory_axis_cf_vc4value`. The fixture keeps memory rank-1 and
flattened while combining Phase 5 i32/f32 elementwise behavior, Phase 8 scalar
control flow, Phase 9 multi-axis launch identity, and Phase 10 mask/memory
legality in one natural kernel.

The mixed fixture checks full, empty, and tail transfer masks, compute-mask
select that does not feed memory predicates, inactive-zero load behavior,
inactive-preserve store behavior, repeated launches including `n=0`, strict CPU
oracles, sentinels, and `active_qpus=12`. The full VC4Value mixed suite passes
with zero mismatches, zero sentinel mismatches, and zero launch failures.

VALUE_MASK_MEMORY_MIXED_ACCEPTANCE=PASS
VALUE_MIXED_REGRESSION=PASS
READY_FOR_PHASE10_7_TTIR_IMPORTER_MASK_MEMORY_STATIC=YES
READY_FOR_TRITON=NO

## 27. Phase 10 final mask/memory legality lock

Phase 10 final-lock keeps the value-to-VC4Kernel contract unchanged from the
Phase 10.4 classifier implementation and records that the full value/TTIR
vertical slice has passed static checks, value hardware isolation, value mixed
acceptance, TTIR importer static lowering, TTIR hardware isolation, and TTIR
mixed acceptance.

The central value mask classifier accepts full, empty, and lane-zero clamped
tail masks for memory transfers. Compute masks remain compute-only. Sparse or
unknown transfer masks remain staged/rejected. For the Phase 10 executable
memory-transfer subset, the central memory legality classifier accepts rank-1
identity global i32/f32 `vector<16>` transfers with zero load padding and
inactive-zero/inactive-preserve policies. This is not the global value-layer
type or shape limit.

PHASE10_RESULT=LOCKED
FEATURE=VALUE_AND_TTIR_MASK_CLASSIFIER_MEMORY_LEGALITY
VALUE_MASK_MEMORY_CONTRACT=LOCKED
VALUE_MASK_CLASSIFIER_IMPLEMENTED=YES
VALUE_MEMORY_LEGALITY_CLASSIFIER_IMPLEMENTED=YES
VALUE_MASK_MEMORY_STATIC=PASS
VALUE_MASK_MEMORY_HARDWARE_ISOLATION=PASS
VALUE_MASK_MEMORY_MIXED_ACCEPTANCE=PASS
REAL_TRITON_MASK_MEMORY_SOURCES=YES
REAL_TTIR_MASK_MEMORY_SNAPSHOTS=YES
TTIR_MASK_MEMORY_IMPORTER_STATIC=PASS
TTIR_MASK_MEMORY_HARDWARE_ISOLATION=PASS
TTIR_MASK_MEMORY_MIXED_ACCEPTANCE=PASS
SPARSE_TRANSFER_MASKS_STAGED=YES
NONZERO_LOAD_OTHER_STAGED=YES
RANK2_STRIDED_GATHER_BLOCK_POINTER_STAGED=YES
VALUE_MIXED_REGRESSION=PASS
TTIR_MIXED_REGRESSION=PASS
LAYERED_REGRESSION_POLICY_APPLIED=YES
NO_TEMPORARY_MASK_MEMORY_WORKAROUNDS=YES
READY_FOR_PHASE11_STRIDED_RANKED_MEMORY_SKELETONS=YES
READY_FOR_TRITON=NO

## 28. Phase 11.3 value strided/ranked memory contract

PHASE11_VALUE_STRIDED_RANKED_MEMORY_CONTRACT=LOCKED
RANK2_ROW_SLICE_IDENTITY_SURFACE=ACCEPTED
RANK2_ROW_SLICE_STRIDED_OUTER_DYNAMIC_SURFACE=ACCEPTED
MEMREF_DIM_METADATA_TO_SCALAR_ARG_CONTRACT=LOCKED
HIDDEN_MEMREF_DESCRIPTOR_ALLOWED=NO
GATHER_LANE_STRIDE_STAGED=YES
READY_FOR_PHASE11_4_VALUE_RANKED_STRIDED_STATIC=YES
READY_FOR_TRITON=NO

Phase 11 extends the planned value memory contract beyond the locked Phase 10
rank-1 identity executable subset. This section is a planning contract only;
main executable lowering is intentionally not implemented in Phase 11.3.

Accepted Phase 11 value forms:

- `RANK1_FLATTENED_SCALAR_STRIDED_ADDRESS`: rank-1 global i32/f32 memref,
  scalar address expression `row * stride + col_block * 16`, rank-1 identity
  transfer, contiguous vector lanes, and Phase 10 full/empty/tail masks.
- `RANK2_ROW_SLICE_IDENTITY`: rank-2 identity global i32/f32 memref with
  `shape_args` for dynamic dimensions, scalar `[row, col]` transfer indices,
  and `vector<16xT>` mapped to the innermost dimension.
- `RANK2_ROW_SLICE_STRIDED_OUTER_DYNAMIC`: rank-2 global i32/f32 memref with
  `strided<[?, 1], offset: 0>`, `shape_args` for dynamic dimensions, one
  `stride_args` entry naming the dynamic outer row stride, static inner stride
  1, and static offset 0.
- `MEMREF_DIM_METADATA_LOWERING`: `memref.dim` on public global memref dynamic
  dimensions resolves to explicit scalar extent args, never hidden memref
  descriptors.

Planned Phase 11.4 static coverage:

- rank-1 flattened stride address lowers;
- rank-2 identity row-slice transfer lowers;
- rank-2 strided outer row-slice transfer lowers;
- `memref.dim` maps to shape args;
- nonunit inner stride rejects;
- lane-varying stride/gather rejects;
- column-slice rejects;
- hidden descriptor extraction rejects.

Staged forms remain non-unit inner stride, lane-varying stride/gather, scatter,
column/vertical slice, rank greater than 2, rank-2 vector tiles, block pointers,
boundary-check and padding-option semantics, reductions, dot, f16/subword
storage lowering, numeric casts, SFU/math expansion, hidden descriptors, and
sparse/unknown transfer masks outside the Phase 10 accepted set.

## 29. Phase 11.4 value ranked/strided static lowering lock

VALUE_RANK2_ROW_SLICE_TO_VC4KERNEL_STATIC=PASS
VALUE_MEMREF_DIM_METADATA_LOWERING=PASS
VALUE_STRIDED_RANKED_ADDRESS_PLANNER=YES
GATHER_LANE_STRIDE_STAGED=YES
HIDDEN_MEMREF_DESCRIPTOR_REJECTED=YES
VALUE_STRIDED_RANKED_MEMORY_STATIC=PASS
READY_FOR_PHASE11_5_VALUE_HARDWARE_ISOLATION=YES
READY_FOR_TRITON=NO

Phase 11.4 implements the static value-to-VC4Kernel executable subset for the
Phase 11 memory skeletons. `vector.transfer_read` and `vector.transfer_write`
now flow through a central `MemoryAddressPlan` helper that reuses the Phase 10
mask classifier and then computes one scalar element base index before the
existing byte-offset, TMU inactive-zero load, and VDW inactive-preserve store
paths.

Accepted static-lowering forms are:

- rank-1 flattened scalar-computed addresses, including `row * stride + col`
  expressions already present in value IR, with contiguous rank-1 transfer
  lanes;
- rank-2 identity row slices over `memref<?x?xT, #vc4value.global>` for
  `T=i32/f32`, where the row stride is the dim-1 extent named by
  `vc4value.shape_args`;
- rank-2 `strided<[?, 1], offset: 0>` row slices, where the outer row stride is
  the scalar named by `vc4value.stride_args`;
- metadata-only `memref.dim` on public global memref dynamic dimensions,
  resolved to explicit scalar extent arguments without descriptor loads.

Still-staged or rejected forms include lane-varying stride/gather, scatter,
column/vertical slices, non-unit inner stride, nonzero layout offsets, hidden
memref descriptors, rank-2 vector/tile transfers, block pointers, reductions,
dot/GEMV/GEMM, f16/subword storage lowering, numeric casts, and sparse/unknown
transfer masks beyond the Phase 10 full/empty/tail mask set.

## 30. Phase 11.5 value ranked/strided hardware isolation lock

VALUE_STRIDED_RANKED_MEMORY_HARDWARE_ISOLATION=PASS
VALUE_RANK1_FLATTENED_STRIDE_HARDWARE=PASS
VALUE_RANK2_IDENTITY_ROW_SLICE_HARDWARE=PASS
VALUE_RANK2_STRIDED_ROW_SLICE_HARDWARE=PASS
VALUE_MEMREF_DIM_METADATA_HARDWARE=PASS
READY_FOR_PHASE11_6_VALUE_MIXED_ACCEPTANCE=YES
READY_FOR_TRITON=NO

Phase 11.5 proves the Phase 11 value memory skeletons on hardware through
targeted isolation fixtures. The hardware runs cover rank-1 flattened
`row * lda + col` address expressions, rank-2 identity row-slice indexing,
rank-2 `strided<[?, 1], offset: 0>` row-slice indexing with explicit
`stride_args`, `memref.dim` lowering to scalar shape args, empty ranked
launches, repeated invocations, Phase 10 tail masks, and row-padding sentinels.

The claim audit keeps gather/lane-varying stride, non-unit inner stride,
column-slice maps, and hidden memref descriptor ABI paths staged or rejected.
No Triton readiness is implied.

## 31. Phase 11.6 value mixed acceptance lock

VALUE_STRIDED_RANKED_MEMORY_MIXED_ACCEPTANCE=PASS
VALUE_MIXED_REGRESSION=PASS
READY_FOR_PHASE11_7_TTIR_IMPORTER_STRIDED_MEMORY_STATIC=YES
READY_FOR_TRITON=NO

Phase 11.6 extends the cumulative VC4Value mixed hardware suite with
`mixed_value_strided_ranked_memory_axis_mask_cf_vc4value`. The fixture combines
rank-1 flattened row stride, rank-2 strided row-slice memory, shape and stride
metadata, `memref.dim`, Phase 9 multi-axis launch, Phase 8 control flow, Phase
10 tail masks and compute-mask selects, inactive-zero reads, inactive-preserve
writes, repeated empty/non-empty launches, and row-padding sentinels.

The full VC4Value mixed regression suite passes on hardware with active_qpus=12
where applicable. Gather/lane-varying stride and hidden memref descriptor
support remain explicitly unclaimed and staged.

## 32. Phase 11 final strided/ranked memory lock

PHASE11_RESULT=LOCKED
FEATURE=VALUE_AND_TTIR_STRIDED_RANKED_MEMORY_SKELETONS
VALUE_STRIDED_RANKED_MEMORY_CONTRACT=LOCKED
VALUE_STRIDED_RANKED_ADDRESS_PLANNER=YES
VALUE_RANK2_ROW_SLICE_TO_VC4KERNEL_STATIC=PASS
VALUE_MEMREF_DIM_METADATA_LOWERING=PASS
VALUE_STRIDED_RANKED_MEMORY_HARDWARE_ISOLATION=PASS
VALUE_STRIDED_RANKED_MEMORY_MIXED_ACCEPTANCE=PASS
REAL_TRITON_STRIDED_MEMORY_SOURCES=YES
REAL_TTIR_STRIDED_MEMORY_SNAPSHOTS=YES
TTIR_STRIDED_MEMORY_IMPORTER_STATIC=PASS
TTIR_STRIDED_MEMORY_HARDWARE_ISOLATION=PASS
TTIR_STRIDED_MEMORY_MIXED_ACCEPTANCE=PASS
GATHER_LANE_STRIDE_STAGED=YES
COLUMN_SLICE_STAGED=YES
HIDDEN_MEMREF_DESCRIPTOR_REJECTED=YES
VALUE_MIXED_REGRESSION=PASS
TTIR_MIXED_REGRESSION=PASS
LAYERED_REGRESSION_POLICY_APPLIED=YES
NO_TEMPORARY_STRIDED_MEMORY_WORKAROUNDS=YES
READY_FOR_PHASE12_REDUCTIONS=YES
READY_FOR_TRITON=NO

Phase 11 final-locks the strided/ranked memory skeleton feature through the
standard value path and the real TTIR bridge. Value lowering now uses the
central address planner for rank-1 flattened scalar strided addresses, rank-2
identity row slices, rank-2 strided-outer row slices, and metadata-only
`memref.dim` shape lowering. The TTIR bridge lowers controlled real
row-strided pointer snapshots into flattened rank-1 value memrefs and then
through the same value planner.

Still-staged forms remain gather/lane-varying stride, column/vertical slices,
non-unit inner stride, hidden memref descriptors, rank-2 vector/tile transfer
forms, block pointers, reductions, dot/GEMV/GEMM, numeric casts, subword/f16
storage expansion, SFU/math expansion, and sparse/unknown memory masks outside
the Phase 10 accepted set.

## 33. Phase 12.5 value reduction hardware isolation lock

VALUE_REDUCTION_HARDWARE_ISOLATION=PASS
VALUE_REDUCTION_I32_ADD_HARDWARE=PASS
VALUE_REDUCTION_F32_FINITE_ADD_HARDWARE=PASS
VALUE_SCALAR_REDUCTION_STORE_HARDWARE=PASS
VALUE_ROW_STRIDED_REDUCTION_HARDWARE=PASS
READY_FOR_PHASE12_6_VALUE_MIXED_ACCEPTANCE=YES
READY_FOR_TRITON=NO

Phase 12.5 proves value-layer reductions on hardware with targeted isolation
fixtures. The accepted executable subset remains `vector.reduction <add>` over
`vector<16xi32>`, finite-tree `vector<16xf32>` add reductions, inactive-zero
tail inputs from Phase 10 loads, scalar rank-1 reduction-output stores, and
Phase 11 row-strided row-slice inputs. Non-add reductions, rank>1 reductions,
dot/GEMV/GEMM, scans, atomics, and default/exact f32 reductions remain staged.

## 34. Phase 12.6 value mixed acceptance lock

VALUE_REDUCTION_MIXED_ACCEPTANCE=PASS
VALUE_MIXED_REGRESSION=PASS
READY_FOR_PHASE12_7_TTIR_IMPORTER_REDUCTION_STATIC=YES
READY_FOR_TRITON=NO

Phase 12.6 adds `mixed_value_reduction_axes_mask_cf_strided_vc4value` to the
VC4Value mixed hardware acceptance suite. The fixture combines Phase 9
multi-axis launch identity, Phase 8 scalar control flow, Phase 10 tail masks
and compute-mask selects, Phase 11 row-strided memory, Phase 12 i32 and
finite-tree f32 add reductions, and scalar reduction-output stores. The suite
passes with strict CPU oracles, sentinels, expected JSON checks, active QPU
requirements, and claim audits. Dot, GEMV, GEMM, non-add reductions, and
rank>1 reductions remain staged.

## 35. Phase 12 final value and TTIR reduction lock

PHASE12_RESULT=LOCKED
FEATURE=VALUE_AND_TTIR_REDUCTIONS
VALUE_REDUCTION_CONTRACT=LOCKED
VALUE_REDUCTION_TO_VC4KERNEL_STATIC=PASS
VALUE_VECTOR_REDUCTION_ADD_I32_STATIC=PASS
VALUE_VECTOR_REDUCTION_ADD_F32_FINITE_STATIC=PASS
VALUE_SCALAR_STORE_FOR_REDUCTION_STATIC=PASS
VALUE_REDUCTION_HARDWARE_ISOLATION=PASS
VALUE_REDUCTION_MIXED_ACCEPTANCE=PASS
REAL_TRITON_REDUCTION_SOURCES=YES
REAL_TTIR_REDUCTION_SNAPSHOTS=YES
TTIR_REDUCTION_IMPORTER_STATIC=PASS
TTIR_REDUCTION_HARDWARE_ISOLATION=PASS
TTIR_REDUCTION_MIXED_ACCEPTANCE=PASS
F32_REDUCTION_FINITE_TREE_POLICY=YES
EXACT_F32_REDUCTION_NOT_CLAIMED=YES
NON_ADD_REDUCTIONS_STAGED=YES
RANK_GT_1_REDUCTIONS_STAGED=YES
DOT_GEMV_STAGED_FOR_PHASE13=YES
VALUE_MIXED_REGRESSION=PASS
TTIR_MIXED_REGRESSION=PASS
LAYERED_REGRESSION_POLICY_APPLIED=YES
NO_TEMPORARY_REDUCTION_WORKAROUNDS=YES
READY_FOR_PHASE13_GEMV_ROWWISE_DOT=YES
READY_FOR_TRITON=NO

The Phase 12 final lock preserves the value-to-VC4Kernel contract for add-only
reductions and scalar reduction-output stores. Value lowering supports
`vector.reduction <add>` over `vector<16xi32>` and explicit finite-tree
`vector<16xf32>` reductions, plus rank-1 scalar `memref.store` outputs. TTIR
support enters through the standard value surface and then this same value
planner.

No exact/default f32 reduction support is claimed. Non-add reductions, rank>1
reductions, `vector.multi_reduction`, dot, GEMV, GEMM, scans, atomics, and
generalized math remain staged for later phases.

## 36. Phase 13 final GEMV row-wise dot lock

PHASE13_RESULT=LOCKED
FEATURE=VALUE_AND_TTIR_GEMV_ROWWISE_DOT
VALUE_GEMV_ROWWISE_DOT_CONTRACT=LOCKED
VALUE_GEMV_ROWWISE_DOT_STATIC=PASS
VALUE_GEMV_F32_ROW_DOT_STATIC=PASS
VALUE_GEMV_I32_ROW_DOT_STATUS=STAGED_BY_I32_POLICY
VALUE_GEMV_ROWWISE_DOT_HARDWARE_ISOLATION=PASS
VALUE_GEMV_ROWWISE_DOT_MIXED_ACCEPTANCE=PASS
REAL_TRITON_GEMV_ROWWISE_DOT_SOURCES=YES
REAL_TTIR_GEMV_ROWWISE_DOT_SNAPSHOTS=YES
TTIR_GEMV_ROWWISE_DOT_IMPORTER_STATIC=PASS
TTIR_GEMV_ROWWISE_DOT_HARDWARE_ISOLATION=PASS
TTIR_GEMV_ROWWISE_DOT_MIXED_ACCEPTANCE=PASS
F32_DOT_FINITE_TREE_POLICY=YES
EXACT_F32_DOT_NOT_CLAIMED=YES
TL_DOT_TT_DOT_STAGED=YES
VECTOR_CONTRACT_STAGED=YES
MULTIBLOCK_K_ACCUMULATION_STAGED=YES
FULL_GEMM_STAGED=YES
VALUE_MIXED_REGRESSION=PASS
TTIR_MIXED_REGRESSION=PASS
LAYERED_REGRESSION_POLICY_APPLIED=YES
NO_TEMPORARY_GEMV_WORKAROUNDS=YES
READY_FOR_PHASE14_ML_STORAGE_NUMERIC_CONVERSION_POLICY=YES
READY_FOR_TRITON=NO

The value planner accepts the Phase 13 row-wise dot only as a composite path:
elementwise f32 multiply over `vector<16xf32>` feeds the finite-tree add
reduction path from Phase 12, and the scalar result stores through the Phase 12
scalar store path. The partial K-block form stores one independent scalar
partial per `(row, kblock)`. I32 row-wise dot remains staged by the current i32
multiply policy.

No new VC4Kernel dot op is introduced. `tl.dot`, `tt.dot`, `vector.contract`,
full GEMM, atomics, cross-program accumulation, exact/default f32 dot, and
multi-block K accumulation into final `y[row]` remain staged.

Phase 14 final lock status:

```text
PHASE14_RESULT=LOCKED
FEATURE=VALUE_AND_TTIR_ML_STORAGE_NUMERIC_CONVERSION_POLICY
VALUE_ML_STORAGE_NUMERIC_CONTRACT=LOCKED
VALUE_F16_STORAGE_F32_COMPUTE_STATIC=PASS
VALUE_F16_STORAGE_F32_COMPUTE_HARDWARE_ISOLATION=PASS
VALUE_F16_STORAGE_F32_COMPUTE_MIXED_ACCEPTANCE=PASS
TTIR_F16_STORAGE_IMPORTER_STATIC=PASS
TTIR_F16_STORAGE_HARDWARE_ISOLATION=PASS
TTIR_F16_STORAGE_MIXED_ACCEPTANCE=PASS
F16_STORAGE_FINITE_POLICY=YES
I32_TO_F32_CAST_STATUS=STAGED_BY_LOWER_HALF_GAP
NATIVE_F16_ARITHMETIC_STAGED=YES
BF16_FP8_STAGED=YES
INT8_INT16_QUANTIZED_STORAGE_STAGED=YES
FP_TO_INT_CASTS_STAGED=YES
EXACT_UNPOLICY_NUMERIC_CASTS_STAGED=YES
SOFTMAX_SFU_STAGED_FOR_PHASE15=YES
VALUE_MIXED_REGRESSION=PASS
TTIR_MIXED_REGRESSION=PASS
LAYERED_REGRESSION_POLICY_APPLIED=YES
NO_TEMPORARY_STORAGE_NUMERIC_WORKAROUNDS=YES
READY_FOR_PHASE15_APPROX_MATH_SFU_SOFTMAX=YES
READY_FOR_TRITON=NO
```

The Phase 14 planner path accepts only f16 storage conversion plus f32 compute.
Native f16 arithmetic, bf16/fp8, quantized int storage, fp-to-int casts, exact
unpolicy numeric casts, softmax/SFU, `tt.dot`, `vector.contract`, and GEMM
remain staged.

## 37. Phase 16.5 attention-apply value hardware isolation

VALUE_ATTENTION_APPLY_V0_HARDWARE_ISOLATION=PASS
VALUE_ATTENTION_APPLY_V0_F32_HARDWARE=PASS
VALUE_ATTENTION_APPLY_V0_SCALED_HARDWARE=PASS
VALUE_ATTENTION_APPLY_V0_F16_STORAGE_HARDWARE=PASS
VALUE_ATTENTION_APPLY_V0_TOLERANCE_POLICY=LOCKED
VALUE_ATTENTION_APPLY_V0_COMPOSITE=YES
PRECOMPUTED_SCORES_ONLY=YES
TRANSPOSED_V_LAYOUT_REQUIRED=YES
SCALAR_GLOBAL_LOAD_STAGED=YES
NONTRANSPOSED_V_GATHER_STAGED=YES
READY_FOR_PHASE16_6_VALUE_MIXED_ACCEPTANCE=YES
READY_FOR_TRITON=NO

Phase 16.5 hardware isolation proves the Phase 16 value attention-apply v0
composite on real VC4 hardware. The proof covers precomputed scores,
transposed-V row-contiguous loads, natural-exp stable softmax, weighted add
reduction, scalar output stores, scalar f32 scale argument, repeat launches,
and f16 score/Vt storage with f32 compute. Scalar global loads, non-transposed
V gather, K=0, QK score generation, dot/contract, multiblock softmax, full
attention, and FlashAttention remain staged.

The `vc4value.attention_apply_v0` attribute is verifier metadata only and is
not used by value-to-VC4Kernel lowering. Phase 16 lowering support is proven by
the actual value IR composite: score transfer read, optional scale
broadcast/multiply, finite-low inactive select, max reduction, natural-exp
softmax path, denominator add reduction, reciprocal/division, transposed-V
transfer read, weighted multiply, add reduction, and scalar store.

## 38. Phase 16.6 attention-apply value mixed acceptance

VALUE_ATTENTION_APPLY_V0_MIXED_ACCEPTANCE=PASS
VALUE_MIXED_REGRESSION=PASS
VALUE_ATTENTION_APPLY_V0_COMPOSITE=YES
PRECOMPUTED_SCORES_ONLY=YES
TRANSPOSED_V_LAYOUT_REQUIRED=YES
SCALAR_GLOBAL_LOAD_STAGED=YES
NONTRANSPOSED_V_GATHER_STAGED=YES
READY_FOR_PHASE16_7_TTIR_IMPORTER_ATTENTION_APPLY_STATIC=YES
READY_FOR_TRITON=NO

Phase 16.6 adds the cumulative mixed fixture
`mixed_value_attention_apply_v0_axes_mask_cf_f16_storage_vc4value`. The fixture
combines Phase 16 attention-apply v0 with multi-axis launch, scalar control
flow, tail masks, row-strided memory, f16 storage loads, i32/f32 elementwise
interaction, natural-exp softmax, f32 weighted reduction, scalar stores, and
active_qpus=12 repeated launches. The full VC4Value mixed hardware suite
passes, and the mixed claim audit covers the new attention-apply claims without
claiming QK score generation, dot/contract, scalar global loads,
non-transposed V gather, full attention, or FlashAttention.

## 38.1 Phase 17.6 online-softmax value mixed acceptance

VALUE_ONLINE_SOFTMAX_MIXED_ACCEPTANCE=PASS
VALUE_MIXED_REGRESSION=PASS
READY_FOR_PHASE17_7_TTIR_IMPORTER_ONLINE_SOFTMAX_STATIC=YES
READY_FOR_TRITON=NO

Phase 17.6 adds the cumulative mixed fixture
`mixed_value_online_attention_apply_axes_mask_cf_f16_storage_vc4value`. The
fixture combines online `m/l/acc` attention state with multi-axis launch,
masks, control flow, finite reductions, row-strided f16 storage, i32/f32 value
interaction, natural-exp softmax, transposed V layout, weighted reductions,
scalar result stores, K values greater than 16, repeated launches, and
`active_qpus=12`. The full VC4Value mixed hardware suite passes, and the
claim audit checks Phase 17 online softmax claims without claiming scalar
global loads, non-transposed V gather, QK score generation, dot/contract,
full attention, or FlashAttention.

## 39. Phase 16 final attention-apply v0 lock

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

Phase 16 final-locks attention-apply v0 as a standard value composite, not a
new `vc4value` operation. The value path remains precomputed-score only and
requires transposed-V row slices. Scalar global loads, non-transposed V gather,
K=0 without an explicit guard, multiblock softmax, QK score generation,
dot/contract, full attention, and FlashAttention remain staged.
