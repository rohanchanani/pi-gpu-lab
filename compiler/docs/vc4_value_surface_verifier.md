# VC4 Value Surface Verifier Contract

PHASE10_VALUE_MASK_CLASSIFIER_CONTRACT=LOCKED
PHASE10_VALUE_MEMORY_LEGALITY_CONTRACT=LOCKED
PHASE12_VALUE_REDUCTION_CONTRACT=LOCKED
PHASE13_VALUE_GEMV_ROWWISE_DOT_CONTRACT=LOCKED
PHASE14_VALUE_ML_STORAGE_NUMERIC_CONTRACT=LOCKED
PHASE15_VALUE_APPROX_MATH_SFU_SOFTMAX_CONTRACT=LOCKED
PHASE16_VALUE_ATTENTION_APPLY_V0_CONTRACT=LOCKED
VALUE_ATTENTION_APPLY_V0_SURFACE=ACCEPTED
VALUE_VECTOR_REDUCTION_ADD_I32_SURFACE=ACCEPTED
VALUE_VECTOR_REDUCTION_ADD_F32_FINITE_SURFACE=ACCEPTED
VALUE_FINITE_F32_MAX_REDUCTION_SURFACE=ACCEPTED
VALUE_SCALAR_MEMREF_STORE_FOR_REDUCTION_SURFACE=ACCEPTED
VALUE_GEMV_ROWWISE_DOT_F32_SURFACE=ACCEPTED
VALUE_GEMV_ROWWISE_DOT_I32_SURFACE=ACCEPTED
VALUE_F16_STORAGE_TO_F32_COMPUTE_SURFACE=ACCEPTED
VALUE_F32_COMPUTE_TO_F16_STORAGE_SURFACE=ACCEPTED
VALUE_APPROX_SFU_EXP_SURFACE=ACCEPTED
VALUE_APPROX_SFU_RECIP_DIV_SURFACE=ACCEPTED
VALUE_SOFTMAX_V0_COMPOSITE_SURFACE=ACCEPTED
VALUE_ATTENTION_APPLY_V0_COMPOSITE_SURFACE=ACCEPTED
F32_REDUCTION_FINITE_TREE_POLICY=YES
F32_DOT_FINITE_TREE_POLICY=YES
F16_STORAGE_FINITE_POLICY=YES
APPROX_MATH_POLICY=EXPLICIT
EXACT_DEFAULT_MATH_REJECTED=YES
ZERO_ACTIVE_SOFTMAX_STATUS=STAGED_OR_EXPLICIT_NOOP_GUARD_REQUIRED
PRECOMPUTED_SCORES_ONLY=YES
TRANSPOSED_V_LAYOUT_REQUIRED=YES
SCALAR_GLOBAL_LOAD_STAGED=YES
NONTRANSPOSED_V_GATHER_STAGED=YES
K_ZERO_ATTENTION_APPLY_STAGED_OR_GUARD_REQUIRED=YES
I32_TO_F32_CAST_STATUS=STAGED_BY_LOWER_HALF_GAP
NATIVE_F16_ARITHMETIC_STAGED=YES
BF16_FP8_STAGED=YES
INT8_INT16_QUANTIZED_STORAGE_STAGED=YES
NON_ADD_REDUCTIONS_STAGED=YES
TL_DOT_TT_DOT_STAGED=YES
VECTOR_CONTRACT_STAGED=YES
MULTIBLOCK_K_ACCUMULATION_STAGED=YES
SPARSE_MEMORY_MASKS_STAGED=YES
NONZERO_LOAD_OTHER_STAGED=YES
RANK2_STRIDED_MEMORY_STAGED=YES
READY_FOR_PHASE10_4_VALUE_MASK_MEMORY_CLASSIFIER_STATIC=YES
READY_FOR_PHASE12_4_VALUE_REDUCTION_STATIC=YES
READY_FOR_PHASE13_4_VALUE_GEMV_STATIC=YES
READY_FOR_PHASE14_4_VALUE_STORAGE_NUMERIC_STATIC=YES
READY_FOR_PHASE15_4_VALUE_SFU_SOFTMAX_STATIC=YES
READY_FOR_PHASE16_4_VALUE_ATTENTION_APPLY_STATIC=YES
READY_FOR_TRITON=NO

## 1. Purpose

`--vc4-verify-value-surface` verifies value-surface admissibility, not current
lowerability. It establishes the source-level compiler contract for the
standard VC4 value surface before Phase 4 ABI work and Phase 5+
value-to-VC4Kernel lowering begin.

The verifier is a static boundary checker for:

```text
func + tiny vc4value + vector + memref + arith + math + scf/cf
```

It does not emit VC4Kernel, does not prove hardware correctness, and does not
run hardware.

## 2. Layer boundary

The intended stack remains:

```text
real Triton / TTIR
  -> standard MLIR value surface
       func + tiny vc4value + vector + memref + arith + math + scf/cf
  -> vc4kernel
  -> ssavc4
  -> scheduled vc4
  -> artifacts/runtime/hardware
```

The value surface must not expose TMU, VDR, VDW, or VPM operations directly.
VC4 hardware path choices belong to later value-to-VC4Kernel planning.

No vector dialect ops are legal inside verified VC4Kernel. VC4Kernel may still
use `vector<16xT>` carrier types for target fragment values.

Phase 3.5 refines this boundary in
`compiler/docs/vc4_value_surface_abstraction_policy.md`: fixed rank-1 and
rank-2 value vectors are surface-admissible when their element types are
allowed, while `vector<16xT>` is the Phase 5 V1 lowerable fragment-normal form
and not the global value-layer type limit.

## 3. What the verifier proves

The verifier proves that an input module stays inside the Phase 3
producer-facing value-surface boundary:

- only allowed value dialect families are used;
- target/lower-half dialects are absent;
- `vc4value` remains the two-op launch identity dialect;
- kernel metadata has the minimum shape required for later verification;
- value memory syntax preserves the transfer/planning boundary;
- unsupported type shapes and sparse-store-shaped operations are rejected.

## 4. What the verifier does not prove

The verifier does not prove:

- value-to-VC4Kernel lowerability;
- full memref ABI correctness;
- vector transfer lowering;
- math lowering policy correctness;
- VC4Kernel verifier correctness;
- scheduled VC4 correctness;
- runtime or hardware correctness;
- TTIR/Triton import support.

Lowerability starts in Phase 5 after Phase 4 ABI lock. TTIR import remains
disallowed until the handwritten value path is proven.

Phase 4 extends the same verification boundary with public value-kernel ABI
rules documented in `compiler/docs/vc4_value_kernel_abi.md`. Phase 4d verifies
the public kernel wrapper, `vc4value.grid_rank`, public argument names, memref
directions, scalar roles, public argument categories, and unknown
`vc4value.*` argument ABI attributes. Phase 4e verifies the exact
`#vc4value.global` public memref ABI: rank 1/rank 2 only, element types
`i8`, `i16`, `i32`, `f16`, and `f32`, explicit `vc4value.shape_args` for
dynamic dimensions, explicit `vc4value.stride_args` for dynamic strides in
strided layouts, `memref.dim` as metadata only, and launch identity axes within
`vc4value.grid_rank`. These checks still do not lower value IR to VC4Kernel.

## 5. Allowed dialect family

The allowed producer-facing value dialect family is exactly:

```text
builtin, func, vc4value, vector, memref, arith, math, scf, cf
```

`builtin` is allowed for modules, attributes, types, and regions. The other
dialects provide standard value syntax and the tiny launch identity gap-filler.
Admitting these dialects is not a claim that all operations in them lower today.

## 6. Forbidden dialect families

The verifier rejects producer dialects that must canonicalize into the value
surface first:

```text
tt, ttg, gpu, linalg, nvgpu, nvvm, rocdl, spirv, iree, stablehlo, mhlo
```

Tensor dialect IR is also outside the initial value surface. Future producer
layers may canonicalize tensor or linalg-like programs into the value surface,
but tensor/linalg operations are not Phase 3.5 value IR.

It also rejects target/lower-half dialects in value input:

```text
vc4kernel, ssavc4, vc4
```

This preserves the one-way path through the standard value surface before
VC4Kernel planning.

## 7. vc4value scope rules

The only valid `vc4value` operations are:

```mlir
vc4value.program_id   {axis = 0|1|2} : index
vc4value.num_programs {axis = 0|1|2} : index
```

`vc4value` must not grow memory, tile, fragment, TMU, VDR, VDW, VPM, barrier,
warp, thread, lane, or physical QPU identity operations. Lane identity remains
standard `vector.step` syntax.

`vc4value.program_id` and `vc4value.num_programs` are logical launch-grid
identity operations, not physical QPU identity operations.

## 8. Kernel metadata rules

`vc4value.program_id` and `vc4value.num_programs` must appear only inside a
`func.func` marked with `vc4value.kernel`.

Every `vc4value.kernel` function must carry `vc4value.grid_rank` as an integer
attribute with value 1, 2, or 3.

For every `vc4value.program_id` or `vc4value.num_programs`, the `axis` value
must be less than the containing kernel's `vc4value.grid_rank`. Phase 2 already
verifies the local axis domain 0..2; Phase 4e adds the function-level grid-rank
relationship.

Other metadata conventions such as `vc4value.target_profile` and
`vc4value.math_policy` remain ordinary named attrs until later phases assign
stronger policy meaning.

## 9. Type guardrails

Phase 3.5 accepts fixed, non-scalable rank-1 and rank-2 vector types and ranked
memref types within the initial value-surface shape contract. Phase 4 narrows
public kernel argument memrefs to the ABI contract without narrowing non-ABI
body vector values.

Allowed vector element types are exactly `i1`, `index`, `i8`, `i16`, `i32`,
`f16`, and `f32`. `vector<16xT>` is the Phase 5 V1 lowerable fragment-normal
form. Non-16 rank-1 vectors are staged for split/tail lowering, and rank-2
vectors are staged for tile/contract planning.

The verifier rejects:

- scalable vectors;
- unsupported vector element types;
- unranked memrefs;
- memref rank greater than 2;
- public memref arguments outside `#vc4value.global`;
- public memref arguments without required dynamic `vc4value.shape_args`;
- malformed or unresolved public memref `vc4value.stride_args`;
- vector rank greater than 2;
- tensor and complex types.

These are Phase 3 staging guardrails, not permanent hardware-impossibility
proofs.

## 10. Memory side-effect policy

The value surface uses standard memory semantics while preserving the later
planner's choice of TMU, VDR, VDW, or VPM. `vector.transfer_read` and
`vector.transfer_write` are admissible standard memory syntax, but accepting
them does not mean lowering exists yet.

The verifier rejects direct memory side-effect operations that bypass the
vector/memref planning boundary:

```text
memref.load
memref.atomic_rmw
memref.generic_atomic_rmw
memref.atomic_yield
memref.copy
memref.dma_start
memref.dma_wait
```

Phase 12 admits the narrow scalar reduction-output store form:

```text
memref.store scalar i32/f32 to rank-1 #vc4value.global output memref
```

This is a source-level ABI admission for reduction results. It does not expose
VDW, does not add a hidden memref descriptor, and does not claim executable
lowering in the verifier phase. All other direct `memref.store` forms remain
rejected.

## 10.1 Phase 12 reduction contract

PHASE12_VALUE_REDUCTION_CONTRACT=LOCKED
VALUE_VECTOR_REDUCTION_ADD_I32_SURFACE=ACCEPTED
VALUE_VECTOR_REDUCTION_ADD_F32_FINITE_SURFACE=ACCEPTED
VALUE_SCALAR_MEMREF_STORE_FOR_REDUCTION_SURFACE=ACCEPTED
F32_REDUCTION_FINITE_TREE_POLICY=YES
NON_ADD_REDUCTIONS_STAGED=YES
DOT_GEMV_STAGED_FOR_PHASE13=YES
READY_FOR_PHASE12_4_VALUE_REDUCTION_STATIC=YES
READY_FOR_TRITON=NO

The accepted Phase 12 value reduction forms are:

- `I32_VECTOR_ADD_REDUCTION`: `vector.reduction <add>` over
  `vector<16xi32>` with scalar `i32` result. Integer reduction follows the
  existing exact modulo/int VC4Kernel reduction policy.
- `F32_VECTOR_ADD_REDUCTION_FINITE_TREE`: `vector.reduction <add>` over
  `vector<16xf32>` with scalar `f32` result. The reduction or containing
  kernel must carry explicit finite-input and finite-tree policy:
  `vc4value.fp_domain = "finite"` and
  `vc4value.reduction_policy = "finite_tree"`.
- `TAIL_REDUCTION_BY_INACTIVE_ZERO`: the reduction op remains unmasked.
  Inactive tail lanes must already contain zero values from Phase 10
  inactive-zero transfer reads using `other=0` or `other=0.0`.
- `SCALAR_REDUCTION_OUTPUT_STORE`: scalar `i32`/`f32` reduction results may be
  stored to rank-1 `#vc4value.global` output memrefs, including inside existing
  `scf`/`cf` control flow.

The Phase 12 verifier contract stages or rejects:

- max, min, product, and custom reductions;
- `vector.multi_reduction`;
- rank greater than 1 reductions;
- scan/prefix forms;
- atomics;
- reductions over f16/subword storage;
- exact/default f32 reductions without finite-tree policy;
- dot, GEMV, and GEMM.

The f32 policy is finite-tree target semantics only. Phase 12 makes no exact
IEEE left-to-right sum claim, and NaN/Inf behavior is not accepted.

## 10.2 Phase 13 GEMV / row-wise dot contract

PHASE13_VALUE_GEMV_ROWWISE_DOT_CONTRACT=LOCKED
VALUE_GEMV_ROWWISE_DOT_F32_SURFACE=ACCEPTED
VALUE_GEMV_ROWWISE_DOT_I32_SURFACE=ACCEPTED
F32_DOT_FINITE_TREE_POLICY=YES
TL_DOT_TT_DOT_STAGED=YES
VECTOR_CONTRACT_STAGED=YES
MULTIBLOCK_K_ACCUMULATION_STAGED=YES
READY_FOR_PHASE13_4_VALUE_GEMV_STATIC=YES
READY_FOR_TRITON=NO

Phase 13 admits GEMV-v0 / row-wise dot as a composition of already locked value
forms. It does not add a first-class value dot operation.

The accepted Phase 13 value forms are:

- `F32_VECTOR16_DOT_COMPOSITE`: elementwise `arith.mulf` over
  `vector<16xf32>`, with the product feeding `vector.reduction <add>` to an
  `f32` scalar. Inputs must be finite and the reduction policy must be
  explicit finite-tree. The result is a scalar f32 value; no exact IEEE sum or
  fused multiply-add claim is made.
- `I32_VECTOR16_DOT_COMPOSITE`: surface-admissible, but executable lowering is
  staged by the current i32 multiply policy. Phase 13.5 hardware isolation
  showed that `vc4value.i32_mul_policy = "mul24_safe"` does not prove exact
  signed i32 dot results for multiply plus reduction.
- `TAIL_DOT_BY_INACTIVE_ZERO`: tail dots do not require a masked reduction op.
  Inactive lanes are zero through Phase 10 transfer-read `other=0` /
  inactive-zero semantics, and the dot reduction remains unmasked.
- `ROW_STRIDED_GEMV_V0`: the A row is represented with Phase 11 row-strided
  memory, X is rank-1 contiguous or scalar-strided accepted memory, K is at
  most one `vector<16>` block for a full row-dot output, and the output is a
  scalar `memref.store` accepted by Phase 12.
- `PARTIAL_KBLOCK_DOT`: a program computes one partial dot for a `(row,
  kblock)` pair and stores that scalar partial result. There is no cross-kblock
  accumulation in Phase 13.

The staged Phase 13 forms are:

- `tl.dot` / `tt.dot`;
- `vector.contract`;
- a dot-shaped custom value op;
- multi-block K accumulation into final `y[row]`;
- atomics or cross-program accumulation;
- rank-2 tile values;
- fma policy, f16/subword inputs, and exact/default f32 dot.

This verifier contract proves only value-surface admission and staged
classification. Executable GEMV lowering and static proof are Phase 13.4 work.

## 11. Sparse store policy

Sparse stores are not an accepted VC4Kernel store path. Sparse VDW stores remain
rejected by the locked VC4Kernel surface.

For Phase 3, the verifier rejects sparse-store-shaped vector operations:

```text
vector.scatter
vector.compressstore
```

Future support requires a hardware-proven or explicitly emulated design and an
updated source contract.

## 11.1 Phase 10 mask classifier contract

Phase 10 locks the value-level mask categories that the later
value-to-VC4Kernel planner must classify before memory lowering. The
value-surface verifier remains an admissibility checker: it may accept staged
standard value IR forms so later passes can emit precise diagnostics.

The central Phase 10 mask classifier categories are:

- `FULL`: no transfer mask, or a `vector.create_mask` active count proven to be
  at least 16 after clamping to `[0, 16]`.
- `EMPTY`: a `vector.create_mask` active count proven to be at most 0 after
  clamping.
- `TAIL_0_TO_16`: `vector.create_mask %count : vector<16xi1>`, with `%count`
  clamped to `[0, 16]` and active lanes forming a prefix starting at lane 0.
- `COMPUTE_MASK`: `vector<16xi1>` masks from compare/logical value forms used
  only by compute operations such as `arith.select`. Phase 10 accepts these as
  compute masks only; they are not accepted as transfer memory predicates.
- `SPARSE_OR_UNKNOWN_MEMORY_MASK`: any `vector<16xi1>` transfer mask not
  classified as `FULL`, `EMPTY`, or `TAIL_0_TO_16`. Phase 10 stages these for
  loads unless a later phase implements exact support, and deterministically
  stages/rejects them for stores.
- `RECT`: staged until the rank-2/tile memory phase.

Triton and value IR may contain unrelated proof or overflow guards. The Phase
10 semantic memory-mask classification is based on the SSA value used by
`vector.transfer_read` or `vector.transfer_write`, not on fixture names,
printed IR substrings, or surrounding decorative operations.

## 11.2 Phase 10 memory legality contract

Phase 10 accepted value transfer legality is:

- memref rank 1;
- `#vc4value.global` memory space;
- identity layout;
- element type `i32` or `f32`;
- `vector<16xi32>` or `vector<16xf32>` transfer vector;
- rank-1 identity permutation map;
- scalar index base;
- `vector.transfer_read` padding/`other` exactly zero;
- transfer-read inactive lanes use safe offset and inactive-zero semantics;
- transfer-write inactive lanes preserve old output.

Phase 10 staged or rejected value transfer forms are:

- nonzero load `other` / padding;
- non-identity transfer map;
- rank greater than 1 memref;
- strided layout;
- tensor or block pointer memory;
- gather or scatter;
- store with sparse or unknown mask;
- subword or f16 memory;
- `boundary_check` / padding-option style producer semantics;
- vector rank greater than 1.

This contract does not add executable lowering. Phase 10.4 owns the static
classifier/lowering implementation and diagnostics. Hardware proof remains for
later Phase 10 hardware prompts.

## 12. Control-flow boundary

`scf` and `cf` are accepted as value-layer control syntax. This does not mean
VC4Kernel can consume raw `scf`.

No raw `scf` may survive into verified VC4Kernel later. Canonicalization or
lowering decisions belong to later phases.

## 13. Math policy boundary

`arith` and `math` dialect syntax belongs to the value surface, but math policy
is not resolved by Phase 3.

Approximate SFU is explicit policy only; exact/default math must not silently
lower to SFU. Later lowering phases must distinguish exact, finite, and
approximate policy before mapping math to VC4Kernel mechanisms.

## 14. Phase 14 ML storage and numeric conversion contract

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

Accepted Phase 14 value-surface forms are:

- `F16_STORAGE_TO_F32_COMPUTE`: `vector.transfer_read` of `vector<16xf16>`
  from rank-1 flattened or Phase 11 row-strided `#vc4value.global` f16
  storage, followed by `arith.extf` to `vector<16xf32>`.
- `F32_COMPUTE_TO_F16_STORAGE`: finite f32 compute followed by
  `arith.truncf vector<16xf32> -> vector<16xf16>` and
  `vector.transfer_write` to f16 global storage. This requires explicit
  `vc4value.f16_storage_policy = "finite"` on the transfer or containing
  kernel.
- `F16_ROW_STRIDED_STORAGE`: Phase 11 flattened or row-slice memory forms may
  carry f16 element type when lane transfers remain contiguous `vector<16>`.
- `F16_GEMV_V0_INPUT_STORAGE`: the Phase 13 row-wise dot composite may read
  f16 A/X inputs, promote them to f32, multiply in f32, reduce with the existing
  finite-tree f32 policy, and store scalar f32 output.

Phase 14 does not claim exact/default f16 rounding semantics. The finite f16
storage policy is a storage-conversion policy over finite values, not native
f16 arithmetic and not a broad IEEE conversion proof.

Staged Phase 14 value forms are:

- native f16 arithmetic;
- f16 reduction or f16 accumulation;
- bf16/fp8 native conversion or arithmetic;
- int8/i16 quantized storage, scale, zero-point, and narrow quantized
  arithmetic policy;
- i32/index-to-f32 casts, staged by the current lower-half numeric-cast gap;
- `arith.fptosi`, `arith.fptoui`, and fp-to-int casts;
- exact/default numeric casts without explicit policy;
- saturation and rich rounding-mode conversions;
- approximate SFU/math and softmax.

The verifier enforces this as a surface and policy boundary only. It does not
emit f16 pack/unpack, does not lower value IR to VC4Kernel, does not import
TTIR, and does not run hardware.

## 14.2 Phase 15 approximate-SFU and softmax contract

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

Phase 15 admits the narrow value-surface forms needed by controlled Triton
SFU/softmax fixtures:

- `APPROX_SFU_POLICY`: approximate math must carry explicit
  `vc4value.math_policy = "approx_sfu"` plus a finite `vc4value.fp_domain`
  spelling such as `"finite"`, `"finite_positive"`, or `"finite_nonzero"`.
  Exact/default math without policy is rejected.
- `VECTOR_F32_EXP_APPROX`: `math.exp` over scalar `f32` or `vector<16xf32>`
  is accepted only as approximate SFU math over finite bounded inputs.
- `VECTOR_F32_RECIP_DIV_APPROX`: `arith.divf` over scalar `f32` or
  `vector<16xf32>` is accepted only as approximate reciprocal/division under
  explicit policy. Denominators must be finite and away from zero in later
  hardware fixtures; exact division is not claimed.
- `OPTIONAL_LOG_RSQRT_APPROX`: `math.log` and `math.rsqrt` are accepted
  because the locked lower-half surface has log and rsqrt SFU modes. Their
  value surface requires positive finite domains.
- `FINITE_F32_MAX_REDUCTION`: the exact accepted spellings are
  `vector.reduction <maxnumf>` and `vector.reduction <maximumf>` over
  `vector<16xf32>` to scalar `f32`, with `vc4value.fp_domain = "finite"`,
  `vc4value.reduction_policy = "finite_tree"`, and
  `vc4value.max_policy = "finite"`.
- `SCALAR_TO_VECTOR_F32_BROADCAST`: `vector.broadcast` from scalar `f32` to
  `vector<16xf32>` is admitted for softmax scalar-to-lane splats.
- `SOFTMAX_V0_COMPOSITE`: no `vc4value.softmax` op is introduced. The value
  surface accepts the explicit composite:

```text
max = vector.reduction maxnumf/maximumf(active_x)
shifted = x - vector.broadcast(max)
expv = math.exp(shifted)
expv_masked = arith.select(mask, expv, zero)
denom = vector.reduction add(expv_masked)
inv = approximate reciprocal/division of denom
out = expv_masked * vector.broadcast(inv)
vector.transfer_write out with the tail mask
```

The accepted softmax domain is one block with active lane count 1..16 and
finite bounded inputs. Approximate output tolerance is a later hardware policy,
not a value-surface proof.

Still staged:

- exact/default math;
- NaN/Inf semantics;
- active-count-zero softmax without an explicit finite no-op guard;
- multiblock softmax;
- block-pointer softmax;
- full attention and FlashAttention;
- generic division without approximate policy;
- quantized softmax.

## 14.3 Subword and f16 storage boundary

Subword lowering remains staged for later phases. f16 storage conversion is a
Phase 14 value feature through storage conversion plus f32 compute; native f16
arithmetic remains rejected by the locked VC4Kernel surface.

Native bf16/fp8 arithmetic or conversion remains outside the locked VC4Kernel
surface unless a future phase proves an emulation strategy and updates the
value contract.

## 14.4 Phase 16 attention-apply v0 contract

PHASE16_VALUE_ATTENTION_APPLY_V0_CONTRACT=LOCKED
VALUE_ATTENTION_APPLY_V0_SURFACE=ACCEPTED
PRECOMPUTED_SCORES_ONLY=YES
TRANSPOSED_V_LAYOUT_REQUIRED=YES
SCALAR_GLOBAL_LOAD_STAGED=YES
NONTRANSPOSED_V_GATHER_STAGED=YES
K_ZERO_ATTENTION_APPLY_STAGED_OR_GUARD_REQUIRED=YES
READY_FOR_PHASE16_4_VALUE_ATTENTION_APPLY_STATIC=YES
READY_FOR_TRITON=NO

Phase 16 admits attention-apply v0 as a standard value IR composite. It does
not add a `vc4value.attention`, `vc4value.softmax_apply`, dot, contract, or
full-attention operation.

The accepted composite is:

```text
scores = vector.transfer_read scores : vector<16xf32>
optional scaled_scores = scores * vector.broadcast(scalar_f32_scale)
active_scores = arith.select(mask, scaled_scores_or_scores, finite_low)
max = vector.reduction maxnumf/maximumf(active_scores)
shifted = active_scores - vector.broadcast(max)
e = math.exp(shifted)
active_e = arith.select(mask, e, zero)
denom = vector.reduction add(active_e)
probs = active_e * vector.broadcast(approx_recip(denom))
v = vector.transfer_read transposed_v : vector<16xf32>
weighted = probs * v
acc = vector.reduction add(weighted)
memref.store acc, output[scalar_index]
```

The same composite may read f16 score/Vt storage when each f16 vector load is
immediately promoted to f32 compute under the Phase 14 finite f16 storage
policy. Native f16 arithmetic remains staged.

Memory requirements:

- scores and Vt use rank-1 flattened memory or the Phase 11 row-strided
  row-slice memory form;
- V is transposed so lanes are contiguous over K, `Vt[d, offs]`;
- output uses the Phase 12 scalar f32 `memref.store` form,
  `O[q * LDO + d]`;
- scalar global loads are staged, including scalar memory loads for scale;
- non-transposed V gather/lane-varying stride forms such as
  `V + offs * D + d` are staged.

Domain and scale requirements:

- active count K is 1..16;
- scores and V values are finite and bounded;
- the denominator is positive by construction for K in 1..16;
- natural `math.exp` and reciprocal/division require explicit
  `vc4value.math_policy = "approx_sfu"` and finite domain policy;
- approximate output tolerance is inherited from the Phase 15 natural-exp
  softmax policy;
- scale is an optional scalar f32 kernel argument or constexpr splat, not a
  scalar global load.

The verifier recognizes the accepted metadata spelling:

```text
vc4value.attention_apply_v0 = "precomputed_transposed_v_active_1_to_16"
```

The verifier also provides deterministic staged diagnostics for explicit
metadata spellings covering zero-active attention-apply, non-transposed V
gather, scalar global load, online/multiblock softmax, QK score generation,
`tl.dot`, `tt.dot`, and `vector.contract`.

Still staged:

- K=0 unless a later phase proves an explicit finite no-op guard;
- QK score generation;
- non-transposed V layout/gather;
- scalar global load;
- online or multiblock softmax;
- `tl.dot`, `tt.dot`, and `vector.contract`;
- block pointers and tensor descriptors;
- full attention and FlashAttention.

## 15. Shuffle, rotate, and lane broadcast boundary

Dynamic rotate and rotate-derived composites are accepted at the locked
VC4Kernel surface, but Phase 3 does not lower value shuffle patterns.

Lane broadcast is a composite idiom, not a vc4kernel op. Arbitrary
shuffle/permutation is rejected at VC4Kernel. Future value-level emulation or
limited shuffle support must update the value-surface matrix and verifier
contract.

## 16. Future lowering gates

Phase 3 must keep:

```text
READY_FOR_VALUE_TO_VC4KERNEL_LOWERING=NO
READY_FOR_TRITON=NO
```

Phase 4 owns value ABI lock. Phase 5+ owns value-to-VC4Kernel lowering. TTIR
import remains disallowed until the handwritten value path is proven.

The Phase 4 ABI handoff permits a future Phase 5 V1 package to target only the
vector<16> i32/f32 elementwise rank-1 contiguous subset first. Broader fixed
vectors, rank-2 memrefs, subword storage, and f16 storage remain staged unless
a later phase proves and implements their lowering.

The verifier must not add conversion patterns, target emission, runtime hooks,
hardware candidate generation, or hardware fixtures.

## 17. Diagnostics and audit contract

Diagnostics should name the invalid operation, dialect, type, or metadata field
and explain the value-surface boundary being enforced. Negative tests must
cover forbidden producer dialects, target/lower-half dialects, vc4value scope
creep, invalid kernel metadata, invalid launch axis/grid-rank relationships,
direct memref side effects, sparse store ops, and unsupported type shapes.

The support matrix is:

```text
compiler/docs/vc4_value_surface_support_matrix.json
```

Phase 4 ABI audits validate the matrix, source tests, pass registration,
public ABI corpus, absence of hidden descriptor claims, absence of hardware-path
selection in `#vc4value.global`, and absence of unscoped lowering/Triton claims.
The source-controlled lock modes are:

```text
python3 compiler/test/ValueSurface/Support/check_vc4_value_surface_matrix.py compiler/docs/vc4_value_surface_support_matrix.json --mode phase4-abi-lock
python3 compiler/test/ValueSurface/Support/audit_vc4_value_surface.py --repo-root . --matrix compiler/docs/vc4_value_surface_support_matrix.json --mode phase4-abi-lock
```

## 18. Phase readiness lines

VC4_VALUE_SURFACE_VERIFIER_CONTRACT_DOC=YES
VC4_VALUE_SURFACE_SUPPORT_MATRIX_SEEDED=YES
READY_FOR_PHASE3C_VALUE_SURFACE_PASS=YES
READY_FOR_VALUE_TO_VC4KERNEL_LOWERING=NO
READY_FOR_TRITON=NO

## 19. Phase 11 strided/ranked memory verifier delta

PHASE11_VALUE_STRIDED_RANKED_MEMORY_CONTRACT=LOCKED
RANK2_ROW_SLICE_IDENTITY_SURFACE=ACCEPTED
RANK2_ROW_SLICE_STRIDED_OUTER_DYNAMIC_SURFACE=ACCEPTED
MEMREF_DIM_METADATA_TO_SCALAR_ARG_CONTRACT=LOCKED
HIDDEN_MEMREF_DESCRIPTOR_ALLOWED=NO
GATHER_LANE_STRIDE_STAGED=YES
READY_FOR_PHASE11_4_VALUE_RANKED_STRIDED_STATIC=YES
READY_FOR_TRITON=NO

The Phase 11 value surface admits only the strided/ranked memory skeletons that
preserve contiguous vector lanes:

- rank-1 flattened scalar strided address arithmetic feeding rank-1 identity
  vector transfers;
- rank-2 identity row-slice transfers with scalar `[row, col]` indices;
- rank-2 explicit `strided<[?, 1], offset: 0>` row-slice transfers with
  `shape_args` for dynamic dimensions and one `stride_args` entry for the outer
  row stride;
- `memref.dim` metadata that resolves to public scalar extent arguments.

The verifier rejects hidden descriptor extraction through
`memref.extract_strided_metadata` and rejects rank-2 explicit strided layouts
whose inner stride is dynamic or not static 1. Executable lowering remains a
later phase; these checks are surface/ABI contract enforcement only.
