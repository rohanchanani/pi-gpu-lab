TOP_HALF_ROBUSTNESS_RESULT=LOCKED
NO_SILENT_TTIR_OP_DROPS=YES
REQUIRED_UNSUPPORTED_OPS_STAGE_EXACTLY=YES
DEAD_PROOF_OPS_WHITELISTED_AND_AUDITED=YES
ARITH_EXTSI_ANDI_REQUIRED_USE_REJECTS=PASS
SOURCE_NAMES_ARE_METADATA_ONLY=YES
ARG_NAME_REWRITES_REMOVED=YES
ARG_NAME_COLLISIONS_DISAMBIGUATED=YES
SHAPE_ARGS_DERIVED_FROM_STRUCTURAL_BOUND_VALUE=YES
READY_FOR_TOP_HALF3_OUTPUT_BOUNDARY_HARDENING=YES
VALUE_OUTPUT_BOUNDARY_HARDENED=YES
UNREALIZED_CONVERSION_CAST_REJECTS=PASS
PRODUCER_BACKEND_LOWER_DIALECT_OUTPUT_REJECTS=PASS
TOP_HALF_FRONTEND_ROBUSTNESS_AUDIT_SCRIPT=YES
NO_STRINGLY_SEMANTIC_CLASSIFICATION=YES
READY_FOR_TOP_HALF4_STATIC_PIPELINE_AUDITS=YES
TOP_HALF_STATIC_ROBUSTNESS_LOCK=YES
CONTROLLED_TTIR_STATIC_PIPELINE=PASS
ELEMENTWISE_TTIR_STATIC_PIPELINE=PASS
FRONTEND_ROBUSTNESS_AUDIT=PASS
STAGED_FEATURE_AUDIT=PASS
NO_TTIR_TOOLCHAIN_REBUILD=YES
NO_TTIR_REGEN_IN_TOP_HALF_HARDENING=YES
READY_FOR_TOP_HALF5_HARDWARE_MIXED_FINAL=YES
TTIR_ISOLATION_REGRESSION=PASS
TTIR_MIXED_REGRESSION=PASS
LAYERED_REGRESSION_POLICY_APPLIED=YES
READY_FOR_FEATURE_LADDER_PHASE9=YES
READY_FOR_TRITON=NO

# VC4 Top-Half Robustness Lock

TopHalf2 hardens the TTIR-to-VC4Value importer without adding new value or TTIR
feature support.

The importer no longer has a broad result-producing op drop path. Successful
per-op lowering must now account for results through value/pointer/canonical
bindings, handle a zero-result op directly, or record an exact ignored-dead
proof/canonicalization op from the narrow whitelist.

Required `arith.extsi` and `arith.andi` uses stage with precise diagnostics.
Dead proof uses are ignored only when recursively unrequired.

Argument names are metadata only. Pointer direction and scalar shape bounds are
derived from TTIR SSA/use-def structure. Source/location names are sanitized for
metadata readability, hard-coded rewrites such as `_ptr` stripping and
`n_elements -> n` are removed, and sanitized collisions are deterministically
disambiguated.

This lock does not implement Phase 9/10 mask classification, common-plan
generalization, gather/scatter, block pointers, reductions, dot, numeric casts,
or generic vector index-cast lowering. `READY_FOR_TRITON=NO` remains true.

## TopHalf3 Output Boundary

TopHalf3 centralizes the importer value-output boundary verifier. The output
module may contain `builtin.module` as the MLIR container and value-layer body
ops from `func`, `vc4value`, `vector`, `memref`, `arith`, `math`, `scf`, and
`cf` only. `func` is restricted to function structure and returns.

The verifier rejects `builtin.unrealized_conversion_cast`, arbitrary builtin
body ops, producer/backend/lower-half dialects, and unconverted tensor or
Triton pointer types. The frontend robustness audit locks out regex, printed
IR/type semantics, fixture/path/name special cases, operation unlink/remove
workarounds, direct lower-half emission, and broad successful lowering paths
without accounting.

## TopHalf4 Static Robustness Lock

TopHalf4 locks the static top-half state before hardware/mixed regression. The
accepted Phase 8.5 controlled TTIR snapshots and the existing Phase 7/7.5
elementwise TTIR snapshots lower through:

```text
TTIR -> VC4Value -> value verifier -> scf-to-cf -> VC4Kernel -> SSAVC4 -> scheduled VC4
```

Each stage is boundary-scanned: value output has no TTIR/backend/lower-half ops
or unrealized casts, VC4Kernel has no producer ops, SSAVC4 has no producer or
VC4Kernel ops, and scheduled VC4 has no producer, VC4Kernel, or SSAVC4 ops.

The frontend robustness audits pass for no silent TTIR op drops, exact staging
of required unsupported operations, whitelisted dead-proof ops, metadata-only
source names, removed argument-name rewrites, disambiguated metadata names,
structural shape-arg derivation, no stringly semantic classification, and the
hardened output boundary.

No TTIR was regenerated, no Triton or LLVM toolchain was created/installed/built,
and no hardware was run in this static lock. `READY_FOR_TRITON=NO` remains true.

## TopHalf5 Hardware/Mixed Final Lock

TopHalf5 applies the layered regression policy after the TopHalf2-4 hardening
commits. The lowest touched implementation layer is the TTIR importer, so the
required hardware/mixed scope is TTIR isolation plus TTIR mixed regression.
Value, VC4Kernel, SSAVC4, scheduled VC4, artifact, and runtime mixed suites are
not required by layer policy for this package.

The source-controlled Phase 7/7.5 TTIR elementwise isolation fixtures and Phase
8.5 controlled TTIR control-flow isolation fixtures passed on hardware with
`active_qpus=12`, zero mismatches, zero sentinel mismatches, and zero launch
failures. The full TTIR mixed acceptance suite also passed on hardware with the
same invariants.

The final audits pass for no silent TTIR op drops, exact staging of required
unsupported operations, whitelisted dead-proof ops, metadata-only source names,
removed argument-name rewrites, disambiguated metadata names, structural
shape-arg derivation, no stringly semantic classification, output-boundary
hardening, TTIR mixed claims, and no toolchain/generator drift.

No TTIR was regenerated, no Triton or LLVM toolchain was created/installed/built,
and `READY_FOR_TRITON=NO` remains true.

## Phase 10.7 Mask/Memory Importer Audit Extension

The top-half robustness audit now includes Phase 10 mask/memory importer
checks. The audit locks structural TTIR memory-mask classification, exact
staged diagnostics for sparse memory masks and nonzero load `other`, no
fixture/name/path special casing, no TTIR text parsing, no regex or substring
semantic classification, and no direct lower-half output.

FRONTEND_ROBUSTNESS_AUDIT=PASS
READY_FOR_TRITON=NO

## Phase 11.7 Strided-Memory Importer Audit Extension

The Phase 11.7 importer audit locks structural row-strided pointer
classification for controlled TTIR memory. The accepted classifier is based on
SSA/use-def structure over `tt.addptr`, `tt.splat`, `tt.make_range`, and
`arith` producers; it does not parse printed TTIR, use regexes, special-case
fixture/kernel/path/source names, infer rank from public names, or emit
lower-half IR directly.

TTIR_STRIDED_MEMORY_IMPORTER_STATIC=PASS
TTIR_ROW_STRIDED_POINTER_LOWERING=YES
TTIR_LANE_VARYING_STRIDE_GATHER_REJECTS=PASS
TTIR_COLUMN_SLICE_REJECTS=PASS
TTIR_RANK_INFERENCE_FROM_NAMES=NO
FRONTEND_ROBUSTNESS_AUDIT=PASS
READY_FOR_PHASE11_8_TTIR_HARDWARE_ISOLATION=YES
READY_FOR_TRITON=NO

## Phase 12.7 Reduction Importer Audit Extension

The Phase 12.7 importer audit locks structural TTIR reduction classification
for controlled real Triton `tl.sum` snapshots. The accepted classifier resolves
private helper symbols and inspects `tt.reduce` regions, reducer block
arguments, `tt.reduce.return`, and typed `arith.addi` / `arith.addf`
combiners. It does not parse printed TTIR, use regexes, special-case fixture
names, infer semantics from source paths or public names, or emit lower-half IR
directly.

TTIR_REDUCTION_IMPORTER_STATIC=PASS
TTIR_TL_SUM_ADD_LOWERING=YES
TTIR_SCALAR_REDUCTION_STORE_LOWERING=YES
TTIR_F32_REDUCTION_FINITE_TREE_POLICY=YES
TTIR_NON_ADD_REDUCTIONS_REJECT=PASS
TTIR_DOT_GEMV_STAGED_FOR_PHASE13=YES
FRONTEND_ROBUSTNESS_AUDIT=PASS
READY_FOR_PHASE12_8_TTIR_HARDWARE_ISOLATION=YES
READY_FOR_TRITON=NO

## Phase 13.7 GEMV Row-Dot Importer Audit Extension

The Phase 13.7 importer audit locks structural TTIR GEMV row-wise dot
classification for controlled real Triton `tl.sum(a * x, axis=0)` snapshots.
The accepted classifier uses SSA/use-def structure, typed operation names,
`tt.reduce` helper resolution, reducer region bodies, and exact policy
attributes to lower the product, add reduction, and scalar store through the
standard value layer.

TTIR_GEMV_ROWWISE_DOT_IMPORTER_STATIC=PASS
TTIR_GEMV_TL_SUM_PRODUCT_LOWERING=YES
TTIR_GEMV_SCALAR_RESULT_STORE_LOWERING=YES
TTIR_GEMV_F32_FINITE_TREE_POLICY=YES
TTIR_TL_DOT_TT_DOT_REJECT=PASS
TTIR_MULTIBLOCK_K_ACCUMULATION_STAGED=YES
FRONTEND_ROBUSTNESS_AUDIT=PASS
READY_FOR_PHASE13_8_TTIR_HARDWARE_ISOLATION=YES
READY_FOR_TRITON=NO

The audit keeps name/path/fixture special casing, raw TTIR text parsing,
regex/substring semantic classification, direct lower-half output,
toolchain/generator work, `tl.dot`/`tt.dot`, vector.contract, and multi-block K
accumulation outside the accepted importer path.

## Phase 14.7 ML Storage/Numeric Importer Audit Extension

The Phase 14.7 importer audit locks structural f16 storage and numeric policy
classification for controlled real Triton snapshots. The accepted classifier is
based on Triton pointer element types, ranked tensor element types, SSA use-def
chains, exact operation names, and value-surface policy attributes. It lowers
only to the standard value layer and never emits VC4Kernel, SSAVC4, or
scheduled VC4 directly.

TTIR_F16_STORAGE_IMPORTER_STATIC=PASS
TTIR_F16_LOAD_F32_COMPUTE_LOWERING=YES
TTIR_F32_COMPUTE_F16_STORE_LOWERING=YES
TTIR_F16_ROW_DOT_F32_ACCUM_LOWERING=YES
TTIR_I32_TO_F32_CAST_STATUS=STAGED_BY_LOWER_HALF_GAP
TTIR_NATIVE_F16_ARITHMETIC_REJECT=PASS
TTIR_FP_TO_INT_REJECT=PASS
FRONTEND_ROBUSTNESS_AUDIT=PASS
READY_FOR_PHASE14_8_TTIR_HARDWARE_ISOLATION=YES
READY_FOR_TRITON=NO

The audit keeps native f16 arithmetic, f16 accumulation, bf16/fp8, quantized
int storage, fp-to-int, i32-to-f32, SFU/math, softmax, `tt.dot`,
vector.contract, GEMM, raw TTIR text parsing, regex/substring semantic
classification, fixture/path/name special casing, toolchain work, and direct
lower-half output outside the accepted importer path.

## Phase 16.7 Attention-Apply Importer Audit Extension

The Phase 16.7 importer audit locks controlled TTIR attention-apply v0 static
lowering as composition of structural C++ importer support for program IDs,
row-contiguous pointer slices, canonical masks, finite reductions, natural-exp
softmax, optional scalar argument splats, weighted add reduction, and scalar
stores. It does not parse raw TTIR text, use regex or substring semantic
classification, special-case fixture names, infer source path semantics, emit
lower-half IR directly, accept scalar `tt.load`, or accept non-transposed V
gather/lane-varying stride.

TTIR_ATTENTION_APPLY_V0_IMPORTER_STATIC=PASS
TTIR_ATTENTION_APPLY_V0_LOWERING=YES
TTIR_ATTENTION_APPLY_V0_TRANSPOSED_V_LAYOUT=YES
TTIR_ATTENTION_APPLY_V0_NATURAL_EXP_SOFTMAX=YES
TTIR_ATTENTION_APPLY_V0_WEIGHTED_SUM=YES
TTIR_NONTRANSPOSED_V_GATHER_REJECT=PASS
TTIR_SCALAR_GLOBAL_LOAD_REJECT=PASS
FRONTEND_ROBUSTNESS_AUDIT=PASS
READY_FOR_PHASE16_8_TTIR_HARDWARE_ISOLATION=YES
READY_FOR_TRITON=NO
