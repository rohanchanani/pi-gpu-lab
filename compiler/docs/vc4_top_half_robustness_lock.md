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
