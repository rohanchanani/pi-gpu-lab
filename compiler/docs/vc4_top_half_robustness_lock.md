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
