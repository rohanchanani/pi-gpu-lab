# VC4 Vector/Triton Phase 7.5 Output Module Design

READY_FOR_TRITON=NO

## Purpose

The C++ TTIR-to-VC4Value importer must not mix parsed TTIR input operations and
generated value-layer operations in the same module while lowering. Phase 7.5
Round 3 replaced the old input-module mutation path with output-module
construction so unsupported TTIR cannot leave partial value IR behind.

This is a lifetime and correctness boundary, not a feature expansion. The
importer still supports only the Phase 7.5 elementwise V1 TTIR subset.

## Current Architecture

The pass analyzes the input `ModuleOp` first:

- top-level operations must be `tt.func`;
- at least one `tt.func` is required;
- unknown module attributes are rejected rather than silently preserved;
- output function names are sanitized and checked for duplicate collisions;
- every source function is planned before the original module is committed.

The pass then creates a clean scratch module and lowers every planned source
function into that scratch module as value IR:

```text
tt.func input module
  -> FunctionPlanner read-only analysis
  -> TTIRToValueModuleBuilder scratch module
  -> func.func + vc4value + vector + memref + arith/math/scf/cf
```

The scratch output module is verified before commit. It may contain only the
value-layer/core dialects required by the value surface. It must not contain
`tt`, `ttg`, `triton_gpu`, backend dialects, `vc4kernel`, `ssavc4`, or scheduled
`vc4`.

Only after all functions lower successfully and the scratch module passes the
output verifier does the pass replace the original module body with the output
operations.

## Failure Atomicity

Failure before commit leaves the input module unmodified. This covers:

- unsupported module attributes;
- missing `tt.func`;
- unsupported top-level operations;
- duplicate sanitized output symbols;
- unsupported TTIR operations such as `tt.dot`, `tt.reduce`, axes 1/2, and
  non-V1 ranges;
- lowerer failures;
- output verifier failures.

The pass signals failure and does not intentionally emit partial value-layer
output for unsupported inputs.

## Multi-Function Policy

Multiple independent Phase 7.5 V1 `tt.func`s are supported. They lower
atomically to multiple value-layer `func.func`s in deterministic source order.

If any source function is unsupported, no value output is committed. This policy
keeps the current useful multi-function behavior while making failure semantics
explicit and testable.

Cross-function calls are not part of the Phase 7.5 V1 support claim.

## Module Attr Copy Policy

Phase 7.5 emits a fresh bare value module. It does not silently preserve TTIR,
Triton backend, target, or unknown module attributes.

Unknown input module attributes are rejected until a later phase deliberately
classifies and tests a value-safe module attribute policy.

## Forbidden Workaround

The importer must not use `Operation::remove`, per-function unlinking, or leaked
TTIR operations as a lifetime solution.

Old TTIR operations are destroyed or replaced only during final commit through
standard MLIR ownership. If a future toolchain trips an assertion during
standard erasure, that is a blocker to diagnose, not a reason to restore the
unlink workaround.

## Relationship To Later Phases

This design is required before Phase 8 control-flow and loop-boundary work
because later TTIR forms will add more failure points. Atomic output-module
construction ensures staged rejects remain honest and do not leave partially
lowered value IR.

Future TTIR feature expansion must continue to enter through the value surface:

```text
TTIR -> standard VC4 value layer -> vc4kernel -> ssavc4 -> scheduled vc4
```

`READY_FOR_TRITON=NO` remains true. Full Triton support is still staged and must
be proven feature by feature through the value-layer path.
