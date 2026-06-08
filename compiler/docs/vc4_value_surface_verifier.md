# VC4 Value Surface Verifier Contract

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
- memref ABI or launch ABI correctness;
- vector transfer lowering;
- math lowering policy correctness;
- VC4Kernel verifier correctness;
- scheduled VC4 correctness;
- runtime or hardware correctness;
- TTIR/Triton import support.

Lowerability starts in Phase 5 after Phase 4 ABI lock. TTIR import remains
disallowed until the handwritten value path is proven.

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
verifies the local axis domain 0..2; Phase 3 adds the function-level grid-rank
relationship.

Other metadata conventions such as `vc4value.target_profile` and
`vc4value.math_policy` remain ordinary named attrs until later phases assign
stronger policy meaning.

## 9. Type guardrails

Phase 3 accepts fixed, non-scalable vector types and ranked memref types within
the initial value-surface shape contract.

The verifier rejects:

- scalable vectors;
- unranked memrefs;
- memref rank greater than 2;
- vector rank greater than 2.

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
memref.store
memref.atomic_rmw
memref.generic_atomic_rmw
memref.copy
memref.dma_start
memref.dma_wait
```

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

## 14. Subword and f16 storage boundary

Subword and f16 storage lowering is staged for later phases. f16 storage
conversion is a future value feature through storage conversion plus f32
compute; native f16 arithmetic remains rejected by the locked VC4Kernel surface.

Native bf16/fp8 arithmetic or conversion remains outside the locked VC4Kernel
surface unless a future phase proves an emulation strategy and updates the
value contract.

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

Later Phase 3 audits must validate the matrix, source tests, pass registration,
and absence of lowering/Triton claims.

## 18. Phase readiness lines

VC4_VALUE_SURFACE_VERIFIER_CONTRACT_DOC=YES
VC4_VALUE_SURFACE_SUPPORT_MATRIX_SEEDED=YES
READY_FOR_PHASE3C_VALUE_SURFACE_PASS=YES
READY_FOR_VALUE_TO_VC4KERNEL_LOWERING=NO
READY_FOR_TRITON=NO
