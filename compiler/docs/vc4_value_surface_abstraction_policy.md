# VC4 Value Surface Abstraction Policy

## 1. Purpose

This Phase 3.5 policy refines the VC4 value-surface contract. The value
surface is a target-profiled standard MLIR subset, not VC4Kernel and not
arbitrary MLIR.

The accepted value-layer dialect family remains:

```text
func + tiny vc4value + vector + memref + arith + math + scf/cf
```

Phase 3.5 does not add lowering, TTIR import, hardware fixtures, or new
`vc4value` operations. `READY_FOR_VALUE_TO_VC4KERNEL_LOWERING` remains `NO`,
and `READY_FOR_TRITON` remains `NO`.

## 2. Hardware Boundary

The value surface must not expose VC4 hardware path names or mechanisms.
Source-visible value IR has no TMU, VDR, VPM, VDW, VC4Kernel fragments,
barriers, resource metadata, physical QPU IDs, or lower-half operations.

Those names belong to later value-to-VC4Kernel planning after the value ABI and
handwritten lowering path are proven.

## 3. Vector Shape Policy

`vector<16xT>` is the Phase 5 V1 lowerable fragment-normal form. It is not the
global value-layer type limit.

Fixed rank-1 `vector<NxT>`, including `N != 16`, is surface-admissible and
staged when `T` is an allowed element type. Later lowering will split these
vectors into `vector<16>` fragments plus tails or loops.

Fixed rank-2 `vector<MxNxT>` is surface-admissible and staged when `T` is an
allowed element type. Later lowering will use tile, contract, and VPM planning.

Scalable vectors are deterministic rejects in the initial value surface.

## 4. Element Types

Allowed vector element types in Phase 3.5 are exactly:

```text
i1, index, i8, i16, i32, f16, f32
```

Element-type meaning:

- `i1` is for masks and predicate sources.
- `index` is for value-level address and program-id arithmetic; later lowering
  must prove legal conversion to i32 where needed.
- `i8` and `i16` are subword storage and data-movement surface types, not
  native arithmetic promises.
- `f16` is a storage-conversion surface type. f16 storage conversion plus f32
  compute is accepted downstream, but native f16 arithmetic is not accepted.
- `i32` and `f32` are the first executable arithmetic carriers for Phase 5 V1.

Unsupported vector element types are deterministic rejects.

## 5. Tensor And Linalg

Tensor and linalg dialects are forbidden in the initial value surface. Future
producer layers may lower tensor or linalg-like programs into this value
surface, but tensor and linalg operations are not Phase 3.5 value IR.

Do not add tensor or linalg to the value surface only to make the layer appear
more abstract. The abstraction point is a target-profiled standard MLIR value
subset, not arbitrary MLIR.

## 6. Readiness Lines

```text
VC4_VALUE_SURFACE_FIXED_VECTOR_TYPES_GENERALIZED=YES
VC4_VALUE_SURFACE_VECTOR16_IS_V1_LOWERABLE_NOT_GLOBAL_LIMIT=YES
VC4_VALUE_SURFACE_NON16_FIXED_VECTORS_STAGED=YES
VC4_VALUE_SURFACE_RANK2_FIXED_VECTORS_STAGED=YES
VC4_VALUE_SURFACE_SCALABLE_VECTORS_REJECTED=YES
VC4_VALUE_SURFACE_TENSOR_LINALG_INITIAL_REJECT=YES
READY_FOR_PHASE4_VALUE_ABI=YES
READY_FOR_VALUE_TO_VC4KERNEL_LOWERING=NO
READY_FOR_TRITON=NO
```
