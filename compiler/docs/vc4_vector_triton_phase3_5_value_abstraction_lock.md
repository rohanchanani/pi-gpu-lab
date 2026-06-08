# VC4 Vector/Triton Phase 3.5 Value-Surface Abstraction Lock

## 1. Purpose

Phase 3.5 refines the Phase 3 value-surface verifier, tests, matrix, and docs
so value-surface admissibility is not permanently limited to `vector<16xT>`.
The refinement keeps `vector<16xT>` as the Phase 5 V1 executable fragment
subset while admitting other fixed, non-scalable vector shapes as staged
surface IR.

Phase 3.5 is static-only. It does not add value-to-VC4Kernel lowering, TTIR
import, hardware fixtures, runtime behavior, or hardware execution.

## 2. Final Abstraction Boundary

The value layer is a target-profiled standard MLIR value surface:

```text
func + tiny vc4value + vector + memref + arith + math + scf/cf
```

It remains above VC4Kernel and must lower through the locked path:

```text
standard value layer -> vc4kernel -> ssavc4 -> scheduled vc4
  -> artifacts/runtime/hardware
```

The value surface is not VC4 hardware IR. It does not expose TMU, VDR, VPM,
VDW, VC4Kernel fragments, barriers, resource metadata, physical QPU IDs, or
lower-half operations.

## 3. Vector Type Policy

`vector<16xT>` is the Phase 5 V1 lowerable fragment shape. It is not the global
value-layer vector limit.

Fixed rank-1 `vector<NxT>` where `N != 16` is surface-admissible and staged for
later splitting into `vector<16>` fragments plus tails or loops.

Fixed rank-2 `vector<MxNxT>` is surface-admissible and staged for later tile,
contract, and VPM planning.

Scalable vectors are rejected.

Rank greater than 2 vectors are rejected in the initial Phase 3.5 profile. This
is an initial surface-policy reject, not a permanent hardware-impossibility
proof.

## 4. Element Type Policy

Allowed vector element types in Phase 3.5 are:

```text
i1, index, i8, i16, i32, f16, f32
```

`i1` is for masks and predicate sources.

`index` is for value-layer address and program-id arithmetic. Later phases must
prove any target i32 conversion.

`i8` and `i16` are subword storage and data-movement surface types. They are not
native arithmetic promises.

`f16` is a storage-conversion surface type. f16 storage conversion plus f32
compute remains the accepted downstream model; native f16 arithmetic remains
rejected.

`i32` and `f32` are the first executable arithmetic carriers for Phase 5 V1.

## 5. Tensor/Linalg Policy

Tensor and linalg dialects are forbidden in the initial value surface. Future
producer layers may lower tensor or linalg-like programs into this value
surface, but tensor and linalg operations are not Phase 3.5 value IR.

This preserves a target-profiled standard MLIR value subset rather than
expanding the boundary to arbitrary MLIR.

## 6. Matrix/Audit Proof

The value-surface support matrix remains a 26-row Phase 3 contract with Phase
3.5 equivalents for the abstraction repair:

- `value.vector.fixed_rank1.width16_fragment_carriers`
- `value.vector.fixed_rank1.non16_staged_split`
- `value.vector.fixed_rank2.staged_tile_contract`
- `value.vector.scalable_reject`
- `value.vector.unsupported_element_reject`
- `value.tensor_linalg.initial_reject`
- `value.vector.f16_storage_not_native_arith`

The equivalent rows distinguish:

- `surface_verifier_behavior=accept` or `reject`
- `phase5_lowering_behavior=lowerable_v1`
- `phase5_lowering_behavior=staged_split_required`
- `phase5_lowering_behavior=staged_tile_or_contract_required`
- `phase5_lowering_behavior=staged_storage_only`
- `phase5_lowering_behavior=rejected`

Final proof commands:

```text
python3 compiler/test/ValueSurface/Support/check_vc4_value_surface_matrix.py compiler/docs/vc4_value_surface_support_matrix.json
python3 compiler/test/ValueSurface/Support/audit_vc4_value_surface_abstraction.py --repo-root . --matrix compiler/docs/vc4_value_surface_support_matrix.json --mode phase3_5
```

## 7. Tests Added/Confirmed

Confirmed Phase 3 valid tests:

- `compiler/test/ValueSurface/value-surface-basic-launch-valid.mlir`
- `compiler/test/ValueSurface/value-surface-grid-rank-3-valid.mlir`
- `compiler/test/ValueSurface/value-surface-vector-types-valid.mlir`
- `compiler/test/ValueSurface/value-surface-memref-types-valid.mlir`
- `compiler/test/ValueSurface/value-surface-vector-transfer-valid.mlir`
- `compiler/test/ValueSurface/value-surface-staged-ops-valid.mlir`

Added Phase 3.5 valid tests:

- `compiler/test/ValueSurface/value-surface-staged-fixed-vector-widths-valid.mlir`
- `compiler/test/ValueSurface/value-surface-staged-rank2-vectors-valid.mlir`

Added Phase 3.5 invalid tests:

- `compiler/test/ValueSurface/value-surface-scalable-vector-invalid.mlir`
- `compiler/test/ValueSurface/value-surface-unsupported-vector-element-invalid.mlir`
- `compiler/test/ValueSurface/value-surface-rank3-vector-invalid.mlir`
- `compiler/test/ValueSurface/value-surface-tensor-linalg-invalid.mlir`
- `compiler/test/ValueSurface/value-surface-native-f16-arith-invalid.mlir`

The staged-ops test uses locally robust syntax for `vector.gather`,
`vector.reduction`, `vector.contract`, `math.exp`, and `vector.shuffle`.

## 8. Non-Goals Preserved

Phase 3.5 preserves these non-goals:

- no value-to-VC4Kernel lowering;
- no memref ABI lowering;
- no kernel launch ABI lowering;
- no vector transfer lowering;
- no vector splitting implementation;
- no vector contract lowering;
- no TTIR/Triton import;
- no hardware fixtures or hardware execution;
- no `vc4value` scope creep.

`vc4value` remains the tiny two-op launch identity dialect:

```text
vc4value.program_id
vc4value.num_programs
```

## 9. Readiness Lines

```text
VC4_VALUE_SURFACE_ABSTRACTION_PHASE3_5_LOCKED=YES
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

## 10. Phase 4 Handoff

Phase 4 may lock the value kernel wrapper, launch ABI, memref ABI, and
address-space conventions on top of this corrected value-surface abstraction.

Phase 4 must not re-narrow the value surface to `vector<16xT>` as a global type
limit. It may define the first executable ABI and lowering subset around Phase
5 V1 `vector<16>` fragment carriers while preserving staged fixed-vector
surface admissibility.
