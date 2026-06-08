# VC4 Vector/Triton Phase 4 - Value ABI Lock

## 1. Result

VC4_VALUE_ABI_PHASE4_LOCKED=YES
VC4_VALUE_ABI_PHASE4_POST35_COMPATIBLE=YES
VC4_VALUE_KERNEL_WRAPPER_ABI_LOCKED=YES
VC4_VALUE_GLOBAL_MEMREF_ABI_LOCKED=YES
VC4_VALUE_ARG_ATTRS_ABI_LOCKED=YES
VC4_VALUE_NO_MEMREF_DESCRIPTOR_ABI=YES
VC4_VALUE_ABI_MEMREF_ELEMENT_TYPES=i8,i16,i32,f16,f32
VC4_VALUE_ABI_RANK1_RANK2_MEMREFS_ADMISSIBLE=YES
VC4_VALUE_ABI_VECTOR_ARGS_REJECTED=YES
VC4_VALUE_ABI_TENSOR_ARGS_REJECTED=YES
VC4_VALUE_ABI_MEMREF_DIM_IS_METADATA_ONLY=YES
READY_FOR_PHASE5_HANDWRITTEN_VALUE_ELEMENTWISE_LOWERING=YES
READY_FOR_VALUE_TO_VC4KERNEL_LOWERING=YES_FOR_PHASE5_ELEMENTWISE_V1
READY_FOR_TRITON=NO

Phase 4 locks the source-visible value ABI only. It does not implement
value-to-VC4Kernel lowering, executable memory lowering, candidate generation,
runtime changes, or hardware execution.

## 2. Final stack boundary

The only accepted stack path remains:

```text
standard value layer -> vc4kernel -> ssavc4 -> scheduled vc4 -> artifacts/runtime/hardware
```

Phase 4 does not add direct VC4KernelToVC4. VC4Tile remains retired. Triton and
TTIR must not lower directly to VC4Kernel; future producer inputs must first
canonicalize into the locked standard value surface.

## 3. Phase 3.5 compatibility

Phase 4 preserves the Phase 3.5 value abstraction. `vector<16xT>` is the Phase
5 V1 lowerable fragment shape, not the whole value-layer type limit. Fixed
non-scalable rank-1 vectors with widths other than 16 remain surface-admissible
and staged for later splitting. Fixed rank-2 vectors remain surface-admissible
and staged for later tile/contract planning. Scalable vectors and tensor/linalg
remain initial value-surface rejects.

## 4. Kernel wrapper ABI

A public value kernel is a `func.func` marked with `vc4value.kernel`. It must
also carry `vc4value.grid_rank` as an integer attribute with value 1, 2, or 3.
The wrapper is a value-layer launch contract only. Phase 5 V1 may initially
lower only `grid_rank = 1`.

## 5. Launch identity ABI

`vc4value.program_id` and `vc4value.num_programs` are logical launch-grid
identity operations. They are legal only inside a public value kernel. Each
used axis must be within the enclosing kernel's `vc4value.grid_rank`; negative
axes and axes greater than or equal to `grid_rank` reject. These operations do
not expose physical QPU, warp, thread, or lane identity.

## 6. Public argument ABI

Every public kernel argument must have `vc4value.arg_name` as a nonempty string
matching `^[A-Za-z_][A-Za-z0-9_]*$`. Names must be unique and must not use
reserved lower-half/runtime names such as `program_id`, `lane_id`, `qpu_id`,
`tmu`, `vdr`, `vdw`, `vpm`, `ssavc4`, `vc4kernel`, or `vc4`.

Public memref arguments must have `vc4value.direction` equal to `in`, `out`, or
`inout`. Scalar arguments may carry `vc4value.scalar_role` equal to `value`,
`extent`, `stride`, `grid_dim`, or `policy`; absence means `value` for the ABI
contract. Dynamic memref extents are named by `vc4value.shape_args`.
Explicit dynamic strides in strided layouts are named by
`vc4value.stride_args`. Both metadata arrays are ArrayAttr-of-StringAttr and
must reference existing scalar arguments where the verifier can cross-check
them.

## 7. Memref ABI

`#vc4value.global` is a VC4Value memory-space attribute for public value ABI
memrefs. It is metadata only: it does not add vc4value memory ops, does not
select TMU/VDR/VPM/VDW, and does not encode resource metadata, safe offsets,
inactive-store policy, VPM rows, barriers, or physical QPU identity.

Phase 4 accepts only ranked public memrefs of rank 1 or rank 2 in
`#vc4value.global` with element types `i8`, `i16`, `i32`, `f16`, or `f32`.
Dynamic dimensions and dynamic strides are explicit scalar arguments, not
hidden memref descriptors. Future lowering maps each public memref argument to
a raw i32 device base pointer formal argument in VC4Kernel plus explicit scalar
metadata where needed.

## 8. Lowerability classification

| Form | Phase 4 ABI status | Phase 5 V1 status | Later path |
| --- | --- | --- | --- |
| i32/f32 rank-1 contiguous elementwise memref subset | ABI-admissible | lowerable candidate | handwritten Phase 5 V1 TMU/VDW elementwise path |
| i8/i16 memrefs | ABI-admissible staged | not Phase 5 V1 lowerable | later subword/P9/P12 path |
| f16 memrefs | ABI-admissible storage only | not Phase 5 V1 lowerable | f16 storage conversion plus f32 compute |
| rank-2 memrefs | ABI-admissible staged | not Phase 5 V1 lowerable | later VPM/VDR/VDW tile planner |
| vector public arguments | rejected | rejected | use vector values inside the body, not ABI args |
| tensor public arguments | rejected | rejected | future producer layer only |
| hidden memref descriptors | rejected | rejected | explicit scalar metadata only |
| unsupported memref elements | rejected | rejected | future proof and contract required |

Phase 4 readiness for value-to-VC4Kernel lowering is scoped only to the future
Phase 5 elementwise V1 package. It is not a claim that lowering exists now.

## 9. f16 storage policy

`f16` buffers are storage-admissible in the public ABI. Native f16 arithmetic
remains rejected. Future executable lowering must use f16 storage conversion
plus f32 compute where that path is legal and proven.

## 10. memref.dim and metadata-only view policy

`memref.dim` is metadata-only at the value surface. It is admissible only when
the queried public `#vc4value.global` memref dimension is static or has a
`vc4value.shape_args` mapping. It must later resolve to explicit scalar extent
arguments or static dimensions. It does not authorize a hidden descriptor,
runtime descriptor load, or unmodeled ABI packet.

Direct executable memory side-effect operations such as `memref.load`,
`memref.store`, atomics, copies, and DMA remain rejected in Phase 4.

## 11. Stride metadata status

STRIDE_ARGS_CROSSCHECK=enforce_now

Phase 4 verifies `vc4value.stride_args` as ArrayAttr-of-StringAttr and
cross-checks explicit dynamic strides in strided memref layouts against
existing `index` or `i32` scalar arguments with compatible scalar roles. Identity
dynamic memrefs do not imply hidden stride descriptors.

## 12. Tests and audits

Final static acceptance includes:

- `ninja -C compiler/build vc4-opt vc4-codegen`
- `run_if_nonempty_lit compiler/test/Dialect/VC4Value`
- `run_if_nonempty_lit compiler/test/ValueSurface`
- `python3 compiler/test/ValueSurface/Support/check_vc4_value_surface_matrix.py compiler/docs/vc4_value_surface_support_matrix.json --mode phase4-abi-lock`
- `python3 compiler/test/ValueSurface/Support/audit_vc4_value_surface.py --repo-root . --matrix compiler/docs/vc4_value_surface_support_matrix.json --mode phase4-abi-lock`
- `python3 compiler/test/Dialect/VC4Value/Support/audit_vc4value_tiny_surface.py --repo-root . --matrix compiler/docs/vc4value_support_matrix.json --mode phase2-lock`
- `python3 compiler/test/Dialect/VC4Kernel/Support/check_vc4kernel_surface_v2_matrix.py compiler/docs/vc4kernel_surface_v2_support_matrix.json --mode final`
- `python3 compiler/test/Dialect/VC4Kernel/Support/audit_vc4kernel_surface_v2.py --repo-root . --matrix compiler/docs/vc4kernel_surface_v2_support_matrix.json --mode p13-final-surface-lock --phase-lock P13`
- `python3 compiler/test/CodeGen/VC4Kernel/Hardware/MixedAcceptance/check_mixed_acceptance_coverage.py --manifest compiler/test/CodeGen/VC4Kernel/Hardware/MixedAcceptance/mixed_acceptance_manifest.json --repo-root . --mode lock`
- `python3 compiler/test/CodeGen/VC4Kernel/Hardware/MixedAcceptance/audit_mixed_fixture_claims.py --claims compiler/test/CodeGen/VC4Kernel/Hardware/MixedAcceptance/mixed_fixture_claims.json --manifest compiler/test/CodeGen/VC4Kernel/Hardware/MixedAcceptance/mixed_acceptance_manifest.json --repo-root .`
- `ninja -C compiler/build check-vc4`
- `git diff --check`

The Phase 4 ABI corpus includes public scalar args, rank-1/rank-2 global memref
element coverage, static memrefs, dynamic shape metadata, dynamic stride
metadata, metadata-only `memref.dim`, launch axes, public vector/tensor argument
rejects, unsupported memref element rejects, and direct memref side-effect
rejects.

## 13. Hardware status

Phase 4 ran no hardware. It adds static verifier, documentation, matrix, audit,
and lit coverage only. Because Phase 4 adds no executable value-to-VC4Kernel
semantics, hardware proof begins in a later executable lowering phase.

## 14. Phase 5 handoff

Phase 5 may implement only the first handwritten value elementwise V1 lowering:

- `grid_rank = 1`;
- `vc4value.program_id` and `vc4value.num_programs` axis 0;
- `vector<16xi32>` and `vector<16xf32>` arithmetic, compare, and select;
- `vector<16xi1>` tail masks;
- `vector.step`;
- `vector.transfer_read` and `vector.transfer_write` for rank-1 i32/f32
  contiguous `#vc4value.global` memrefs;
- TMU safe-offset inactive-zero loads;
- VDW inactive-preserve stores.

Phase 5 V1 must not implement non-16 vector splitting, i8/i16/f16 lowering,
rank-2/VPM tile lowering, Triton import, or general value-to-VC4Kernel lowering
outside that scoped elementwise subset.

## 15. Caveats

Phase 4 is static-only and no valid value ABI IR is assumed
hardware-executable. `READY_FOR_VALUE_TO_VC4KERNEL_LOWERING` is scoped to
`YES_FOR_PHASE5_ELEMENTWISE_V1`; it is not a general lowering-ready claim.
Triton remains blocked until handwritten value lowering is proven.
