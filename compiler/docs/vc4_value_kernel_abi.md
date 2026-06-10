# VC4 Value Kernel ABI

## 1. Purpose and layer boundary

Phase 4 defines the value-layer public ABI only. It locks the source-visible
kernel wrapper, launch identity, public argument metadata, and logical memref
argument contract that later handwritten lowering will consume.

Phase 4 does not lower value IR to VC4Kernel, does not lower memrefs, does not
emit artifacts, does not run hardware, and does not claim Triton support.

## 2. Accepted stack path

The only accepted path remains:

```text
standard value layer -> vc4kernel -> ssavc4 -> scheduled vc4 -> artifacts/runtime/hardware
```

There is no direct VC4KernelToVC4 path, no VC4Tile revival, and no Triton
direct-to-vc4kernel path. TTIR/Triton ingestion remains blocked until the
handwritten value path is proven.

## 3. Kernel wrapper

A public value kernel is a `func.func` operation marked with
`vc4value.kernel`.

Every public value kernel must have `vc4value.grid_rank` as an integer
attribute with value 1, 2, or 3. Grid rank is a value-layer launch contract.
Phase 5 V1 may lower only `grid_rank = 1` initially, but Phase 4 accepts the
rank 1/2/3 wrapper ABI.

Canonical wrapper shape:

```mlir
func.func @kernel(...) attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  func.return
}
```

## 4. Launch identity

`vc4value.program_id` and `vc4value.num_programs` are logical launch-grid
identity operations. They are not physical QPU identity, lane identity, warp
identity, or runtime resource metadata.

Every launch identity op must appear inside a `vc4value.kernel` function. Its
`axis` attribute must be an integer in the range:

```text
0 <= axis < enclosing vc4value.grid_rank
```

## 5. Public argument schema

Every public kernel argument must have `vc4value.arg_name` as a nonempty
`StringAttr`.

`vc4value.arg_name` values must:

- be unique within the function;
- match `^[A-Za-z_][A-Za-z0-9_]*$`;
- not use a reserved lower-half, runtime, builtin, or hardware-path name.

Every public memref argument must also have `vc4value.direction` as exactly one
of:

```text
in, out, inout
```

Public scalar arguments may have `vc4value.scalar_role` as one of:

```text
value, extent, stride, grid_dim, policy
```

`vc4value.scalar_role` is optional in Phase 4. Absence means `value` in the ABI
documentation, but any present unknown role is rejected.

Public memref arguments may have:

```text
vc4value.shape_args = ["name0", "name1", ...]
vc4value.stride_args = ["name0", "name1", ...]
```

Both attributes are ArrayAttr-of-StringAttr metadata. Every named entry must
refer to an existing scalar public kernel argument by `vc4value.arg_name`.

## 6. Public scalar ABI

Phase 4 accepts these public scalar argument types:

```text
index, i32, f32
```

`index` is value-level address, extent, stride, grid, or policy arithmetic.
Later lowering may convert `index` to i32 only where the target profile proves
that conversion legal.

Phase 4 rejects public scalar arguments with vector, tensor, tuple, none,
opaque/resource/target dialect, complex, i1, i8, i16, i64, f16, f64, bf16,
float8, or unsupported types.

## 7. Public memref ABI

Phase 4 accepts public memref arguments only when they are ranked rank-1 or
rank-2 memrefs in `#vc4value.global` memory space with one of these element
types:

```text
i8, i16, i32, f16, f32
```

The value-layer memref is a logical memory object. It carries shape, layout,
element type, memory space, direction, and explicit shape/stride metadata for
later planning. Phase 4e verifies this public argument contract statically; it
does not lower any memref.

Phase 5 V1 may lower only the first executable subset: i32/f32 rank-1
contiguous global memrefs for elementwise vector<16> kernels. i8, i16, f16,
rank-2, and dynamic-stride memrefs are ABI-admissible or metadata-admissible
but staged for later subword, f16-storage, or tile/VPM planning.

## 8. No hidden memref descriptor ABI

The Phase 4 public ABI is a bare-pointer logical-memref ABI. Future lowering
maps each public memref argument to a raw i32 device base pointer formal
argument in VC4Kernel.

Dynamic dimensions and dynamic strides are explicit scalar arguments, not
hidden memref descriptors. A memref type with dynamic dimensions must name the
corresponding extent scalar args through `vc4value.shape_args`. Explicit
strided layouts with dynamic strides must name stride scalar args through
`vc4value.stride_args`; identity-layout dynamic memrefs do not require hidden
stride metadata.

`memref.dim` is metadata-only at the value surface. It must lower later to
explicit scalar extent arguments or proven static dimensions. It does not
authorize a hidden descriptor, runtime descriptor load, or unmodeled ABI
packet. Phase 4e accepts `memref.dim` only on public `#vc4value.global` memref
arguments whose queried dynamic dimension has a `vc4value.shape_args` mapping.

## 9. Memory space #vc4value.global

`#vc4value.global` is a memory-space attribute only.

It does not add vc4value memory operations. It does not select TMU, VDR, VPM,
or VDW. It does not encode resource metadata, safe offsets, inactive-store
policy, VPM rows, barriers, or physical QPU identity.

It marks public buffers as VC4 global/device-addressable logical memory for
later value-to-VC4Kernel planning.

## 10. Lowerability classes

| Form | Phase 4 ABI status | Phase 5 V1 status | Later path |
| --- | --- | --- | --- |
| i32/f32 rank-1 contiguous memref | ABI-admissible | lowerable candidate | TMU/VDW elementwise |
| i8/i16 rank-1/rank-2 memref | ABI-admissible | not Phase 5 lowerable | subword/P9/P12 path |
| f16 memref | ABI-admissible storage only | not Phase 5 lowerable | f16 storage conversion + f32 compute |
| rank-2 memref | ABI-admissible | not Phase 5 lowerable | VPM/VDR/VDW tile planner |
| vector public arg | rejected | rejected | use vector values inside body, not ABI |
| tensor public arg | rejected | rejected | future producer layer only |

## 11. Reserved names

These `vc4value.arg_name` values are reserved:

```text
program_id
num_programs
lane_id
qpu_id
physical_qpu_id
warp_id
thread_id
num_qpus
vpm_base_row
semaphore_base
uniform_index
uniform_offset
tmu
vdr
vdw
vpm
ssavc4
vc4kernel
vc4
```

Later audits may add lower-half runtime names as the launch ABI expands. Public
value ABI names must not collide with names that are owned by runtime,
lower-half metadata, physical identity, or hardware path selection.

## 12. Canonical examples

Rank-1 f32 saxpy-style ABI:

```mlir
func.func @saxpy(
    %n: index {vc4value.arg_name = "n", vc4value.scalar_role = "extent"},
    %a: f32 {vc4value.arg_name = "a"},
    %x: memref<?xf32, #vc4value.global> {vc4value.arg_name = "x",
                                         vc4value.direction = "in",
                                         vc4value.shape_args = ["n"]},
    %y: memref<?xf32, #vc4value.global> {vc4value.arg_name = "y",
                                         vc4value.direction = "inout",
                                         vc4value.shape_args = ["n"]})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %pid = vc4value.program_id {axis = 0 : i32} : index
  %np = vc4value.num_programs {axis = 0 : i32} : index
  func.return
}
```

Grid-rank 3 launch identity ABI:

```mlir
func.func @grid3(%n: index {vc4value.arg_name = "n", vc4value.scalar_role = "extent"})
    attributes {vc4value.kernel, vc4value.grid_rank = 3 : i32} {
  %x = vc4value.program_id {axis = 0 : i32} : index
  %y = vc4value.program_id {axis = 1 : i32} : index
  %z = vc4value.program_id {axis = 2 : i32} : index
  %nx = vc4value.num_programs {axis = 0 : i32} : index
  %ny = vc4value.num_programs {axis = 1 : i32} : index
  %nz = vc4value.num_programs {axis = 2 : i32} : index
  func.return
}
```

Rank-1/rank-2 memref ABI surface examples:

```mlir
func.func @buffers(
    %n: index {vc4value.arg_name = "n", vc4value.scalar_role = "extent"},
    %m: index {vc4value.arg_name = "m", vc4value.scalar_role = "extent"},
    %a: memref<?xi8, #vc4value.global> {vc4value.arg_name = "a",
                                        vc4value.direction = "in",
                                        vc4value.shape_args = ["n"]},
    %b: memref<?xi16, #vc4value.global> {vc4value.arg_name = "b",
                                         vc4value.direction = "out",
                                         vc4value.shape_args = ["n"]},
    %c: memref<?xi32, #vc4value.global> {vc4value.arg_name = "c",
                                         vc4value.direction = "inout",
                                         vc4value.shape_args = ["n"]},
    %h: memref<?xf16, #vc4value.global> {vc4value.arg_name = "h",
                                         vc4value.direction = "inout",
                                         vc4value.shape_args = ["n"]},
    %t: memref<?x?xf32, #vc4value.global> {vc4value.arg_name = "t",
                                           vc4value.direction = "inout",
                                           vc4value.shape_args = ["m", "n"]})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  func.return
}
```

f16 storage note:

```text
memref<?xf16, #vc4value.global> is ABI-admissible storage. Native f16
arithmetic remains rejected; later lowering must use f16 storage conversion
plus f32 compute where legal.
```

memref.dim metadata-only example:

```mlir
func.func @dims(
    %n: index {vc4value.arg_name = "n", vc4value.scalar_role = "extent"},
    %x: memref<?xf32, #vc4value.global> {vc4value.arg_name = "x",
                                         vc4value.direction = "in",
                                         vc4value.shape_args = ["n"]})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %c0 = arith.constant 0 : index
  %dim = memref.dim %x, %c0 : memref<?xf32, #vc4value.global>
  func.return
}
```

These snippets use the Phase 4 syntax. `#vc4value.global` is locked by this
contract and implemented as a VC4Value dialect memory-space attribute.

## 13. Rejected examples

Phase 4 rejects:

- a public kernel argument without `vc4value.arg_name`;
- duplicate `vc4value.arg_name` values in one public kernel;
- malformed or reserved `vc4value.arg_name` values;
- a memref public argument without `vc4value.direction`;
- a memref public argument whose direction is not `in`, `out`, or `inout`;
- a public memref without `#vc4value.global`;
- a public memref that assumes a hidden descriptor for shape or stride;
- a vector public argument such as `vector<16xf32>`;
- a tensor public argument such as `tensor<?xf32>`;
- a public memref with f64 element type;
- a public memref with bf16 or float8 element type;
- direct TMU/VDR/VDW/VPM concepts in the value ABI.

## 14. Phase 5 handoff

The Phase 5 V1 lowerable subset is limited to vector<16> i32/f32 elementwise
rank-1 contiguous kernels first. Phase 4 makes the public ABI ready for that
future package, but Phase 4 does not implement lowering and valid value ABI IR
is not assumed to be hardware-executable.

## 15. Phase 11 strided/ranked memory ABI delta

PHASE11_VALUE_STRIDED_RANKED_MEMORY_CONTRACT=LOCKED
RANK2_ROW_SLICE_IDENTITY_SURFACE=ACCEPTED
RANK2_ROW_SLICE_STRIDED_OUTER_DYNAMIC_SURFACE=ACCEPTED
MEMREF_DIM_METADATA_TO_SCALAR_ARG_CONTRACT=LOCKED
HIDDEN_MEMREF_DESCRIPTOR_ALLOWED=NO
GATHER_LANE_STRIDE_STAGED=YES
READY_FOR_PHASE11_4_VALUE_RANKED_STRIDED_STATIC=YES
READY_FOR_TRITON=NO

Phase 11 narrows the rank-2 executable-planning contract without changing the
bare-pointer public ABI. Accepted Phase 11 value forms are:

- `RANK1_FLATTENED_SCALAR_STRIDED_ADDRESS`: public
  `memref<?xT, #vc4value.global>` with `T = i32/f32`; scalar index arithmetic
  may compute `row * stride + col_block * 16`, but the transfer itself remains
  rank-1 identity and contiguous along vector lanes.
- `RANK2_ROW_SLICE_IDENTITY`: public
  `memref<?x?xT, #vc4value.global>` with `T = i32/f32`; `shape_args` names both
  dynamic dimensions, row-slice transfers use scalar indices `[row, col]`, and
  `vector<16xT>` maps to the innermost dimension.
- `RANK2_ROW_SLICE_STRIDED_OUTER_DYNAMIC`: public
  `memref<?x?xT, strided<[?, 1], offset: 0>, #vc4value.global>` with
  `T = i32/f32`; `shape_args` names dynamic dimensions and `stride_args` names
  the dynamic outer row stride. The inner stride is statically 1 and the offset
  is statically 0.
- `MEMREF_DIM_METADATA_LOWERING`: `memref.dim` on public `#vc4value.global`
  dynamic dimensions maps to explicit scalar extent arguments. It does not
  authorize hidden descriptor loads.

Non-unit inner stride, lane-varying stride/gather, column/vertical slices,
rank greater than 2, vector rank greater than 1 for a transfer result, f16/i8/i16
storage lowering, block pointers, boundary checks, padding options, hidden
descriptors, and sparse/unknown transfer masks outside the Phase 10 accepted
mask set remain staged.

## 16. Readiness lines

The intended final Phase 4 lock lines are:

```text
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
```
