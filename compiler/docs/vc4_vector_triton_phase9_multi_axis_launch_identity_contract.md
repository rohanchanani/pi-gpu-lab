PHASE9_FEATURE=VALUE_AND_TTIR_MULTI_AXIS_LAUNCH_IDENTITY
VALUE_MULTI_AXIS_SURFACE_CONTRACT=LOCKED
VC4VALUE_GRID_RANK_1_2_3_EXECUTABLE_FOR_LAUNCH_IDENTITY=YES
VC4VALUE_PROGRAM_ID_AXES_0_1_2_LOWERABLE_NOW=YES
VC4VALUE_NUM_PROGRAMS_AXES_0_1_2_LOWERABLE_NOW=YES
PHASE9_MEMORY_MODEL=FLATTENED_RANK1_ONLY
PHASE9_MASK_MODEL=CANONICAL_LINEARIZED_TAIL_ONLY
MASK_CLASSIFIER_SCOPE_CREEP=NO
RANK2_MEMORY_SCOPE_CREEP=NO
VALUE_MULTI_AXIS_TO_VC4KERNEL_STATIC=PASS
VC4VALUE_GRID_RANK_2_3_LOWERING=YES
VC4VALUE_PROGRAM_ID_AXES_0_1_2_TO_VC4KERNEL=YES
VC4VALUE_NUM_PROGRAMS_AXES_0_1_2_TO_VC4KERNEL=YES
GRID_XYZ_REQUEST_INFO_STATIC_AUDIT=PASS
NO_MASK_CLASSIFIER_SCOPE_CREEP=YES
NO_RANK2_MEMORY_SCOPE_CREEP=YES
READY_FOR_PHASE9_5_VALUE_HARDWARE_ISOLATION=YES
READY_FOR_TRITON=NO

# Phase 9 Multi-Axis Launch Identity Contract

Phase 9 locks the value-surface contract delta for multi-axis logical launch
identity before value-to-VC4Kernel implementation work.

`vc4value.grid_rank` values 1, 2, and 3 are executable launch-identity ranks for
this feature. `vc4value.program_id` axes 0, 1, and 2 are Phase 9 lowerable-now
targets when the enclosing kernel rank admits the axis. `vc4value.num_programs`
axes 0, 1, and 2 are the matching lowerable-now targets.

These operations expose logical launch-grid identity only. They are not
physical QPU identity, warp identity, thread identity, lane identity, or a
scheduler placement promise.

## Logical Mapping

The runtime launch grid has logical dimensions:

```text
grid.x = num_programs(0)
grid.y = num_programs(1)
grid.z = num_programs(2)
```

For a flattened logical block id, the logical program ids are:

```text
pid0 = logical_block_id % grid.x
pid1 = (logical_block_id / grid.x) % grid.y
pid2 = logical_block_id / (grid.x * grid.y)
```

`vc4value.num_programs(axis)` returns `grid.x`, `grid.y`, or `grid.z` for axes
0, 1, and 2 respectively. For `grid_rank = 1`, only axis 0 is legal. For
`grid_rank = 2`, axes 0 and 1 are legal. For `grid_rank = 3`, axes 0, 1, and 2
are legal.

## Phase 9 Memory And Mask Boundary

Phase 9 memory remains flattened rank-1 only:

```text
PHASE9_MEMORY_MODEL=FLATTENED_RANK1_ONLY
```

Multi-axis launch identity may compute a flattened block id and then a rank-1
linear element index. It must not introduce rank-2 memory, rank-2 transfer
planning, gather/scatter, block pointers, tensor descriptors, or VPM tile
planning.

The only accepted dynamic mask model for this phase is the canonical linearized
tail:

```text
idx = flattened_block_id * 16 + lane
mask = idx < n
PHASE9_MASK_MODEL=CANONICAL_LINEARIZED_TAIL_ONLY
```

Phase 9 does not implement a general mask classifier. Sparse masks, rect masks,
unknown dynamic masks, rank-2 masks, vector branch conditions, and
memory-legality expansion remain outside this feature.

## Implementation Handoff

This contract allows later Phase 9 implementation prompts to lower axes 0, 1,
and 2 through the existing required stack:

```text
VC4Value -> vc4kernel -> ssavc4 -> scheduled vc4
```

It does not authorize direct lowering from value IR or TTIR to scheduled `vc4`,
does not alter the `vc4value` dialect surface, does not implement mask
classifier behavior, and does not mark global Triton readiness.

## Phase 9.4 Static Lowering Lock

Phase 9.4 implements the value executable launch-identity delta:

```text
VALUE_MULTI_AXIS_TO_VC4KERNEL_STATIC=PASS
VC4VALUE_GRID_RANK_2_3_LOWERING=YES
VC4VALUE_PROGRAM_ID_AXES_0_1_2_TO_VC4KERNEL=YES
VC4VALUE_NUM_PROGRAMS_AXES_0_1_2_TO_VC4KERNEL=YES
GRID_XYZ_REQUEST_INFO_STATIC_AUDIT=PASS
NO_MASK_CLASSIFIER_SCOPE_CREEP=YES
NO_RANK2_MEMORY_SCOPE_CREEP=YES
READY_FOR_PHASE9_5_VALUE_HARDWARE_ISOLATION=YES
READY_FOR_TRITON=NO
```

`vc4value.program_id` and `vc4value.num_programs` now lower structurally to the
matching `vc4kernel.program_id` and `vc4kernel.num_programs` axis attributes for
axes 0, 1, and 2 when the axis is within the enclosing `vc4value.grid_rank`.
Axis 0 behavior remains compatible with the earlier single-axis path.

The launch identity remains logical grid identity. The generated launch ABI
passes `vc4_dim3 grid` through to `vc4LaunchKernel`, and generated uniform
packers consume the runtime request-info fields for `program_id_x/y/z` and
`num_programs_x/y/z`. The runtime request-info contract is the logical mapping
defined above, not physical QPU identity.

This static lock does not broaden memory or mask legality. The value-to-VC4Kernel
path still accepts only flattened rank-1 global memrefs and the canonical
linearized `vector.create_mask` tail form; rank-2 memory and noncanonical masks
remain staged for later phases.
