ROADMAP_PHASE9_MULTI_AXIS_ALIGNMENT=YES
PHASE9_FEATURE=VALUE_AND_TTIR_MULTI_AXIS_LAUNCH_IDENTITY
PHASE10_FEATURE=MASK_CLASSIFIER_AND_MEMORY_LEGALITY
PREVIOUS_PHASE9_MASK_DOCS_SUPERSEDED_BY_FEATURE_LADDER=YES
READY_FOR_TRITON=NO

# Phase 9 Multi-Axis Launch Identity Plan

Phase 9 is the value and TTIR multi-axis logical launch identity feature. It
starts only after the Phase 8.5 TTIR control-flow final lock and the top-half
robustness lock are both present.

## Prerequisites

Phase 8.5 locked TTIR scalar/coherent control flow for the accepted subset. The
top-half robustness lock hardened the importer and value-output boundary so the
next feature can build on structural semantics rather than printed IR, names,
paths, or fixture-specific behavior.

## Feature Scope

Phase 9 aligns and then implements multi-axis logical launch identity:

- value-layer `vc4value.program_id` and `vc4value.num_programs` use across
  accepted 2D and 3D logical grid ranks;
- real TTIR `tt.get_program_id` and `tt.get_num_programs` axis handling for the
  controlled source-visible forms selected by the phase;
- value-to-VC4Kernel lowering that preserves logical launch identity through
  the existing lower-half path;
- fixture and audit coverage proving that multi-axis launch identity is not
  derived from source names or fixture paths.

The feature exists because GEMV, GEMM, and attention-style kernels need row,
column, batch, or block coordinates before the compiler can naturally classify
their masks and memory legality.

## Non-Goals

Phase 9 must not implement mask classifier behavior, rank-2 memory planning,
gather/scatter legality, strided transfer expansion, block pointers, tensor
descriptors, reductions, dot, or GEMV/GEMM lowering.

Mask classifier and memory legality are Phase 10:

```text
PHASE10_FEATURE=MASK_CLASSIFIER_AND_MEMORY_LEGALITY
```

## Acceptance Discipline

Controlled fixtures define acceptance. Exploratory probes may inform the
inventory, but they are not acceptance targets unless promoted into checked-in
fixtures with exact source, emitted TTIR, manifest metadata, expected staged or
lowered behavior, and audits.

Executable support must continue to lower through:

```text
VC4Value -> vc4kernel -> ssavc4 -> scheduled vc4
```

No direct lowering from TTIR or value IR to scheduled `vc4` is allowed, and
`READY_FOR_TRITON=NO` remains true.

## Controlled Fixture Designs

Phase 9 controlled TTIR fixtures live under:

```text
examples/triton/phase9_multi_axis_launch/sources/
examples/triton/phase9_multi_axis_launch/generated/
examples/triton/phase9_multi_axis_launch/manifest.json
```

They are scoped to multi-axis logical launch identity plus already accepted
elementwise, canonical tail, linearized load/store, and Phase 8.5
scalar/coherent control-flow forms. They must not introduce mask classifier,
rank-2 memory, gather/scatter, block pointer, dot, reduction, SFU, or numeric
cast feature work.

The corresponding value fixture designs for later prompts are:

- `value_multi_axis_pid2d_i32_vc4value`: a `vc4value.kernel` function with
  `grid_rank = 2`, `vc4value.program_id` axes 0 and 1, axis-0
  `vc4value.num_programs`, flattened block id, `vector.step`, canonical
  `idx < n` tail mask, and i32 stores encoding pid0, pid1, and lane.
- `value_multi_axis_pid3d_i32_vc4value`: a `grid_rank = 3` value kernel using
  program-id axes 0, 1, and 2 plus num-programs axes 0 and 1 to form
  `(pid2 * num_programs(1) + pid1) * num_programs(0) + pid0`, then storing
  an i32 lane encoding through a canonical linearized tail.
- `value_multi_axis_num_programs_axes_vc4value`: a `grid_rank = 3` value
  kernel that reads `vc4value.num_programs` axes 0, 1, and 2 and stores
  grid-dimension identity values through the same rank-1 linearized memory
  contract.
- `value_multi_axis_tail_cf_vc4value`: a `grid_rank = 2` value kernel combining
  multi-axis launch identity with already supported scalar coherent
  control-flow and a canonical tail-masked rank-1 transfer.
- `mixed_value_multi_axis_cf_tail_vc4value`: a cumulative value mixed fixture
  combining elementwise arithmetic, canonical tail masks, rank-1 load/store,
  scalar coherent control-flow, and axes 1/2 launch identity.

These designs are source-facing contracts only in Phase 9.2. They are not
implemented until later Phase 9 prompts.
