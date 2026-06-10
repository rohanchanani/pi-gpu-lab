PHASE9_RESULT=LOCKED
FEATURE=VALUE_AND_TTIR_MULTI_AXIS_LAUNCH_IDENTITY
VALUE_MULTI_AXIS_SURFACE_CONTRACT=LOCKED
VALUE_MULTI_AXIS_TO_VC4KERNEL_STATIC=PASS
VALUE_MULTI_AXIS_HARDWARE_ISOLATION=PASS
VALUE_MULTI_AXIS_MIXED_ACCEPTANCE=PASS
REAL_TRITON_MULTI_AXIS_SOURCES=YES
REAL_TTIR_MULTI_AXIS_SNAPSHOTS=YES
TTIR_MULTI_AXIS_IMPORTER_STATIC=PASS
TTIR_MULTI_AXIS_HARDWARE_ISOLATION=PASS
TTIR_MULTI_AXIS_MIXED_ACCEPTANCE=PASS
VALUE_MIXED_REGRESSION=PASS
TTIR_MIXED_REGRESSION=PASS
LAYERED_REGRESSION_POLICY_APPLIED=YES
FRONTEND_ROBUSTNESS_AUDIT=PASS
NO_STRINGLY_AXIS_CLASSIFICATION=YES
PHASE9_NO_TTIR_REGEN_IN_HARDWARE_PHASES=YES
NO_MASK_CLASSIFIER_SCOPE_CREEP=YES
NO_RANK2_MEMORY_SCOPE_CREEP=YES
READY_FOR_PHASE10_MASK_CLASSIFIER_AND_MEMORY_LEGALITY=YES
READY_FOR_TRITON=NO

# Phase 9 Multi-Axis Launch Identity Final Lock

Phase 9 is locked for value and TTIR multi-axis launch identity. The accepted
feature is logical launch-grid identity, not physical QPU identity:

```text
pid0 = logical_block_id % grid.x
pid1 = (logical_block_id / grid.x) % grid.y
pid2 = logical_block_id / (grid.x * grid.y)
num_programs(0/1/2) = grid.x/grid.y/grid.z
```

## Accepted Forms

Accepted value forms:

- `vc4value.grid_rank` 1, 2, and 3 for launch identity.
- `vc4value.program_id` axes 0, 1, and 2 when `axis < grid_rank`.
- `vc4value.num_programs` axes 0, 1, and 2 when `axis < grid_rank`.
- Flattened rank-1 global memory with `vector<16xT>` transfer reads/writes.
- Canonical linearized tail masks only.

Accepted TTIR forms:

- `tt.get_program_id x/y/z` lowering structurally to value axes 0/1/2.
- `tt.get_num_programs x/y/z` lowering structurally to value axes 0/1/2.
- `vc4value.grid_rank` inferred from the maximum accepted TTIR launch axis.
- Controlled 2D and 3D linearized tails:

```text
block_id = pid1 * num_programs0 + pid0
block_id = (pid2 * num_programs1 + pid1) * num_programs0 + pid0
idx = block_id * 16 + lane
mask = idx < n
```

Controlled real Triton sources and source-controlled TTIR snapshots are under
`examples/triton/phase9_multi_axis_launch/`.

## Staged Forms

Phase 9 intentionally does not accept or implement:

- generic mask classifier behavior;
- rank-2 memory or rank-2 transfer planning;
- gather, scatter, block pointers, or tensor descriptors;
- reductions or dot;
- `arith.sitofp` or `arith.fptosi`;
- TTG/backend dialects or Python semantic lowering.

Those forms remain Phase 10 or later work. Phase 10 is the mask classifier and
memory-legality phase.

## Hardware Summary

Value hardware isolation passed for 2D pid, 3D pid, num_programs axes, and
tail/control-flow interaction. Value mixed acceptance passed with
`mixed_value_multi_axis_cf_tail_vc4value`.

TTIR hardware isolation passed for the controlled real TTIR snapshots:

- `ttir_multi_axis_pid2d_b16_vc4triton`
- `ttir_multi_axis_pid3d_b16_vc4triton`
- `ttir_multi_axis_num_programs_b16_vc4triton`
- `ttir_multi_axis_cf_tail_b16_vc4triton`

TTIR mixed acceptance passed with `mixed_ttir_multi_axis_cf_tail_vc4triton`.
The full TTIR mixed suite also covered the existing Phase 7 and Phase 8.5 mixed
fixtures. Hardware result lines require `active_qpus=12`, `lanes=16`, zero
output mismatches, zero sentinel mismatches, zero launch failures, and nonzero
output hashes.

## Audits

Final audits passed for:

- value mixed coverage and claims;
- value Phase 9 isolation claims;
- TTIR mixed coverage and claims;
- TTIR Phase 9 isolation claims;
- frontend robustness;
- structural, non-stringly axis classification;
- value output boundary;
- no TTIR regeneration in hardware and mixed phases;
- no mask-classifier scope creep;
- no rank-2 memory scope creep.

`READY_FOR_TRITON=NO` remains locked. Phase 9 proves an incremental
source-visible feature through hardware; it does not open global Triton
readiness.
