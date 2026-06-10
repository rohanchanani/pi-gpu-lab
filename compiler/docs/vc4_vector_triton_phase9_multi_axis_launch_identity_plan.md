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
