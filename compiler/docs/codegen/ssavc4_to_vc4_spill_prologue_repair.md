# SSAVC4ToVC4 Spill Prologue Repair

This document records the lower-half repair for the Phase 8R control-flow
timeout investigation.

```text
ROOT_CAUSE=SPILL_ACTIONS_EMITTED_BEFORE_HIDDEN_SPILL_UNIFORMS
FIX=SPILL_RESOURCE_UNIFORMS_ARE_UNIFORM_PREFIX_PROLOGUE
SPILL_UNIFORM_PREFIX_PROLOGUE=YES
SPILL_ACTION_BEFORE_HIDDEN_SPILL_UNIFORMS=NO
HIDDEN_SPILL_RESOURCE_METADATA_EXACT=YES
HIDDEN_SPILL_RESOURCE_METADATA_MARKS_VPM_QPU_WRITE=YES
HIDDEN_SPILL_RESOURCE_METADATA_MARKS_VDW=YES
ORIGINAL_PHASE8R_CONTROL_FLOW_TIMEOUT_RERUN=PASS
TWO_QPU_N0_LAUNCH0_POISON_REPRO=PASS
REPEAT_INVOCATION_PROBES=PASS
SSAVC4_MULTI_EXIT_MERGE_BLOCK_ARG_SPILL_HARDWARE=PASS
FIXTURE_WEAKENING_REJECTED=YES
LAYERED_MIXED_REGRESSION=PASS
READY_TO_RESUME_PHASE8R_OR_PHASE85=YES
READY_FOR_TRITON=NO
```

## Root Cause

Spilled scheduled VC4 kernels appended hidden compiler spill uniforms after the
public uniform stream, then delayed reading those uniforms until the first
non-uniform-read scheduled template. The scheduler could still attach spill
pre-actions to early uniform-read templates, so generated code could execute
hidden VPM/VDW/VDR spill traffic before `spill_frame_base` and `spill_vpm_row`
had been read into their reserved registers.

That produced VDW/VDR/VPM activity using uninitialized spill resource registers.
The observed timeout was therefore a lower-half spill prologue bug, not an
infinite value-level CFG.

## Repair

Spilled kernels now use two `uniform_prefix` launch ABI entries:

- `spill_frame_base`, uniform index 0
- `spill_vpm_row`, uniform index 1

Public source uniforms and ordinary uniform-packed builtins shift by two only
for spilled kernels. No-spill kernels keep their previous ABI and uniform order.

`SSAVC4ToVC4` emits a real scheduled prologue for spilled kernels:

```text
mov ra28, unif  // spill_frame_base
mov ra27, unif  // spill_vpm_row
```

Those reads are emitted before iterating scheduled templates and before any
allocator spill pre-action can be emitted.

Hidden compiler spill metadata also marks the actual resource use:

- `uses_vpm = true`
- `uses_vpm_qpu_read = true`
- `uses_vpm_qpu_write = true`
- `uses_vdr = true`
- `uses_vdw = true`
- `requires_vpm_base_row_builtin = true` when spill VPM rows are required

The earlier multi-predecessor merge block-argument spill repair remains part of
the same lower-half correctness boundary: phi-like merge block arguments with
successor operands use reserved register homes under spill pressure.

## Permanent Coverage

Static coverage checks:

- spilled ABI prefix order, shifted public uniforms, and hidden spill resource
  metadata in SSAVC4 emit tests;
- no-spill ABI preservation in the minimal SSAVC4 emit test;
- scheduled prologue ordering and multi-predecessor merge block-argument spill
  lowering in `Conversion/SSAVC4ToVC4`.

Hardware coverage:

- the original unreduced Phase 8R mixed value control-flow fixture remains
  source-controlled and passes at `active_qpus=12`;
- the two-QPU `n=0` poison reproducer and repeated `n=1` probe passed during
  diagnosis without fixture weakening;
- `ssavc4_multi_exit_merge_block_arg_spill_vc4` is the permanent lower-half
  hardware fixture for the merge block-argument spill class and passes at
  `active_qpus=12`.

Layered mixed regression was run after the repair before this finalization
package. This finalization prompt intentionally did not rerun layered mixed
regression.
