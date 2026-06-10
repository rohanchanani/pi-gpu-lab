# VC4 Vector/Triton Phase 8 Control-Flow Lock

PHASE8_RESULT=LOCKED
PHASE8_CONTROL_FLOW_BOUNDARY=LOCKED
UPSTREAM_SCF_TO_CF_BOUNDARY=YES
RAW_SCF_NOT_IN_VC4KERNEL=YES
PHASE8_STATIC_CF_LOCKED=YES
VALUE_CF_CONTROL_FLOW_ACCEPTED=YES
VALUE_CF_BLOCK_ARGUMENTS_ACCEPTED=YES
VALUE_CF_CONTROL_FLOW_STATIC=PASS
VALUE_CF_BLOCK_ARGUMENTS_STATIC=PASS
VALUE_SCF_TO_CF_BOUNDARY_LOCKED=YES
TTIR_CONTROL_FLOW_STAGED=YES
TTIR_CPP_IMPORTER_STATIC_REGRESSION=PASS
NO_TEMPORARY_VALUE_CF_WORKAROUNDS=YES
NO_TEMPORARY_FRONTEND_OR_VALUE_WORKAROUNDS=YES
READY_FOR_PHASE8F_HARDWARE_ISOLATION=YES
PHASE8_VALUE_CF_ISOLATION_HARDWARE=PASS
VALUE_CF_BRANCH_GUARD_HARDWARE=PASS
VALUE_CF_IF_ELSE_MERGE_HARDWARE=PASS
VALUE_CF_SCALAR_LOOP_HARDWARE=PASS
VALUE_CF_VECTOR_CARRY_LOOP_HARDWARE=PASS
VALUE_CF_LOOP_BOUNDARY_HARDWARE=PASS
READY_FOR_PHASE8G_MIXED_ACCEPTANCE=YES
PHASE8_VALUE_CF_MIXED_ACCEPTANCE=PASS
VALUE_CF_MIXED_ACCEPTANCE=PASS
TTIR_CPP_IMPORTER_REGRESSION=PASS
VALUE_MIXED_REGRESSION=PASS
VC4KERNEL_MIXED_REGRESSION=PASS
PHASE8_MIXED_CLAIM_AUDIT=PASS
READY_FOR_PHASE8H_FINAL_LOCK=YES
TTIR_CONTROL_FLOW_IMPORT=NO
ACTIVE_NEXT_STEP=PHASE8_5_TTIR_CONTROL_FLOW_BRIDGE
READY_FOR_PHASE8_5_TTIR_CONTROL_FLOW_BRIDGE=YES
READY_FOR_PHASE8R_CONTROL_FLOW_COMPLETENESS_REPAIR=YES
PHASE8R_CF_COMPLETENESS_SCOPE=ACTIVE
SCF_WHILE_VALUE_TARGET=SUPPORT_NOW
NESTED_STRUCTURED_CF_VALUE_TARGET=SUPPORT_NOW
TL_RANGE_STYLE_LOOP_SKELETON_VALUE_TARGET=SUPPORT_NOW
PERSISTENT_LOOP_SKELETON_VALUE_TARGET=SUPPORT_NOW
VECTOR_BRANCH_CONDITION_POLICY=DETERMINISTIC_REJECT_AS_CFG_USE_MASKS
IRREDUCIBLE_CFG_POLICY=PROBE_NOT_REQUIRED_FOR_SANE_TRITON
PHASE8R_SCF_WHILE_STATIC=PASS
PHASE8R_NESTED_STRUCTURED_CF_STATIC=PASS
PHASE8R_TL_RANGE_STYLE_LOOP_STATIC=PASS
PHASE8R_PERSISTENT_LOOP_SKELETON_STATIC=PASS
READY_FOR_PHASE8RD_SWITCH_MULTI_EXIT_PROBE=YES
PHASE8R_CF_SWITCH_INDEX_SWITCH_POLICY=LOCKED
PHASE8R_MULTI_EXIT_REDUCIBLE_LOOP_POLICY=LOCKED
PHASE8R_IRREDUCIBLE_CFG_POLICY=PROBED_NOT_REQUIRED_FOR_TRITON
READY_FOR_PHASE8RE_LOWER_HALF_CF_COMPLETION=NOT_NEEDED
READY_FOR_PHASE8RF_HARDWARE_ISOLATION=YES
READY_FOR_PHASE9_MASK_CLASSIFIER_AND_RICHER_MEMORY_LEGALITY=NO_PENDING_PHASE8_5_TTIR_CONTROL_FLOW_BRIDGE
READY_FOR_TRITON=NO

## Boundary

Phase 8 is a value-layer control-flow phase. It does not add TTIR
control-flow import and does not add new math, memory, reduction, dot, or
Triton features.

Phase 8R makes the scalar/coherent control-flow completeness target explicit.
QPU control flow is scalar/coherent across SIMD lanes: a QPU program has one
program counter for its 16 lanes. Per-lane control divergence is not accepted as branch CFG and must be represented with masks/selects.
Vector branch conditions are rejected as CFG.

`scf` is a value-surface input convenience only. Executable lowering to
VC4Kernel consumes `cf`, not raw `scf`. The required structured-control
boundary is the upstream MLIR pass:

```text
--convert-scf-to-cf
```

Custom SCF lowering is not part of Phase 8. If upstream SCF-to-CF ever becomes
unavailable, the phase must stop with a named blocker instead of inventing a
local lowering.

Phase 8R target forms are `scf.if`, `scf.for`, `scf.while`, and nested
structured control flow, all through upstream scf-to-cf before verified
VC4Kernel. scf.while is in scope now through upstream scf-to-cf. The
tl.range-style loop skeleton and persistent-loop skeleton are value-level
targets before TTIR import; unsupported loop body features remain staged by
body feature, not by control flow.

`scf.parallel`, `scf.forall`, and `scf.reduce` are not simple scalar
control-flow support. They remain staged/rejected for parallel and reduction
phases. Phase 8Rd locks `cf.switch` support by expanding scalar switches to a
deterministic `cf.cond_br` chain before verified VC4Kernel reaches the lower
half. `scf.index_switch` is supported when upstream `--convert-scf-to-cf`
canonicalizes it to `cf.switch`; raw `scf.index_switch` remains staged at the
value-surface boundary. Irreducible CFG is probe/classify only, not required
for sane Triton Phase 8.5 support.

## Executable Pipeline

For SCF-bearing value input:

```text
value input with scf
  -> --convert-scf-to-cf
  -> --vc4-verify-value-surface
  -> --convert-vc4-value-to-vc4kernel
  -> --verify-vc4kernel
```

Pure `cf` value input may start at `--vc4-verify-value-surface`.

Verified VC4Kernel must contain no raw `scf`, `vector`, `memref`, `func`,
`vc4value`, TTIR, `ssavc4`, or scheduled `vc4` producer operations. It may
contain accepted VC4Kernel operations, accepted scalar `arith`, and standard
`cf.br`/scalar-i1 `cf.cond_br`.

Phase 8R wording: verified VC4Kernel contains no raw scf or producer dialects.

## Phase 8Rc Static SCF While And Loop Skeletons

Phase 8Rc proved that upstream `--convert-scf-to-cf` lowers `scf.while`,
nested structured control flow, tl.range-style loop skeletons, and
persistent-loop skeletons into explicit `cf.br`/`cf.cond_br` CFG with block
arguments. The existing value-to-VC4Kernel path then lowers those shapes to
verified VC4Kernel, SSAVC4, and scheduled VC4 statically.

```text
PHASE8R_SCF_WHILE_STATIC=PASS
PHASE8R_NESTED_STRUCTURED_CF_STATIC=PASS
PHASE8R_TL_RANGE_STYLE_LOOP_STATIC=PASS
PHASE8R_PERSISTENT_LOOP_SKELETON_STATIC=PASS
READY_FOR_PHASE8RD_SWITCH_MULTI_EXIT_PROBE=YES
READY_FOR_TRITON=NO
```

This is static proof only. Phase 8Rc did not add TTIR control-flow import and
did not run hardware.

## Phase 8Rd Switch, Multi-Exit, And Irreducible CFG Policy

Phase 8Rd resolved the remaining scalar CFG edge cases before Phase 8.5.

```text
PHASE8R_CF_SWITCH_INDEX_SWITCH_POLICY=LOCKED
PHASE8R_MULTI_EXIT_REDUCIBLE_LOOP_POLICY=LOCKED
PHASE8R_IRREDUCIBLE_CFG_POLICY=PROBED_NOT_REQUIRED_FOR_TRITON
READY_FOR_PHASE8RE_LOWER_HALF_CF_COMPLETION=NOT_NEEDED
READY_FOR_PHASE8RF_HARDWARE_ISOLATION=YES
READY_FOR_TRITON=NO
```

`cf.switch` is value-surface-admissible only for scalar i32 flags. The
value-to-VC4Kernel path lowers it to explicit scalar comparisons and
`cf.cond_br`/`cf.br`, so verified VC4Kernel still presents only the branch
forms already accepted by the lower half. `scf.index_switch` is supported
through upstream `--convert-scf-to-cf`, which emits `cf.switch` in this LLVM
version; raw `scf.index_switch` remains a staged raw-SCF form.

Multi-exit reducible loops, including early exits, two scalar exits joining at
a merge block, and a loop-carried `vector<16xf32>` accumulator with early
scalar exit, pass the static `value -> vc4kernel -> ssavc4 -> scheduled vc4`
pipeline.

The raw irreducible CFG probe reaches SSAVC4 and is blocked in scheduled VC4
lowering with the exact diagnostic:

```text
SSAVC4 block-argument lowering supports only natural loops with conservative loop-carried data values
```

This is a lower-half natural-loop limitation, not a hardware-impossible claim.
Irreducible CFG is not required for sane Triton Phase 8.5 control-flow import.

## Runner Policy

VC4Value runners that encounter `scf` input must run upstream
`--convert-scf-to-cf` explicitly before value-to-VC4Kernel lowering. They must
log the canonicalization and preserve the after-scf-to-cf intermediate IR.

Pure `cf` inputs retain the existing direct value-to-VC4Kernel path.

## Phase 8f Isolation Hardware

Phase 8f ran focused value-layer control-flow isolation fixtures on real VC4
hardware through the standard value-to-VC4Kernel-to-SSAVC4-to-scheduled-VC4
path. The fixtures are direct `cf` value inputs, so no Phase 8f hardware
fixture required SCF canonicalization.

All Phase 8f isolation fixtures passed with `active_qpus=12`, `lanes=16`,
nonzero output hashes, zero mismatches, zero sentinel mismatches, and zero
launch failures:

- `value_cf_branch_guard_tail_vc4value`
- `value_cf_if_else_merge_vc4value`
- `value_cf_counted_loop_scalar_block_args_vc4value`
- `value_cf_loop_vector_carry_tail_vc4value`

READY_FOR_TRITON remains NO.

## Active Next Step

Phase 8 is locked as the value-layer control-flow phase. It does not claim TTIR
control-flow import.

The active next phase is:

```text
READY_FOR_PHASE8_5_TTIR_CONTROL_FLOW_BRIDGE=YES
READY_FOR_PHASE9_MASK_CLASSIFIER_AND_RICHER_MEMORY_LEGALITY=NO_PENDING_PHASE8_5_TTIR_CONTROL_FLOW_BRIDGE
READY_FOR_TRITON=NO
```

Any historical Phase 9 readiness language is superseded by this Phase 8.25
vertical workflow lock. Phase 9 mask classifier and richer memory legality may
begin only after Phase 8.5 locks TTIR control-flow support/reject
classification.

## Phase 8R Prerequisite Compatibility / Next Step

Phase 8 remains locked with no temporary frontend or value workarounds. Phase
8R control-flow completeness repair and Phase 8.5 TTIR control-flow bridge are
the next permitted steps before Phase 9 mask classifier and richer memory
legality.

```text
NO_TEMPORARY_FRONTEND_OR_VALUE_WORKAROUNDS=YES
READY_FOR_PHASE8_5_TTIR_CONTROL_FLOW_BRIDGE=YES
READY_FOR_PHASE8R_CONTROL_FLOW_COMPLETENESS_REPAIR=YES
READY_FOR_PHASE9_MASK_CLASSIFIER_AND_RICHER_MEMORY_LEGALITY=NO_PENDING_PHASE8_5_TTIR_CONTROL_FLOW_BRIDGE
READY_FOR_TRITON=NO
```

## Phase 8g Mixed Acceptance

Phase 8g added mixed value-layer control-flow hardware acceptance fixtures that
compose `cf.br`, `cf.cond_br`, counted loops, vector block arguments,
`program_id`, `vector.step`, f32/i32 value paths, TMU-backed transfer reads,
tail-masked transfer writes, and VDW inactive preserve behavior.

All Phase 8g value control-flow mixed fixtures passed on real VC4 hardware with
`active_qpus=12`, `lanes=16`, nonzero output hashes, zero mismatches, zero
sentinel mismatches, and zero launch failures:

- `mixed_value_cf_saxpy_loop_select_tail_vc4value`
- `mixed_value_cf_i32_f32_dual_path_tail_vc4value`

The Phase 7.5 C++ TTIR importer mixed hardware regression, Phase 5 VC4Value
mixed hardware regression, and P13 VC4Kernel mixed hardware regression passed.
The VC4Kernel regression completed with the existing mixed runner plus targeted
reruns for transient pre-boot recovery; every final-manifest fixture produced a
passing `VC4_TEST_RESULT`.

READY_FOR_TRITON remains NO.
