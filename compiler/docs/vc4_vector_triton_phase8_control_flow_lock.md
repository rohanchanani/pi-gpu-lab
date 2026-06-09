# VC4 Vector/Triton Phase 8 Control-Flow Lock

PHASE8_CONTROL_FLOW_BOUNDARY=LOCKED
UPSTREAM_SCF_TO_CF_BOUNDARY=YES
RAW_SCF_NOT_IN_VC4KERNEL=YES
PHASE8_STATIC_CF_LOCKED=YES
VALUE_CF_CONTROL_FLOW_STATIC=PASS
VALUE_CF_BLOCK_ARGUMENTS_STATIC=PASS
VALUE_SCF_TO_CF_BOUNDARY_LOCKED=YES
TTIR_CONTROL_FLOW_STAGED=YES
TTIR_CPP_IMPORTER_STATIC_REGRESSION=PASS
NO_TEMPORARY_VALUE_CF_WORKAROUNDS=YES
READY_FOR_PHASE8F_HARDWARE_ISOLATION=YES
PHASE8_VALUE_CF_ISOLATION_HARDWARE=PASS
VALUE_CF_BRANCH_GUARD_HARDWARE=PASS
VALUE_CF_IF_ELSE_MERGE_HARDWARE=PASS
VALUE_CF_SCALAR_LOOP_HARDWARE=PASS
VALUE_CF_VECTOR_CARRY_LOOP_HARDWARE=PASS
READY_FOR_PHASE8G_MIXED_ACCEPTANCE=YES
PHASE8_VALUE_CF_MIXED_ACCEPTANCE=PASS
TTIR_CPP_IMPORTER_REGRESSION=PASS
VALUE_MIXED_REGRESSION=PASS
VC4KERNEL_MIXED_REGRESSION=PASS
PHASE8_MIXED_CLAIM_AUDIT=PASS
READY_FOR_PHASE8H_FINAL_LOCK=YES
TTIR_CONTROL_FLOW_IMPORT=NO
READY_FOR_TRITON=NO

## Boundary

Phase 8 is a value-layer control-flow phase. It does not add TTIR
control-flow import and does not add new math, memory, reduction, dot, or
Triton features.

`scf` is a value-surface input convenience only. Executable lowering to
VC4Kernel consumes `cf`, not raw `scf`. The required structured-control
boundary is the upstream MLIR pass:

```text
--convert-scf-to-cf
```

Custom SCF lowering is not part of Phase 8. If upstream SCF-to-CF ever becomes
unavailable, the phase must stop with a named blocker instead of inventing a
local lowering.

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
