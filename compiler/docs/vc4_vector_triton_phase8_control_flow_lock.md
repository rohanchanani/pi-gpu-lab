# VC4 Vector/Triton Phase 8 Control-Flow Lock

PHASE8_CONTROL_FLOW_BOUNDARY=LOCKED
UPSTREAM_SCF_TO_CF_BOUNDARY=YES
RAW_SCF_NOT_IN_VC4KERNEL=YES
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

Hardware was not run for this boundary lock. READY_FOR_TRITON remains NO.
