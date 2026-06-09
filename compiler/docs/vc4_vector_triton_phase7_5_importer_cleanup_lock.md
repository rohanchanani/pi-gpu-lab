# VC4 vector Triton Phase 7.5 importer cleanup lock

TRITON_TO_VC4VALUE_CONVERSION_LIT_SUITE=YES
CPP_TTIR_TO_VC4VALUE_ACCEPTED_STATIC_PATH=YES
PYTHON_SEMANTIC_TTIR_IMPORTER_RETIRED=YES
PYTHON_TTIR_GENERATION_RETAINED=YES
ACCEPTED_TTIR_TO_VALUE_PATH_IS_CPP=YES
LEGACY_PYTHON_SEMANTIC_AUDIT=PASS
DEFAULT_BUILD_REMAINS_TRITON_FREE=YES
VC4_TRITON_FRONTEND_DEPS_ADAPTER=YES
READY_FOR_IMPORTER_HARDENING_TYPE_AXIS_ERASE=YES
READY_FOR_PHASE8_CONTROL_FLOW_AND_LOOP_BOUNDARY=NO_PENDING_IMPORTER_HARDENING
READY_FOR_TRITON=NO

## What This Cleanup Addressed

This follow-up lock closes the first two post-Phase-7.5 cleanup concerns:

```text
dedicated TritonToVC4Value conversion tests: LOCKED
legacy Python semantic TTIR-to-VC4Value importer path: RETIRED
```

The dedicated lit suite lives under:

```text
compiler/test/Conversion/TritonToVC4Value
```

It statically proves the accepted importer pipeline from real
source-controlled TTIR snapshots:

```text
real TTIR tt dialect
  -> vc4-triton-opt --convert-triton-to-vc4-value
  -> standard VC4 value IR
  -> vc4-opt --vc4-verify-value-surface
  -> vc4-opt --convert-vc4-value-to-vc4kernel
  -> vc4-opt --verify-vc4kernel
```

The old Python semantic TTIR-to-value bridge is no longer an accepted path for
tests, runners, or current-state documentation.

## Remaining Importer Hardening Concerns

This cleanup does not add new TTIR lowering semantics. The remaining importer
hardening concerns are intentionally left for the next packages:

```text
Triton type adapter hardening
program_id axis detection and diagnostic hardening
tt.func erase/unlink/import cleanup hardening
```

Those concerns must be resolved before Phase 8 control-flow and loop-boundary
work starts.

## Accepted C++ Semantic Import Path

The current accepted semantic TTIR-to-VC4Value path is:

```bash
compiler/build-triton-llvm/bin/vc4-triton-opt \
  input.ttir.mlir \
  --convert-triton-to-vc4-value \
  -o output.value.mlir
```

The resulting value IR is then validated and lowered through the existing value
pipeline:

```bash
compiler/build-triton-llvm/bin/vc4-opt \
  output.value.mlir \
  --vc4-verify-value-surface \
  --convert-vc4-value-to-vc4kernel \
  --verify-vc4kernel
```

The default `compiler/build` lane remains Triton-free and does not build or
require `vc4-triton-opt`.

## Allowed Python Role

Python remains allowed for:

```text
Triton source-to-TTIR generation
TTIR snapshot regeneration and comparison
TTIR inventory and explanation helpers that do not emit semantic value IR
```

Python is not an accepted semantic TTIR-to-VC4Value lowering path. The retired
mode reports:

```text
Python semantic TTIR-to-VC4Value lowering is retired. Use vc4-triton-opt --convert-triton-to-vc4-value.
```

## Test Suite Summary

The dedicated conversion suite contains success coverage for real TTIR
snapshots:

```text
vector-add b16 TTIR -> VC4 value IR -> VC4Kernel verifier
saxpy/select/tail TTIR -> VC4 value IR -> VC4Kernel verifier
i32 add/select/tail TTIR -> VC4 value IR -> VC4Kernel verifier
```

It also includes staged or negative coverage for currently unsupported TTIR and
backend forms where deterministic parser-valid inputs are available.

The suite is optional-frontend aware:

```text
compiler/build: Triton importer tests are unsupported when vc4-triton-opt is absent
compiler/build-triton-llvm: Triton importer tests run and pass
```

## Audit Summary

The R1d source boundary audits confirmed:

```text
no fixture-name dispatch in TritonToVC4Value.cpp
no regex semantic parser in TritonToVC4Value.cpp
no direct vc4kernel/ssavc4/vc4.qpu emission from the TTIR importer
no accepted TTGIR/NVGPU/NVVM path
no accepted Python semantic TTIR-to-value path
Triton object/library closure is quarantined behind VC4TritonFrontendDeps
default compiler/build has no vc4-triton-opt target
```

Historical documentation may mention the Phase 7 Python smoke bridge only when
it is clearly described as retired or historical. Current accepted tests and
runners must use the C++ importer for semantic TTIR-to-value lowering.

## Scope

No new executable semantics were added in this cleanup.

No hardware was run.

`READY_FOR_TRITON` remains `NO`.
