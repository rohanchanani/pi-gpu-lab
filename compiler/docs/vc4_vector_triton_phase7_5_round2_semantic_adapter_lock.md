# VC4 Vector/Triton Phase 7.5 Round 2 Semantic Adapter Lock

PHASE75R2_SEMANTIC_ADAPTER_LOCKED=YES
PHASE75R2_RESULT=LOCKED
TTIR_TYPE_ADAPTER_HARDENED=YES
TTIR_ATTR_ADAPTER_HARDENED=YES
POINTER_TYPE_CLASSIFICATION_TYPED=YES
AXIS_CLASSIFICATION_EXACT=YES
AXES_1_2_STAGED=YES
MAKE_RANGE_STRUCTURED_CLASSIFICATION=YES
PRINTED_TYPE_SEMANTIC_MATCHING=NO
PRINTED_ATTR_SUBSTRING_SEMANTIC_MATCHING=NO
MAKE_RANGE_TEXT_FALLBACK=NO
ARG_NAME_SEMANTIC_HEURISTIC=NO
STRUCTURAL_ARGUMENT_ROLE_CLASSIFICATION=YES
SEMANTIC_STRING_MATCHING_AUDIT=PASS
CPP_IMPORTER_STATIC_REGRESSION=PASS
CPP_IMPORTER_HARDWARE_REGRESSION=PASS
VALUE_MIXED_REGRESSION=PASS
VC4KERNEL_MIXED_REGRESSION=PASS
READY_FOR_PHASE75R2F_HARDWARE_REGRESSION=YES
READY_FOR_PHASE75R3_OUTPUT_MODULE_HARDENING=YES
READY_FOR_PHASE8_CONTROL_FLOW_AND_LOOP_BOUNDARY=NO_PENDING_OUTPUT_MODULE_HARDENING
READY_FOR_TRITON=NO

## Scope

Round 2 hardens the accepted C++ TTIR-to-VC4Value importer path without adding
new TTIR feature support. The importer remains a Phase 7.5 elementwise V1 bridge
from parsed `tt` dialect IR into the standard VC4 value layer.

The default compiler build remains Triton-free. Triton headers, generated APIs,
and object closure are used only by the optional `compiler/build-triton-llvm`
lane that builds `vc4-triton-opt`.

## What Was Fixed

Pointer classification now uses `TTIRTypeAdapter` and the typed Triton
`mlir::triton::PointerType` API. It no longer prints MLIR types or searches for
`!tt.ptr<`.

Program axis and `tt.make_range` classification now use `TTIRAttrAdapter`.
Axes are read from generated Triton op/attr APIs, and `tt.make_range` start/end
come from structured accessors or exact integer attributes inside the adapter.
The old printed-operation fallback for range detection is removed.

Argument roles now come from parsed MLIR types and SSA use-def structure:

- pointer directions from `tt.load` and `tt.store` uses;
- tail bound from the canonical `offsets < bound` mask;
- scalar compute roles from parsed scalar types;
- shape metadata from the structurally proven tail-bound argument.

Source names remain provenance metadata only.

## Allowed String Uses

The importer may still use strings for:

- exact MLIR operation names such as `tt.load` and `tt.store`;
- exact attribute names such as `axis` and `vc4value.arg_name`;
- diagnostic messages;
- value-layer operation construction at the output boundary;
- source-location provenance metadata such as `vc4value.arg_name`.

## Forbidden String Uses

The importer must not use strings to decide TTIR semantics:

- no printed type matching for pointer classification;
- no printed attr/op matching for axis or range classification;
- no permissive substring matching for axes, ranges, or types;
- no `!tt.ptr<` parsing;
- no operation-printing fallback for `tt.make_range`;
- no source argument name heuristic for tail-bound, scalar role, memory
  direction, shape role, or type lowering;
- no Python semantic lowering path in accepted tests;
- no direct `vc4kernel`, `ssavc4`, or scheduled `vc4` emission from
  `TritonToVC4Value`.

## Remaining Staged Features

Round 2 does not add axes 1/2, grid rank greater than 1, wider ranges,
reductions, dot/contract, gather/scatter, block pointers, tensor descriptors,
rank-2 tensors, atomics, or control flow.

Axis 1/y and 2/z are recognized exactly but remain staged until value
grid-rank greater than 1 lowering exists. Non-`0..16` `tt.make_range` forms
remain staged. Unsupported pointer element types and block-pointer-like forms
remain staged or rejected by the current target profile.

## Static Regression

Round 2e static regression passed:

- default `compiler/build` tools and `check-vc4`;
- optional `compiler/build-triton-llvm` tools and `check-vc4`;
- `compiler/test/Conversion/TritonToVC4Value`;
- `compiler/test/CodeGen/Triton`;
- `compiler/test/TritonFrontend`;
- `compiler/test/Conversion/VC4ValueToVC4Kernel`;
- `compiler/test/CodeGen/VC4Value`;
- `compiler/test/CodeGen/VC4Kernel`;
- static TTIR-to-VC4Value-to-VC4Kernel pipeline smokes for vector-add,
  saxpy/select, and i32 add/select snapshots.

The pipeline smoke outputs are clean at the layer boundary: value IR contains no
TTIR or lower-half dialects, and VC4Kernel output contains no producer dialects.

## Hardware Final Acceptance

Round 2f hardware regression passed. The C++ importer isolation and mixed
hardware suites still lower source-controlled TTIR snapshots through
`vc4-triton-opt --convert-triton-to-vc4-value`, VC4Value verification,
VC4Kernel verification, SSAVC4, scheduled VC4, generated artifacts, and real
VC4 hardware.

Round 2 changed importer classification internals only. Accepted TTIR
elementwise V1 semantics remained hardware-correct: the C++ importer hardware
regression, Phase 5 VC4Value mixed regression, and P13 VC4Kernel mixed
regression all passed with zero mismatches, zero sentinel mismatches, and zero
launch failures.

The semantic string matching audit and Python semantic importer retirement
audit also passed. The accepted semantic TTIR-to-value path remains the C++
importer.

## Next Step

The exact next step is Phase 7.5 Round 3 output-module and lifetime hardening.
Round 3 is still required before Phase 8 control flow and loop-boundary work.
Round 2f does not claim readiness for broader Triton support and does not flip
the Triton readiness marker to affirmative.
