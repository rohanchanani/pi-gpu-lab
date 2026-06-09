# VC4 Vector/Triton Phase 7.5 C++ Frontend Lock

VC4_TRITON_CPP_FRONTEND_PHASE7_5_LOCKED=YES
VC4_TRITON_CPP_FRONTEND_OPTIONAL_BUILD=YES
CORE_BUILD_REMAINS_TRITON_FREE=YES
TTIR_TO_VC4VALUE_CPP_IMPORTER_ACCEPTED=YES
PHASE7_PYTHON_SEMANTIC_IMPORTER_RETIRED=YES
TTIR_ELEMENTWISE_V1_CPP_IMPORTER_HARDWARE=PASS
TTIR_ACCEPTED_INPUT_HAS_NO_TTGIR_OR_NVIDIA_OPS=YES
READY_FOR_PHASE8_CONTROL_FLOW_AND_LOOP_BOUNDARY=YES
READY_FOR_TRITON=NO

## Scope

Phase 7.5 locks the long-term C++ frontend foundation for the Phase 7
elementwise V1 TTIR subset. The accepted semantic lowering path is:

```text
real source-controlled .ttir.mlir
  -> compiler/build-triton-llvm/bin/vc4-triton-opt --convert-triton-to-vc4-value
  -> func/vc4value/vector/memref/arith value IR
  -> --vc4-verify-value-surface
  -> --convert-vc4-value-to-vc4kernel
  -> --verify-vc4kernel
  -> vc4kernel
  -> ssavc4
  -> scheduled vc4
  -> artifacts/runtime
  -> real VC4 hardware
```

`VC4_ENABLE_TRITON_CPP_FRONTEND` defaults OFF. The default `compiler/build`
lane remains Triton-free and has no `vc4-triton-opt` target. The optional
`compiler/build-triton-llvm` lane builds `vc4-triton-opt` when the option is ON.
`VC4TritonFrontendDeps` is the only adapter target that may contain Triton
source, build, include, object, or library closure details.

## Accepted Frontend

The C++ importer parses real TTIR through MLIR/Triton dialect registration and
lowers only the Phase 7 elementwise V1 forms into the standard VC4 value layer.
It emits value IR only and must not emit `vc4kernel`, `ssavc4`, or scheduled
`vc4` directly.

The accepted source-controlled TTIR snapshots are:

- `examples/triton/phase6/generated/vector_add_b16.ttir.mlir`;
- `examples/triton/phase6/generated/saxpy_select_b16.ttir.mlir`;
- `examples/triton/phase6/generated/i32_add_select_b16.ttir.mlir`.

The old Python `vc4-triton-import --mode lower-elementwise-v1` semantic path is
retired from accepted static and hardware tests. Python remains allowed for
Triton source to TTIR snapshot generation and inventory.

## Hardware Lock

The C++ importer path is hardware-proven by the Phase 7.5 isolation and mixed
TTIR suites at active_qpus=12, lanes=16, with CPU oracles, sentinels, nonzero
output hashes, value-surface verification, and value-to-VC4Kernel verification.
The mixed fixture claims are source-controlled and audited under
`compiler/test/CodeGen/Triton/Hardware/MixedAcceptance/`.

## Boundaries

Accepted TTIR input is `tt` dialect TTIR only. TTGIR, `triton_gpu`, `gpu`,
`nvgpu`, `nvvm`, `rocdl`, LLVM IR, PTX, cubin, and other target backend forms
are not accepted frontend inputs.

Phase 7.5 does not add reductions, dot/contract, gather/scatter, block
pointers, atomics, rank-2 tensors, control flow, axes 1/2, or
`BLOCK_SIZE != 16` support.
