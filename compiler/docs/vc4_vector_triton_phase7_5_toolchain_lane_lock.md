# VC4 vector Triton Phase 7.5 toolchain lane lock

VC4_TRITON_CPP_TOOLCHAIN_LANE_LOCKED=YES
TRITON_TAG=v3.7.0
TRITON_LLVM_HASH=ac5dc54d509169d387fcfd495d71853d81c46484
TRITON_LLVM_PREFIX=/Users/rohanchanani/vc4-toolchains/triton-home-v3.7.0/.triton/llvm/llvm-ac5dc54d-macos-arm64
VC4_TRITON_LLVM_BUILD_DIR=compiler/build-triton-llvm
TRITON_CPP_PARSE_BUILD_DIR=/Users/rohanchanani/vc4-toolchains/triton-home-v3.7.0/triton-build-v3.7.0-prebuilt-llvm
CPP_FRONTEND_OPTION=VC4_ENABLE_TRITON_CPP_FRONTEND
CORE_BUILD_REMAINS_TRITON_FREE=YES
VC4_BUILD_TRITON_LLVM_CHECK_VC4=PASS
TRITON_CPP_DIALECT_BUILD=PASS
TRITON_CPP_DIALECT_PARSE_PROOF=PASS
TRITON_NVIDIA_DEPS_TRANSITIVE_ONLY=YES
TTIR_ACCEPTED_INPUT_HAS_NO_TTGIR_OR_NVIDIA_OPS=YES
READY_FOR_CPP_TTIR_TO_VC4VALUE_CONTEXT_COLLECTION=YES
READY_FOR_CPP_TTIR_TO_VC4VALUE_SOURCE_DRAFTING=YES
READY_FOR_TRITON=NO

## Locked Lane

The optional Triton C++ parse lane is locked for context collection and initial C++ source drafting. It uses:

```text
Triton source: /Users/rohanchanani/vc4-toolchains/triton-v3.7.0
Triton build:  /Users/rohanchanani/vc4-toolchains/triton-home-v3.7.0/triton-build-v3.7.0-prebuilt-llvm
LLVM prefix:   /Users/rohanchanani/vc4-toolchains/triton-home-v3.7.0/.triton/llvm/llvm-ac5dc54d-macos-arm64
VC4 build:     compiler/build-triton-llvm
```

The normal `compiler/build` remains Triton-free.

## Validation

Required validation passed:

```text
compiler/build vc4-opt/vc4-codegen: PASS
compiler/build check-vc4: PASS
compiler/build-triton-llvm vc4-opt/vc4-codegen: PASS
compiler/build-triton-llvm check-vc4: PASS
Triton C++ dialect dependency closure: PASS
Real TTIR C++ parse proof: PASS
TTIR no-TTGIR/NVIDIA input audit: PASS
```

The accepted real TTIR input was:

```text
compiler/test/CodeGen/Triton/Hardware/Run/triton_saxpy_select_tail_vc4triton/input.ttir.mlir
```

The parse proof command was:

```bash
.vc4_auto/triton_cpp_dialect_dependency_closure/probes/cpp_parse_probe_existing_build_build/parse_ttir_probe \
  compiler/test/CodeGen/Triton/Hardware/Run/triton_saxpy_select_tail_vc4triton/input.ttir.mlir \
  .vc4_auto/triton_cpp_dialect_dependency_closure/probes/ttir_roundtrip_existing_build.mlir
```

## Transitive Dependency Classification

The C++ parser link closure required TritonGPU/TritonNvidiaGPU-related object code because upstream Triton v3.7.0's TTIR C++ library objects reference those symbols.

Classification:

```text
TritonGPU/TritonNvidiaGPU/NVIDIA-related linkage: TRANSITIVE_BUILD_DEPENDENCY_ONLY
```

This does not authorize VC4 frontend support for TTGIR/NVGPU/NVVM/NVIDIA operations. The accepted TTIR input and roundtrip were audited and contained no forbidden TTGIR/NVIDIA dialect operations.

The VC4 frontend path remains:

```text
real TTIR tt dialect
  -> C++ TTIR-to-VC4Value importer/conversion
  -> standard value layer
  -> vc4kernel
  -> ssavc4
  -> scheduled vc4
```

It must not lower through:

```text
TTGIR
TritonGPU
NVGPU
NVVM
PTX
CUDA
NVIDIA codegen
```

## Scope

No TTIR-to-VC4Value lowering was implemented in this phase.

The Phase 7 Python importer remains in place.

No hardware was run.

`READY_FOR_TRITON` remains `NO`.

## Next Step

Proceed to TTIR-to-VC4Value C++ context collection for initial source drafting.
