# VC4 Triton C++ frontend toolchain lane

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

## Lane Summary

This optional lane uses Triton v3.7.0 with Triton's prebuilt LLVM/MLIR package:

```text
/Users/rohanchanani/vc4-toolchains/triton-home-v3.7.0/.triton/llvm/llvm-ac5dc54d-macos-arm64
```

That prefix reports LLVM/MLIR `23.0.0git` and corresponds to Triton's pinned LLVM hash:

```text
ac5dc54d509169d387fcfd495d71853d81c46484
```

The normal `compiler/build` lane remains Homebrew LLVM-backed and Triton-free. The optional VC4 lane is:

```text
compiler/build-triton-llvm
```

## Triton Source And Build

Triton source:

```text
/Users/rohanchanani/vc4-toolchains/triton-v3.7.0
```

Triton build used for the C++ parse proof:

```text
/Users/rohanchanani/vc4-toolchains/triton-home-v3.7.0/triton-build-v3.7.0-prebuilt-llvm
```

Configure command:

```bash
cmake -S "$TRITON_SRC" -B "$TRITON_BUILD" -G Ninja \
  -DTRITON_BUILD_PYTHON_MODULE=OFF \
  -DTRITON_BUILD_PROTON=OFF \
  -DTRITON_BUILD_UT=OFF \
  -DTRITON_OFFLINE_BUILD=ON \
  -DLLVM_DIR="$LLVM_PREFIX/lib/cmake/llvm" \
  -DMLIR_DIR="$LLVM_PREFIX/lib/cmake/mlir" \
  -DLLD_DIR="$LLVM_PREFIX/lib/cmake/lld" \
  -DCMAKE_BUILD_TYPE=Release
```

## Parse Proof

The real source-controlled TTIR input was:

```text
compiler/test/CodeGen/Triton/Hardware/Run/triton_saxpy_select_tail_vc4triton/input.ttir.mlir
```

The proof used a small C++ parser probe under:

```text
.vc4_auto/triton_cpp_dialect_dependency_closure/probes/cpp_parse_probe_existing_build
```

The exact parse command was:

```bash
.vc4_auto/triton_cpp_dialect_dependency_closure/probes/cpp_parse_probe_existing_build_build/parse_ttir_probe \
  compiler/test/CodeGen/Triton/Hardware/Run/triton_saxpy_select_tail_vc4triton/input.ttir.mlir \
  .vc4_auto/triton_cpp_dialect_dependency_closure/probes/ttir_roundtrip_existing_build.mlir
```

The command printed:

```text
CPP_TTIR_PARSE_PROBE=PASS
```

The roundtrip output was nonempty and retained TTIR structure including `tt.func`, `tt.load`, and `tt.store`.

## C++ Dependencies

The probe registered the standard MLIR dialects needed by the TTIR snapshot plus Triton `tt`.

The minimal observed C++ link closure used object files from:

```text
TritonIR
TritonGPUIR
TritonTools
TritonNvidiaGPUIR
TritonAnalysis Utility.cpp.o
TritonGPUTransforms Utility.cpp.o
f2reduce
```

`TritonGPUIR`, `TritonTools`, `TritonNvidiaGPUIR`, and `f2reduce` are recorded as `TRANSITIVE_BUILD_DEPENDENCY_ONLY`. Their presence is an upstream Triton C++ library dependency for linking the `tt` dialect parser probe. They are not accepted VC4 frontend input dialects and are not part of a lowering path.

The VC4 frontend path must reject TTGIR/NVGPU/NVVM/NVIDIA input dialect operations and must not lower through TTGIR, TritonGPU, NVGPU, NVVM, PTX, CUDA, or NVIDIA codegen.

## Scope

No TTIR-to-VC4Value lowering was implemented in this lane setup package.

Current post-Phase-7.5 state: accepted TTIR-to-VC4Value semantic lowering is
the optional C++ `vc4-triton-opt --convert-triton-to-vc4-value` path. The
Python `vc4-triton-import` tool is retained only for TTIR inventory/snapshot
workflows; its old `lower-elementwise-v1` semantic lowering mode is retired and
must not be used as an accepted test, runner, or documentation path.

No hardware was run.

`READY_FOR_TRITON` remains `NO`.

## Next Step

Run TTIR-to-VC4Value C++ context collection for the initial source drafting package.
