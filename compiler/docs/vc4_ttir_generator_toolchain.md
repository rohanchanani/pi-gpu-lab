# VC4 TTIR Generator Toolchain

VC4_TTIR_GENERATOR_TOOLCHAIN_SPEC=YES
TRITON_TAG=v3.7.0
TRITON_SOURCE_HEAD=5f3f125e8f63c24613f1f73b937442864f263f94
TRITON_LLVM_HASH=ac5dc54d509169d387fcfd495d71853d81c46484
TRACKED_SETUP_TOOL=tools/vc4_setup_ttir_generator.py
TRACKED_VALIDATION_TOOL=tools/vc4_check_ttir_generator.py
TRACKED_TOOLCHAIN_SPEC=tools/vc4_ttir_generator_toolchain.json
TRACKED_REQUIREMENTS=tools/requirements-triton-generator.txt
NORMAL_CHECK_VC4_REQUIRES_PYTHON_TRITON=NO
READY_FOR_TRITON=NO

## Purpose

The VC4 Triton path uses real Triton frontend TTIR for source evidence and
fixture generation. The generator environment is optional: normal `check-vc4`
uses checked-in `.ttir.mlir` snapshots and must not depend on Python Triton.

This document defines the reproducible setup lane for developers who need to
regenerate TTIR snapshots from real Triton source.

## What Is Tracked

Tracked in this repository:

- `tools/vc4_emit_ttir.py`: emits frontend TTIR from a real `@triton.jit`
  kernel.
- `tools/vc4_setup_ttir_generator.py`: creates a local generator venv from an
  explicit pinned Triton checkout and LLVM prefix.
- `tools/vc4_check_ttir_generator.py`: validates import, generation, audit, and
  `vc4-triton-opt` roundtrip.
- `tools/vc4_ttir_generator_toolchain.json`: machine-readable pinned
  toolchain and smoke-test specification used by the setup and validation
  tools.
- `tools/requirements-triton-generator.txt`: small Python runtime dependencies
  for the generator venv. Triton itself is intentionally not listed.
- source-controlled TTIR snapshots and manifests under
  `compiler/test/CodeGen/Triton/Snapshots/`.

Not tracked:

- Triton checkout;
- LLVM toolchain bundle;
- generator venv;
- copied `libtriton.so` / `libproton.so`;
- generated scratch outputs under `.vc4_auto`.

## Required External Inputs

The setup tool expects the developer to provide or fetch these pinned inputs:

```text
Triton source tag:  v3.7.0
Triton source head: 5f3f125e8f63c24613f1f73b937442864f263f94
LLVM hash:          ac5dc54d509169d387fcfd495d71853d81c46484
```

The current local convention is:

```text
Triton source: /Users/rohanchanani/vc4-toolchains/triton-v3.7.0
LLVM prefix:   /Users/rohanchanani/vc4-toolchains/triton-home-v3.7.0/.triton/llvm/llvm-ac5dc54d-macos-arm64
```

The setup tool is path-based by default. It does not clone Triton, download
LLVM, or install an unpinned Triton wheel.

## Setup

Create a generator environment:

```bash
python3 tools/vc4_setup_ttir_generator.py \
  --spec tools/vc4_ttir_generator_toolchain.json \
  --triton-source /Users/rohanchanani/vc4-toolchains/triton-v3.7.0 \
  --llvm-prefix /Users/rohanchanani/vc4-toolchains/triton-home-v3.7.0/.triton/llvm/llvm-ac5dc54d-macos-arm64 \
  --out .vc4_auto/ttir_generator \
  --force
```

If the Triton checkout has multiple or nonstandard Python extension build
outputs, pass:

```bash
--triton-python-extension-dir /path/to/triton/_C
```

That directory must contain `libtriton.so`; `libproton.so` is copied when
present.

The setup tool writes:

```text
.vc4_auto/ttir_generator/venv
.vc4_auto/ttir_generator/triton_python_overlay
.vc4_auto/ttir_generator/run_vc4_emit_ttir.sh
.vc4_auto/ttir_generator/generator_env.json
```

The overlay makes the pinned built extension visible before source-tree stubs
while executing the real Triton v3.7.0 Python package.

## Validation

Validate the generator against a real source snapshot and the C++ TTIR parser
lane:

```bash
python3 tools/vc4_check_ttir_generator.py \
  --spec tools/vc4_ttir_generator_toolchain.json \
  --generator-root .vc4_auto/ttir_generator \
  --vc4-triton-opt compiler/build-triton-llvm/bin/vc4-triton-opt
```

The validator:

- invokes `.vc4_auto/ttir_generator/run_vc4_emit_ttir.sh`;
- generates a real TTIR snapshot from a source-controlled Triton kernel;
- audits required TTIR forms such as `tt.func`, `scf.if`, `scf.yield`,
  `tt.load`, and `tt.store`;
- rejects backend dialects (`ttg.`, `triton_gpu.`, `nvgpu.`, `nvvm.`) and
  `arith.sitofp` in the smoke snapshot;
- roundtrips the generated TTIR through `vc4-triton-opt`;
- preserves `READY_FOR_TRITON=NO`.

## Snapshot Generation

Use the generated wrapper for fixture-specific snapshot generation:

```bash
.vc4_auto/ttir_generator/run_vc4_emit_ttir.sh SOURCE.py \
  --kernel-name KERNEL \
  --signature '*fp32,*fp32,i32,i32,16' \
  --target cuda:80:32 \
  --num-warps 1 \
  --num-stages 3 \
  --required-triton-major-minor 3.7 \
  --out OUT.ttir.mlir \
  --metadata-out OUT.metadata.json
```

Every promoted snapshot must also parse or roundtrip through:

```bash
compiler/build-triton-llvm/bin/vc4-triton-opt OUT.ttir.mlir -o ROUNDTRIP.mlir
```

## Project Invariants

- Do not check in generator venvs, Triton checkouts, LLVM bundles, or copied
  shared libraries.
- Do not use unpinned `pip install triton` as compiler evidence.
- Do not make normal `check-vc4` depend on Python Triton.
- Source-controlled TTIR snapshots are reproducibility artifacts, not a
  replacement for real generated TTIR provenance.
- `READY_FOR_TRITON` remains `NO` until the project separately locks full
  Triton support.
