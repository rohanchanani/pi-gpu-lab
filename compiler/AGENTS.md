# AGENTS.md

## Project purpose

This repository is building an MLIR-based backend for the Raspberry Pi VideoCore IV GPU (VC4/QPU).

## Current milestone

The current milestone is **only** the `vc4` MLIR dialect.

That means:

- implement the `vc4` dialect IR surface
- implement dialect registration, parser/printer, verifiers, types, attrs, and tests
- keep the implementation aligned with `docs/vc4-dialect-spec.md`

Do **not** implement any of the following unless the task explicitly asks for them:

- lowering from upstream `gpu` dialect
- code generation to qasm
- kernel launcher generation (`kernel_launch.c` / `.h`)
- runtime integration
- fragment/TLB/stencil/scoreboard/control-list/shader-state features

## Source of truth

Read these first before changing the dialect:

1. `docs/vc4-dialect-spec.md`
2. `docs/vc4-implementation-guide.md`
3. `docs/codex-vc4-prompts.md` when following the staged implementation plan

If code and docs disagree, prefer the dialect spec unless the task explicitly says to revise the spec.

## Scope guardrails

This milestone is **compute/QPU focused**.

Keep:
- QPU structured ops
- QPU scheduled sink ops
- uniforms
- TMU / SFU
- VPM / DMA
- semaphore
- mutex
- host interrupt
- thread switching / program end
- user-program queue and relevant V3D system hooks

Do not reintroduce:
- TLB / tile buffer
- scoreboard
- stencil
- varyings
- fragment shader builtins
- control lists
- GL/NV/VG shader-state records

## Repository conventions

- Use a standalone out-of-tree MLIR project structure.
- Prefer CMake + Ninja.
- Prefer TableGen/ODS for op/type/attr definitions.
- Add custom C++ only where ODS/declarative assembly is not enough.
- Keep structured and scheduled forms separate.
- Do not mix speculative future work into the current milestone.

## Expected layout

Use or preserve this general layout:

- `include/vc4/...`
- `lib/...`
- `tools/vc4-opt/...`
- `test/...`
- `docs/...`

Keep dialect IR code under a clear `Dialect/VC4` or equivalent subtree.

## Build and test expectations

If the build system exists, use an out-of-tree build directory:

```bash
cmake -G Ninja -S . -B build
ninja -C build vc4-opt check-vc4
```

If the project needs `MLIR_DIR` / `LLVM_DIR`, pass them explicitly rather than hardcoding machine-specific paths in source files.

Every non-trivial dialect change should include or update tests.

Minimum expected tests for dialect work:

- parser/printer round-trip tests
- verifier-negative tests
- `vc4-opt` smoke tests
- scheduled-form legality tests
- function-form segregation tests

## Change checklist

Before considering a task done:

- update the implementation
- update tests
- update docs if the implemented dialect surface changed
- verify new ops/types/attrs are registered
- verify parser/printer round-trips for new IR syntax
- verify negative tests exist for important verifier rules

## Implementation style

- Prefer explicit enum names that mirror hardware concepts.
- Keep op names stable and predictable.
- Keep verifiers local and precise.
- Reserve cross-instruction hazards for later passes unless the task explicitly asks for them.
- Avoid silently broadening the dialect surface.

## When uncertain

If there is ambiguity, do not invent a new design. Reconcile the code with `docs/vc4-dialect-spec.md` and keep the implementation narrowly within the milestone.