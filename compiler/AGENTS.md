# AGENTS.md

## Project purpose

This repository is building an MLIR-based backend for the Raspberry Pi VideoCore IV GPU (VC4/QPU).

## Current compiler stack

The active lower stack is:

```text
vc4tile        // planned M4 VC4 Tile dialect above SSAVC4
  v
ssavc4        // implemented M3 target-specific SSA machine IR
  v
vc4           // scheduled, QASM-near sink dialect
  v
QASM / shader arrays / kernel_launch.c/h / manifest/layout artifacts
  v
vc4_runtime
```

M4 is the VC4 Tile dialect (`vc4tile`) and lowering from `vc4tile` to `ssavc4`. Direct producer lowering from Triton, IREE, MLIR `gpu`, or other frontend IR into `vc4tile` is future work after M4.

Do **not** implement any of the following unless the task explicitly asks for them:

- lowering from upstream producer IR such as MLIR `gpu`, Triton, or IREE
- direct `gpu -> ssavc4` or `gpu -> vc4` shortcuts
- changes to the scheduled `vc4` artifact/runtime lower half unless required by the task
- fragment/TLB/stencil/scoreboard/control-list/shader-state features

## Source of truth

Read these first before changing the relevant dialect layer:

1. `docs/vc4tile_architecture_and_project_plan.md` for M4 `vc4tile` work
2. `docs/codegen/ssavc4-ir-design-m3-post-cleanup.md` for the SSAVC4 lower half
3. `docs/vc4-dialect-spec.md` for the scheduled `vc4` sink

If code and docs disagree, prefer the layer-specific current design doc unless the task explicitly says to revise it.

## Scope guardrails

Compiler work is **compute/QPU focused**.

Keep:
- `vc4tile` as the planned tile/kernel IR above SSAVC4
- `ssavc4` as target-specific machine SSA with existing spilling and block-argument/edge-copy lowering support
- scheduled `vc4` as the QASM-near artifact sink
- launch/resource metadata and the existing `vc4_runtime` ABI

Do not reintroduce:
- removed structured `vc4` operations as the active sink surface
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
- Keep `vc4tile`, `ssavc4`, and scheduled `vc4` forms separate.
- Do not mix producer-integration work into M4 unless the task explicitly asks for it.

## Expected layout

Use or preserve this general layout:

- `include/vc4/...`
- `lib/...`
- `tools/vc4-opt/...`
- `test/...`
- `docs/...`

Keep dialect IR code under clear dialect subtrees such as `Dialect/VC4Tile`, `Dialect/SSAVC4`, or `Dialect/VC4`.

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
- lowering legality tests for the layer being changed
- scheduled-form legality tests when touching scheduled `vc4`

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

If there is ambiguity, do not invent a new design. Reconcile the code with the layer-specific current design doc and keep the implementation narrowly within the requested milestone.
