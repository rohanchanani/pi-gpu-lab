# AGENTS.md

## Project goal
Build a restricted CUDA-subset compiler and runtime for the Raspberry Pi VideoCore IV GPU (VC4), targeting QPUs and emitting `vc4asm` first.

## Current priority
Create an extensible compiler foundation that can reach the first end-to-end generated kernel, likely SAXPY.

## Read these first
1. `README.md`
2. `docs/project-scope.md`
3. `docs/architecture.md`
4. `docs/task-board.md`

## Ground rules
- Prefer small, reversible steps over broad speculative design.
- Do not broaden language scope without updating `docs/project-scope.md`.
- Keep the distinction explicit between:
    - CUDA-subset frontend semantics
    - VC4-native backend lowering
- Preserve existing low-level/manual VC4 materials in this repo.
- For nontrivial changes, update docs and tests in the same change.
- Avoid large refactors unless they clearly simplify the architecture.

## Planning
For multi-step work:
- write or update a short plan in `docs/task-board.md`
- keep one main task per session when possible
- leave behind a short “next steps” note after significant changes

## Build / test conventions
- Put reproducible commands in `scripts/`
- Prefer scripts over ad hoc shell commands in docs
- If a build/test workflow is missing, add a stub script before adding major code
- Keep tests lightweight and text-based early on

## Compiler architecture expectations
The intended pipeline is:

tiny CUDA subset
-> MLIR scalar dialects (`func`, `arith`, `scf`, likely `memref`)
-> MLIR `vector` dialect with width-16 execution groups
-> custom `vc4` dialect
-> `vc4asm`

Do not skip directly to target-specific assembly unless the task is explicitly about manual kernels or experiments.

## VC4-specific cautions
- QPU execution is effectively width-16 SIMD
- backend must eventually account for hazards, delay slots, special registers, and staged memory movement
- do not assume full CUDA semantics
- do not implement `__shared__`, barriers, or advanced CUDA features unless they are in scope in `docs/project-scope.md`

## Early-stage coding preferences
- hardcoded or toy frontend inputs are acceptable
- parser work is lower priority than IR design and lowering structure
- pseudo-assembly / text dumps are acceptable intermediate milestones
- first milestone is correctness and architecture clarity, not optimization

## What to avoid right now
- full CUDA parser
- full runtime API design
- broad optimizer work
- premature support for atomics, synchronization, or advanced memory semantics
- oversized target dialect design