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
5. `docs/vc4asm-notes.md`
6. `docs/reference-links.md`

## Ground rules
- Prefer small, reversible steps over broad speculative design.
- Do not broaden language scope without updating `docs/project-scope.md`.
- Keep the distinction explicit between:
  - CUDA-subset frontend semantics
  - VC4-native backend lowering
- Preserve existing low-level/manual VC4 materials in this repo.
- Treat `third_party/VC4C` as reference-only prior art unless explicitly asked to modify it.
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

## vc4asm / backend guidance
- For vc4asm-related backend work, read `docs/vc4asm-notes.md` first.
- Use `docs/reference-links.md` for authoritative assembler references.
- Keep the `vc4` dialect above raw assembler directives, macros, and bitfield encodings unless the task is specifically about emission.
- Model machine structure first:
  - uniforms
  - staged DMA load/store
  - VPM reads/writes
  - arithmetic
  - later: control flow / scheduling
- Do not mirror raw `vc4asm` syntax too early.

## VC4C guidance
- `third_party/VC4C` is reference-only prior art.
- Use it to study backend lowering, runtime conventions, and VC4-specific implementation ideas.
- Do not copy structure blindly.
- Do not modify it unless explicitly asked.

## Early-stage coding preferences
- Hardcoded or toy frontend inputs are acceptable.
- Parser work is lower priority than IR design and lowering structure.
- Pseudo-assembly / text dumps are acceptable intermediate milestones.
- First milestone is correctness and architecture clarity, not optimization.
- Prefer backend concepts generalized around machine structure, not around the current kernel.

## What to avoid right now
- full CUDA parser
- full runtime API design
- broad optimizer work
- premature support for atomics, synchronization, or advanced memory semantics
- oversized target dialect design
- raw vc4asm emission details in the dialect
- schedule/register/delay-slot modeling before the staged backend IR is clear