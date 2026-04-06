# AGENTS.md

## Project identity
`pi-gpu-lab` contains two related but distinct bodies of work:

- existing low-level / handwritten VideoCore IV QPU experiments and demos under `code/`
- a new compiler project under `compiler/` whose job is to lower GPU-style kernel IR to VC4/QPU execution

The compiler is the novel part. It should become an MLIR-based backend path for the Raspberry Pi VideoCore IV GPU, not a reimplementation of the handwritten examples and not a clone of VC4C.

## What this compiler is for
Build an honest MLIR lowering stack for VC4/QPUs with `vc4asm` as the first concrete emission target.

The intended direction is:

high-level kernel IR
-> `gpu`
-> `vector`
-> custom staged-memory `vc4` dialect
-> `vc4asm`

Longer term, kernels may come from multiple source levels:
- CUDA-like kernels first
- possibly later Triton-like or PyTorch-ish generated forms
- possibly direct generation at an intermediate IR level

But the current research target is narrower: determine which IR level is best for generation, lowering, and optimization while still landing on a real VC4 backend.

## Immediate priority
Get one narrow, real, end-to-end lowering path for a SAXPY-like kernel.

That path should:
- reflect actual VC4 machine structure
- stay small enough to understand completely
- avoid kernel-specific hacks that would poison later extension
- stop at `vc4asm` before attempting a full runtime/frontend story

The first generated kernel should be comparable in intent to the handwritten SAXPY path in `code/3-saxpy/`.

## Read first
Before changing compiler architecture or repo guidance, read these:

1. `README.md`
2. `docs/vc4asm-notes.md`
3. `docs/reference-links.md`
4. `code/3-saxpy/README.md`
5. `code/3-saxpy/saxpy.qasm`

When using VC4C as prior art, start with:

1. `third_party/VC4C/Readme.md`
2. targeted backend files relevant to the task
   Examples:
   - `third_party/VC4C/src/periphery/VPM.h`
   - `third_party/VC4C/src/intermediate/MemoryInstruction.cpp`
   - `third_party/VC4C/src/intrinsics/WorkItems.cpp`

## Repo roles
- `compiler/`: the new MLIR-based compiler work. This is where new backend design should live.
- `code/`: handwritten and runtime-side VC4 experiments. Treat these as executable ground truth and reference behavior.
- `docs/vc4asm-notes.md`: compact backend-shaping guidance for the staged `vc4` dialect.
- `docs/reference-links.md`: authoritative external references for `vc4asm` details.
- `third_party/VC4C/`: reference-only prior art for backend facts, lowering ideas, runtime conventions, and machine constraints.

## Core architectural stance
Keep these distinctions explicit:

- frontend semantics vs backend machine structure
- MLIR GPU/vector lowering vs VC4-specific staged-memory lowering
- staged `vc4` IR vs raw `vc4asm` syntax/details
- project-owned compiler architecture vs reference-only ideas from VC4C

Do not collapse these layers because a handwritten kernel already exists.

## VC4 backend model
The custom `vc4` dialect should model machine structure first, especially:

- uniforms
- DMA memory <-> VPM staging
- VPM read/write structure
- vector arithmetic on width-16 execution groups
- later: control flow, thread end, and scheduling-sensitive details

It should not initially model:

- raw `vc4asm` directives or helper macro syntax
- bitfield encodings
- exact register allocation choices
- delay-slot filling
- hazard scheduling policy

Rule of thumb:
model what the machine is doing, not how `vc4asm` happens to spell it.

## What the handwritten kernels are for
The handwritten kernels under `code/` are important because they show what successful VC4 execution actually looks like:

- uniform-driven launch parameters
- QPU width-16 SIMD execution
- DMA staging into VPM
- VPM-mediated compute
- explicit thread termination
- practical launch/runtime conventions already working in this repo

Use them to validate backend concepts and emission expectations.
Do not force the compiler IR to mirror their assembly syntax one-to-one.

For the current milestone, `code/3-saxpy/saxpy.qasm` is the most important concrete reference.

## What VC4C is for
`third_party/VC4C` matters, but as prior art only.

Use it to study:
- backend facts about VC4/QPU behavior
- work-item/work-group conventions
- memory and VPM handling ideas
- lowering boundaries that turned out to matter in a real compiler
- late backend concerns such as register pressure, hazards, and scheduling

Do not:
- modify `third_party/VC4C` unless explicitly asked
- copy its structure blindly into `compiler/`
- let its OpenCL-specific frontend/runtime assumptions dictate this project

This project’s contribution is the MLIR `gpu`/`vector` -> VC4 backend path and the experiments built on top of it.

## Near-term compiler expectations
Early compiler work is allowed to be narrow and artificial if it preserves architectural clarity.

Acceptable right now:
- hardcoded or handwritten MLIR test inputs
- toy SAXPY-shaped kernels
- text dumps and schematic emission
- partial lowering passes that make the stage boundary visible
- minimal custom dialect surface area

Lower priority right now:
- a full CUDA parser
- a polished runtime API
- broad optimization work
- wide kernel coverage
- feature-complete target modeling

## Scope guardrails
Unless the repo docs are updated to say otherwise, do not add:
- `__shared__`
- barriers or synchronization
- atomics
- texture/TMU-heavy abstractions as the main path
- broad CUDA compatibility claims
- a large custom target dialect that anticipates every future feature

Assume a narrow kernel subset and earn each expansion with a concrete use case.

## Planning and change discipline
For multi-step compiler work:
- keep one main task per session when possible
- leave short session notes and next steps in the most relevant nearby doc when significant changes are made
- if no suitable project doc exists yet, add a small compiler-local note instead of inventing a large tracking structure

Prefer:
- small, reversible patches
- docs and tests in the same change for nontrivial work
- scripts in `scripts/` for reproducible checks
- lightweight text-based tests early on

If a workflow does not exist yet, add a small script or stub before building a lot of machinery around it.

## What good progress looks like right now
Good near-term changes make one of these clearer:
- how a `gpu`/`vector` representation maps to VC4 staged memory
- how width-16 execution groups are represented before emission
- how uniforms, DMA, VPM, and arithmetic appear in the custom `vc4` dialect
- how a SAXPY-like kernel can lower end to end without ad hoc special cases

Bad near-term changes:
- skipping straight from high-level IR to handwritten assembly strings
- designing around one kernel in a way that cannot generalize
- importing raw assembler concepts too early
- spending time on late backend scheduling before the staged IR is stable
- broadening scope without updating the written guidance

## VC4 realities to keep in mind
- QPU execution is effectively width-16 SIMD.
- VPM is the central staging mechanism for the current backend model.
- Real kernels have hazards, delay slots, special registers, and explicit thread termination.
- Those details matter, but most belong below the first staged `vc4` IR layer.
- The backend must remain honest about the machine even when the frontend subset is tiny.

## Default decision rule
When in doubt, choose the change that makes the staged MLIR-to-VC4 path more explicit, testable, and extensible for SAXPY without dragging raw assembler detail upward.
