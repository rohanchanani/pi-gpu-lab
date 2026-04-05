# vc4asm notes

This file is a compact working reference for backend development.
It is not the source of truth. The authoritative references are in
`docs/reference-links.md`.

## Purpose
Keep the compiler IR above raw `vc4asm` syntax for as long as possible.

The `vc4` dialect should express:
- staged memory movement
- VPM reads/writes
- simple arithmetic
- later, control-flow structure

The dialect should **not** initially encode:
- raw instruction bitfields
- exact `vc4asm` macro syntax
- directive-level assembler details
- delay-slot filling or scheduling details

Those belong to later lowering / emission stages.

## Current backend mental model
For simple kernels like SAXPY, the backend shape is:

1. read uniforms
2. stage vectors from memory into VPM
3. read staged vectors from VPM
4. do arithmetic
5. write result back to VPM
6. DMA staged result back to memory
7. later: loops / branching / thread end

This matches the handwritten SAXPY kernel more closely than a generic
"target launch" marker.

## Useful vc4asm concepts

### Uniforms
Typical kernels begin by reading launch/runtime values from `unif`.
Examples:
- base addresses
- scalar parameters
- qpu count / qpu index
- iteration count

Compiler implication:
- keep a backend op like `vc4.uniform`
- do not lower directly to raw `mov ..., unif` too early

### VPM
VPM is the visible staging area between memory and QPU-side computation.

Typical pattern:
- DMA memory -> VPM
- VPM -> QPU register/value
- compute
- QPU register/value -> VPM
- DMA VPM -> memory

Compiler implication:
- model `dma_load_vpm`, `vpm_read`, `vpm_write`, `dma_store_vpm`
- these are good first backend concepts

### VDR / VDW helpers
The assembler provides helper macros/forms for DMA setup, such as:
- `vdr_setup_0(...)`
- `vdw_setup_0(...)`
- `vdr_h32(...)`
- `dma_h32(...)`

Compiler implication:
- do not encode these exact forms into the dialect yet
- keep the dialect one level above them
- later emission can choose the proper helper form

### VPM setup helpers
The assembler also provides helpers like:
- `vpm_setup(...)`

Compiler implication:
- row / layout / staging-location hints may eventually appear as simple
  dialect attributes
- exact macro syntax should remain an emission concern for now

### Arithmetic
Simple floating-point arithmetic in the handwritten SAXPY kernel is just:
- `fmul`
- `fadd`

Compiler implication:
- early backend IR can use either:
    - a generic arithmetic op with a small kind attribute, or
    - a few distinct arithmetic ops
- do not overbuild yet

### Branches and delay slots
Branches exist and have delay-slot concerns in real QPU code.

Compiler implication:
- do not model delay-slot filling or exact branch encoding yet
- keep scheduling and final instruction shaping for a later stage

### `thrend` / end-of-thread behavior
Real kernels terminate explicitly.

Compiler implication:
- final thread-end ops belong in a later backend stage, not in the first
  staged-memory seed unless needed

## Guidance for current compiler work

### Good vc4 dialect concepts right now
- uniform load
- staged DMA load into VPM
- staged read from VPM
- arithmetic
- staged write to VPM
- staged DMA store out of VPM

### Bad vc4 dialect concepts right now
- SAXPY-specific ops
- raw assembler directives
- raw `vc4asm` macro strings
- bitfield encodings
- register allocation details
- branch delay slot handling

## Rule of thumb
Model the **machine structure**, not the **assembler syntax**.

Good:
- "load a vector into VPM row X"
- "read a staged vector"
- "store a staged vector back"

Too low-level for now:
- "emit exactly this `vdr_setup_0(...)` form"
- "set this exact setup bitfield"

## Expected near-term path
Near-term compiler flow should look like:

vec16 kernel IR
-> tiny staged `vc4` IR
-> later: more detailed/scheduled backend IR
-> later: emitted `vc4asm`