---

### `docs/architecture.md`

```md
# Architecture

## Overview
This project is a restricted-CUDA frontend with VC4-native lowering.

The architecture is intentionally staged so that:
- CUDA-like source structure is preserved early
- width-16 execution becomes explicit in the middle
- VC4-specific details only appear late

## Compiler pipeline

### 1. Frontend / scalar kernel layer
Purpose:
- represent a tiny CUDA subset in a clean, target-independent form

Expected MLIR dialects:
- `func`
- `arith`
- `scf`
- likely `memref`

Responsibilities:
- scalar arithmetic
- scalar control flow
- builtins like `threadIdx.x`, `blockIdx.x`, `blockDim.x`, `gridDim.x`
- simple loads/stores

This layer should still look conceptually like a scalar-thread kernel language.

### 2. Vec16 layer
Purpose:
- make the real execution model explicit

Expected MLIR dialect:
- `vector`

Responsibilities:
- width-16 grouping
- lane IDs
- vector arithmetic
- masks / predication
- vectorized control-flow lowering strategy
- logical mapping from scalar CUDA threads to 16-lane execution groups

This is the most important architectural boundary in the compiler.

### 3. VC4 layer
Purpose:
- express target-specific concepts that no longer fit cleanly in generic MLIR dialects

Representation:
- custom `vc4` dialect

Responsibilities:
- low-level target operations
- explicit staged target-side memory operations
- backend-oriented control flow
- target-specific attributes and constraints
- later: hooks for hazards, scheduling boundaries, and special resource usage

The `vc4` dialect is intended to be the final structured IR before text
emission. Early on, it should make staged memory movement through backend
resources such as VPM visible without trying to model the full machine.

### 4. Emission layer
Purpose:
- emit `vc4asm` text first

Responsibilities:
- convert `vc4` dialect operations into assembly-oriented text
- provide readable dumps for debugging
- eventually reflect target constraints clearly

## Design rules

### Rule 1: keep frontend and backend semantics separate
Do not let VC4-specific constraints leak into the scalar kernel layer unless absolutely necessary.

### Rule 2: vectorization is an explicit compiler stage
Do not treat width-16 behavior as an implementation detail hidden everywhere.
It should appear as a clear lowering boundary.

### Rule 3: keep the custom target dialect small
The custom `vc4` dialect should start minimal.
Only move concepts into it when generic MLIR dialects stop being helpful.

### Rule 4: optimize for inspectability early
Readable IR dumps and testable lowering passes matter more than early performance work.

## Proposed repository structure

```text
compiler/
README.md
mlir/
include/pi_gpu/
VC4/
Conversion/
lib/
VC4/
Conversion/
test/
