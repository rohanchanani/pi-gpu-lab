# Project scope

## Goal
Build a restricted CUDA-subset compiler and runtime for the Raspberry Pi VideoCore IV GPU (VC4), targeting QPUs and emitting `vc4asm` first.

This is **not** a full CUDA implementation. The project uses CUDA-like source syntax as a frontend while lowering to a VC4-native execution model.

## Core design stance
- Source language: tiny CUDA subset
- Internal execution model: width-16 vectorized groups mapped to VC4 QPUs
- Backend target: `vc4asm` first
- Initial emphasis: extensible architecture and end-to-end lowering, not feature breadth

## Non-goals for v1
The following are explicitly out of scope for v1:
- full CUDA semantics
- full CUDA parser
- `__shared__`
- `__syncthreads()`
- atomics
- dynamic allocation
- recursion
- function pointers
- advanced pointer aliasing semantics
- texture/surface language features
- production-quality optimization
- LLVM backend or machine code generation

## v1 frontend subset
Supported kernel-level constructs:
- `__global__` kernel concept
- 1D grids and 1D blocks only
- `threadIdx.x`
- `blockIdx.x`
- `blockDim.x`
- `gridDim.x`
- `int`, `float`, flat pointers to buffers
- arithmetic
- comparisons
- `if`
- simple counted loops
- simple indexed loads/stores such as:
    - `x[i]`
    - `y[i]`
    - `x[i + c]`

## v1 execution model
The frontend presents a scalar-thread view, but the backend does not execute scalar CUDA threads directly.

Semantic contract:
- one source CUDA thread = one logical scalar instance
- backend packs 16 logical threads into one vector execution group
- one vector execution group corresponds to one QPU-side unit of execution
- multiple such groups may later be distributed across QPUs
- control flow will ultimately be lowered through vector masks / predication

## v1 memory model
- source pointers refer to flat global buffers
- v1 does not expose `__shared__`
- v1 does not expose block-wide synchronization
- memory movement details are backend concerns
- initial kernels may lower through simplified memory paths before richer VC4-specific tiling support is added

## IR direction
The intended compiler pipeline is:

tiny CUDA subset
-> MLIR scalar dialects (`func`, `arith`, `scf`, likely `memref`)
-> MLIR `vector` dialect with width-16 vector groups
-> custom `vc4` dialect
-> `vc4asm`

## First milestone
First real milestone:
- represent a SAXPY-like kernel
- lower it through the compiler stack
- produce either:
    - readable pseudo-assembly, or
    - `vc4asm`-oriented emission

Example target kernel:

```cpp
__global__ void saxpy(float* x, float* y, float a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a * x[i] + y[i];
  }
}