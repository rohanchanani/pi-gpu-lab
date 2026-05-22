# VC4Tile Architecture and Ecosystem Integration Plan

**Status:** planning/design artifact for the next VC4 frontend milestone.  
**Project state assumed:** `ssavc4` exists as the pre-register-allocation / pre-scheduling VC4 machine SSA layer, `vc4` remains the scheduled QASM-near sink, and the current runtime/artifact path can execute compiled VC4 kernels through `vc4_runtime`.  
**Locked dialect name:** **VC4 Tile dialect**, mnemonic **`vc4tile`**.

---

## 0. Executive summary

The next major compiler artifact should be a new MLIR dialect named:

```text
Dialect name:      VC4 Tile dialect
Mnemonic:          vc4tile
C++ namespace:     ::mlir::vc4tile
ODS prefix:        VC4Tile
Recommended path:  compiler/include/vc4/Dialect/VC4Tile/IR/
                   compiler/lib/Dialect/VC4Tile/IR/
```

`vc4tile` is the project’s VC4-specific **tile-kernel IR**. It sits between external kernel-producing ecosystems and the existing VC4 lower half:

```text
Triton TTIR / IREE late executable IR
  ↓
vc4tile       // VC4 Tile dialect: explicit tile/kernel IR
  ↓
ssavc4        // VC4 SSA machine IR, pre-RA/pre-scheduling
  ↓
vc4           // scheduled QASM-near sink dialect
  ↓
QASM / C / H artifacts / vc4_runtime
  ↓
VideoCore IV hardware
```

The most important design decision is that the shared producer-facing boundary should **not** be raw upstream `gpu`, raw Triton TTIR, raw IREE executable IR, or `ssavc4`. It should be a new target-specific dialect: `vc4tile`.

`vc4tile` is a compiler-facing tile IR with CUDA-like kernel/block concepts, Triton-like masked tile/vector memory semantics, and VC4-specific resource semantics. It is intentionally higher than `ssavc4`, because it should not expose TMU request/read tokens, VDW setup details, semaphore protocols, physical register allocation, scheduler hazards, or QASM details. It is intentionally lower than IREE/Triton source IRs, because it has already committed to VC4’s 16-lane QPU tile, VC4 memory/resource constraints, and the project’s launch/resource model.

The core rule is:

```text
Do not write:
  Triton → SSAVC4
  IREE   → SSAVC4

Write:
  Triton TTIR subset       → vc4tile
  IREE late executable IR  → vc4tile
  vc4tile                  → ssavc4
```

That is the anti-duplication boundary. The expensive VC4-specific lowering work is implemented once: `vc4tile → ssavc4 → vc4 → artifacts/runtime`.

---

## 1. Evidence base and source of truth

This plan is grounded in four classes of evidence.

### 1.1 Current VC4 runtime and launch ABI

The current runtime header defines:

```c
#define VC4_RUNTIME_MAX_QPUS 12u
#define VC4_RUNTIME_LANE_WIDTH 16u

#define VC4_KERNEL_SCHEDULE_INDEPENDENT_VECTOR 1u
#define VC4_KERNEL_SCHEDULE_COOPERATIVE_BLOCK 2u

#define VC4_KERNEL_FLAG_COOPERATIVE_BLOCK 0x00000001u
#define VC4_KERNEL_FLAG_REQUIRE_FULL_BLOCK_RESIDENCY 0x00000002u
#define VC4_KERNEL_FLAG_USES_BARRIER 0x00000004u
#define VC4_KERNEL_FLAG_USES_SHARED_VPM 0x00000008u
```

It also defines `vc4_kernel_image` fields for schedule mode, flags, maximum warps per block, semaphore usage, VPM bytes/rows per block, max resident blocks, and spill-frame metadata. It defines `vc4_launch_request_info` fields for logical request, logical block ID, logical warp ID, VPM base row, barrier semaphore IDs, resident request ID, and spill-frame base/stride. That runtime API is not merely an independent-vector host shim. It already encodes independent-vector kernels, cooperative-block kernels, shared VPM resource allocation, barrier semaphore assignment, and spill-frame launch metadata.

The current runtime implementation validates kernel descriptors, computes persistent program layout, uploads each kernel’s code once into a persistent program allocation, precomputes uniform-pointer arrays, supports dynamic VC4 heap allocation/copies, and launches kernels through `vc4LaunchKernel`. It computes independent-vector waves and cooperative-block resident waves differently. For cooperative-block launches it computes resident blocks from QPU slots, VPM rows/bytes, semaphore counts, and max-resident-block constraints, then fills each request’s logical block/warp IDs, VPM rows, barrier semaphores, and spill-frame fields before enqueueing SRQPC/SRQUA pairs.

### 1.2 VC4-as-CUDA mapping guide

The current mapping guide’s core conclusion is:

```text
Model the Raspberry Pi VC4 GPU as one tiny CUDA-like SM
with 12 physical warp slots, 16 SIMD lanes per warp,
and one global 4 KiB user-visible shared-memory window
backed by VPM.
```

It also states:

```text
CUDA warp             → one QPU user-program request, 16 SIMD lanes
CUDA lane             → one QPU SIMD element
CUDA thread block     → group of 1..12 logical QPU warps
CUDA shared memory    → allocation in the single global 4 KiB VPM window
CUDA __syncthreads()  → validated four-semaphore reusable QPU barrier
global load           → TMU direct memory lookup
global store          → QPU register → VPM → VDW DMA store
```

The guide emphasizes that physical `QPU_NUMBER` must not be used as logical warp identity, that the VPM window is global rather than per slice, and that barrier-enabled blocks require full residency.

### 1.3 SSAVC4 / scheduled VC4 boundary

The M3 plan defines the lower-half boundary:

```text
ssavc4 = pre-register-allocation / pre-scheduling SSA target machine IR
vc4    = post-register-allocation / post-scheduling artifact sink IR
```

It also says M3 is not a frontend producer-lowering milestone. The current milestone boundary is M4 `vc4tile → ssavc4`; later Triton/IREE/MLIR-GPU-like producer adapters should target `vc4tile` rather than bypassing into SSAVC4 or scheduled `vc4`.

Parts of the historical M3 plan are now stale: the plan described a no-spill initial policy, while the current lower half includes spill-frame metadata, spilling, and block-argument / phi-like edge-copy lowering. The stable part of the M3 evidence is the dialect boundary: `ssavc4` is the correct target machine-SSA sink below `vc4tile`, and scheduled `vc4` remains the QASM-near sink.

### 1.4 Upstream ecosystem evidence

Official upstream documentation supports the planned producer strategy:

- MLIR’s `gpu` dialect is a middle-level CUDA/OpenCL-like abstraction with `gpu.module`, `gpu.func`, and `gpu.launch_func`; kernel outlining lowers `gpu.launch` to `gpu.launch_func` in a dedicated `gpu.module`. This is useful conceptually and sometimes syntactically, but raw upstream `gpu` is not the whole PyTorch/JAX/Triton ecosystem boundary.
- IREE provides the most relevant PyTorch/JAX compiler middle-end. IREE’s HAL dialect represents hardware abstraction concepts such as buffers, views, synchronization primitives, command buffers, executable dispatch, and devices. Its HAL translation passes translate `hal.executable.variant` ops for a target backend from generic MLIR dialects such as `linalg` to target-specific dialects such as LLVM or SPIR-V. For VC4, the target-specific sink should be `vc4tile`/`ssavc4`, not NVVM/ROCDL/SPIR-V.
- IREE Codegen documentation says a backend analyzes entry-point functions inside `hal.executable.variant`, chooses the compilation pipeline, and decides parameters such as tile sizes. That means VC4 target configuration is part of the compiler-side work.
- Triton has its own MLIR dialect stack. Published Triton compilation material shows TTIR containing tile-level operations such as `tt.get_program_id`, `tt.make_range`, pointer arithmetic, `tt.load`, arithmetic, and `tt.store` with masks. Later Triton-GPU IR can already contain target-specific CUDA/AMD layout information, so the first VC4 adapter should consume TTIR or an early target-neutral Triton form rather than late CUDA-shaped TTGIR.
- StableHLO is a portability layer between ML frameworks and compilers. JAX can export StableHLO, IREE consumes StableHLO, and IREE/Turbine integrates PyTorch, Torch-MLIR, and IREE. Therefore the PyTorch/JAX path should be through IREE, not through a custom Torch/JAX frontend written directly for VC4.

Reference links are collected in [Appendix C](#appendix-c-upstream-reference-links).

---

## 2. Locked naming decision

### 2.1 Name

The dialect is named:

```text
VC4 Tile dialect
```

with mnemonic:

```mlir
vc4tile
```

Example operations:

```mlir
vc4tile.kernel
vc4tile.program_id
vc4tile.block_id
vc4tile.warp_id
vc4tile.lane_range
vc4tile.masked_load
vc4tile.masked_store
vc4tile.shared_alloc
vc4tile.shared_load
vc4tile.shared_store
vc4tile.barrier
vc4tile.reduce
vc4tile.rotate
```

### 2.2 Why not `VC4KernelInput`

`VC4KernelInput` is a useful design-role phrase, but it is not a good dialect name. It names the dialect by how the current milestone consumes it rather than by what the dialect is. Dialect names should be stable beyond a single milestone.

Use `VC4KernelInput-v0` only as a profile/spec phrase if needed:

```text
VC4KernelInput-v0 = the initial accepted subset/profile of the vc4tile dialect.
```

The dialect itself is `vc4tile`.

### 2.3 Why not `vc4k`

`vc4k` is compact, but opaque. A reader seeing:

```mlir
vc4k.masked_load
```

has to learn that `k` means kernel. A reader seeing:

```mlir
vc4tile.masked_load
```

immediately sees the level of abstraction: this is a VC4 tile-level kernel IR, not scheduled QPU code and not generic upstream `gpu`.

### 2.4 Why `vc4tile` is not misleading

The dialect will contain more than “tile math” operations: it will contain kernel entry ops, launch/resource metadata, program/block/warp/lane ID ops, global memory ops, cooperative block concepts, shared VPM concepts, barriers, reductions, and rotations.

That does not make the name misleading, because “tile” is the dialect’s central abstraction:

```text
one vc4tile lane tile     = one 16-lane QPU SIMD vector
one vc4tile warp tile     = one logical QPU request
one vc4tile block tile    = 1..12 cooperating QPU requests
one vc4tile shared tile   = VPM rows/columns backing block-local data
```

In this dialect, “tile” means a VC4 execution/memory tile, not generic high-level tensor tiling. The prefix `vc4` is important: this is not a generic `tile` dialect and not NVIDIA CUDA Tile IR. It is the VC4-specific tile dialect.

---

## 3. Core project decision

### 3.1 The desired ecosystem shape

The long-term ecosystem goal is:

```text
JAX
  → StableHLO
  → IREE Flow/Stream/HAL + IREE codegen
  → IREE VC4 compiler target
  → vc4tile
  → ssavc4
  → vc4
  → artifacts/runtime
  → hardware

PyTorch
  → Torch-MLIR / IREE Turbine / possibly StableHLO
  → IREE Flow/Stream/HAL + IREE codegen
  → IREE VC4 compiler target
  → vc4tile
  → ssavc4
  → vc4
  → artifacts/runtime
  → hardware

Triton
  → TTIR or early target-neutral Triton MLIR
  → Triton-to-vc4tile adapter
  → vc4tile
  → ssavc4
  → vc4
  → artifacts/runtime
  → hardware
```

### 3.2 The central minmax / greatest-useful-common-boundary decision

`vc4tile` should be the highest useful common target for both:

```text
Triton TTIR
IREE late executable/codegen IR
```

and the lowest useful producer-facing source for the existing lower half:

```text
ssavc4 → vc4 → artifacts/runtime
```

This does not mean there is an existing mathematical greatest-lower-bound dialect shared by Triton and IREE. There is not. Triton and IREE are separate compiler stacks. The design choice is to create a **small VC4-specific tile-kernel contract** that both stacks can lower into mechanically.

### 3.3 Why not raw upstream `gpu`

Raw upstream `gpu` is not the right sole dialect boundary.

It is useful because it gives CUDA/OpenCL-like concepts such as:

```text
gpu.module
gpu.func
gpu.launch_func
gpu.thread_id
gpu.block_id
gpu.block_dim
gpu.grid_dim
gpu.barrier
workgroup/private/global address spaces
```

However:

1. PyTorch and JAX do not naturally converge directly at raw `gpu`; they more naturally enter IREE through StableHLO, Torch-MLIR/Turbine, Linalg/TOSA, and IREE’s Flow/Stream/HAL/codegen pipeline.
2. Triton uses TTIR/TTGIR dialects rather than upstream `gpu` as its primary compiler stack.
3. IREE late executable IR may contain `scf`, `memref`, `vector`, IREE codegen attributes/ops, IREE GPU helper dialects, and target configuration rather than a clean upstream `gpu.module`.
4. Upstream `gpu` is scalar-SIMT-shaped, while VC4 is physically 16-lane SIMD. A faithful VC4 boundary should expose vector/tile/mask semantics directly.

Therefore, `gpu` is a useful source of concepts and sometimes a useful normalization dialect, but the project’s shared boundary should be `vc4tile`.

### 3.4 Why not `ssavc4`

`ssavc4` is too low for the shared producer boundary. It contains target machine concepts such as TMU/VPM/VDW/semaphore/barrier effects, launch/resource metadata, SSA dataflow, and QPU-machine lowering hooks. External adapters from IREE/Triton should not need to know about TMU request tokens, VDW setup details, semaphore protocols, QPU mutex details, spill uniforms, or machine scheduling hazards.

The producer boundary should say:

```text
masked_load from global buffer at vector offsets
masked_store to global buffer at vector offsets
shared tile allocation/load/store
barrier
reduce/rotate
program/block/warp/lane IDs
```

The `vc4tile → ssavc4` lowering should decide:

```text
TMU request/read sequencing
VPM row allocation details
VDW store lowering
semaphore barrier protocol
mutex serialization
spill-frame interaction
ssavc4 block arguments and control-flow lowering
```

### 3.5 Why not IREE-only

IREE is the right path for JAX/PyTorch, but not for Triton kernels. Triton is already a kernel DSL/compiler with tile-level IR. Forcing Triton through IREE first would introduce unnecessary coupling and likely lose useful Triton structure.

The right split is:

```text
IREE producer adapter  → vc4tile
Triton producer adapter → vc4tile
```

not:

```text
Triton → IREE → vc4tile
```

---

## 4. Semantic position of `vc4tile`

### 4.1 Level of abstraction

`vc4tile` is:

```text
post-dispatch
post-bufferization
post-tiling/distribution where possible
pre-SSAVC4
pre-register-allocation
pre-scheduling
pre-QASM
```

It should contain exactly one kernel/dispatch body at a time, with explicit buffers, scalar arguments, launch IDs, tile/lane values, masks, memory operations, and block/shared-memory resources.

### 4.2 Comparison to CUDA, Triton, cuTile, and ThunderKittens

The best analogy is:

```text
vc4tile is a VC4-specific tile-kernel IR,
spiritually closer to CUDA Tile IR / cuTile / CuTe / ThunderKittens
than to scalar CUDA C,
but compiler-facing rather than user-facing.
```

It is not “above CUDA” or “below CUDA” in a strict linear sense. CUDA C is a scalar SIMT source language; `vc4tile` is a vector/tile compiler IR. The right comparison is:

```text
CUDA C:
  scalar thread program with implicit warp execution and compiler-managed SIMT behavior

Triton:
  tile/block kernel DSL with blocked tensor values and masks

vc4tile:
  target-specific tile-kernel IR with explicit 16-lane VC4 tiles,
  masks, program/block/warp/lane IDs, coalesced global memory semantics,
  cooperative VPM/barrier resources, and direct lowering to SSAVC4
```

For VC4, the tile abstraction is not decorative. The physical QPU lane width is 16, and the VPM row width is naturally 16 32-bit words. A 16-lane vector tile is both an execution unit and a memory-layout unit.

---

## 5. Core execution model

### 5.1 Device model

The target model used by `vc4tile` is:

```text
one CUDA-like SM
12 physical QPU warp slots
16 lanes per QPU warp
192 max logical resident threads
one global 4 KiB VPM shared-memory window
16 hardware semaphores
persistent program image with resident kernel code
```

`vc4tile` should never model VC4 slices as separate SMs. Slices matter for performance, but they are not independent shared-memory domains.

### 5.2 Kernel model

A `vc4tile.kernel` represents one compiled kernel entry point.

The kernel has:

```text
device/global buffer arguments
scalar arguments
launch geometry metadata
resource metadata
one region containing tile code
```

The region represents the work done by one logical QPU request / one 16-lane warp tile, or by one warp tile within a cooperative block.

### 5.3 Independent-vector mode

Independent-vector mode is used for kernels with no cross-warp cooperation:

```text
no shared VPM
no block barrier
no block-level cooperation
```

Each logical request processes a vector tile:

```text
logical_request = program_id / tile ID
lane            = 0..15
global element  = logical_request * 16 + lane
```

This is appropriate for:

```text
elementwise kernels
SAXPY/vector add
simple maps
coalesced masked loads/stores
separate-pass reductions
simple affine tiles with no shared memory
```

### 5.4 Cooperative-block mode

Cooperative-block mode is used for kernels with:

```text
shared VPM
barriers
cross-warp reductions
block-level tiling
matmul/transpose-like cooperative tiles
```

A cooperative block contains:

```text
1..12 logical QPU warp tiles
```

The runtime must schedule all warps of a barrier-participating block as a resident group. Each resident block gets:

```text
a VPM row range
a semaphore range
logical block ID
warps_per_block
logical warp IDs 0..warps_per_block-1
```

### 5.5 Lane model

The canonical lane value is:

```text
lane = 0..15
```

Lowering to `ssavc4` uses `ssavc4.element_number` or equivalent as the source of the actual QPU lane ID.

### 5.6 Thread-like IDs

`vc4tile` may expose CUDA-like ID helpers, but they are derived from tile semantics:

```text
warp_id_in_block = uniform logical_warp_id
lane_id          = lane_range element / ELEMENT_NUMBER
flat_thread_id   = warp_id_in_block * 16 + lane_id
```

For 1-D blocks:

```text
thread_id.x = flat_thread_id
```

For 3-D blocks:

```text
thread_id.x = flat_thread_id % blockDim.x
thread_id.y = (flat_thread_id / blockDim.x) % blockDim.y
thread_id.z = flat_thread_id / (blockDim.x * blockDim.y)
```

`vc4tile` should not use physical `QPU_NUMBER` for logical identity.

---

## 6. Dialect contract

### 6.1 Required top-level operation

The dialect should define a kernel operation:

```mlir
vc4tile.kernel @name(%args...) attributes {
  ...
} {
  ...
}
```

or, if it is more convenient to integrate with existing MLIR function infrastructure, an operation compatible with `FunctionOpInterface`, such as:

```mlir
vc4tile.func @name(%args...) attributes {vc4tile.kernel = true} {
  ...
}
```

Recommendation: use a function-like kernel op named `vc4tile.kernel` if possible. The dialect’s purpose is kernel IR; making the top-level op explicit clarifies the boundary.

### 6.2 Recommended attributes

A `vc4tile.kernel` should carry or derive:

```text
public_name
schedule_mode                      independent_vector | cooperative_block
warp_size                          fixed 16
max_warps_per_block
block_dim / launch shape policy
uses_shared_vpm
uses_barrier
requires_full_block_residency
vpm_rows_per_block
vpm_bytes_per_block
semaphores_per_block
max_resident_blocks
tail_policy                        exact_multiple | masked_tail
uniform_abi profile
spill policy / spill metadata passthrough
```

Some of these may initially be canonicalized into existing `vc4.launch_abi` and `vc4.resource` dictionaries during `vc4tile → ssavc4`.

### 6.3 Types

Core value types:

```text
i1                       scalar boolean, mostly uniform
i32                      scalar integer/uniform
f32                      scalar float/uniform
index                    accepted only with explicit i32 lowering policy
vector<16xi1> or mask    lane mask
vector<16xi32>           16-lane integer tile
vector<16xf32>           16-lane float tile
```

The core lowering form should operate on `vector<16xT>` chunks.

The outer dialect may optionally allow larger static tile values:

```text
vector<Nxi32>
vector<Nxf32>
```

where `N` is a static multiple of 16. A shared stripmine pass should lower those to `vector<16xT>` chunks before the core `vc4tile → ssavc4` lowering. This is important for Triton, because Triton kernels often use block sizes larger than 16.

Initial unsupported types:

```text
i64
f64
f16/bf16 unless explicitly added
i8/i16 arithmetic unless packed/widened policy exists
dynamic/ragged vector lengths
opaque producer-specific tensor types
```

### 6.4 Values and uniformity

`vc4tile` should explicitly distinguish:

```text
uniform scalar values
varying lane/vector values
masks
```

Uniform values include:

```text
kernel scalar args
buffer base pointers
program/block IDs
grid/block dims
warps_per_block
VPM/semaphore metadata
constants
```

Varying values include:

```text
lane_range
thread_id
global_id
vector memory offsets
per-lane loaded data
vector arithmetic results
```

When uniform values are used with varying values, the lowering should splat them to `vector<16xT>`.

### 6.5 Masks

Masks are first-class.

Masks are used for:

```text
tail-safe loads
tail-safe stores
select/predication
bounds checks
partial final warp participation
```

The dialect should avoid pretending that divergent SIMT control flow is native. Per-lane decisions should become masks/selects/predicated memory operations whenever possible.

### 6.6 Control flow

Supported control flow should be divided into:

```text
uniform control flow:
  normal branches/loops where the condition is block-/warp-uniform

masked dataflow:
  per-lane conditions represented as masks and selects

unsupported initially:
  arbitrary divergent side-effecting control flow
  divergent barrier participation
  native SIMT reconvergence semantics
```

`vc4tile` may use standard `cf`/`scf` where it is legal and verifier-constrained, or define its own branch ops if that simplifies lowering. Since current `ssavc4` supports block arguments / phi-like edge operands, `vc4tile → ssavc4` should use that mechanism rather than inventing public phi-like SSAVC4 operations.

---

## 7. Core operations

This section names operations conceptually. Exact ODS spelling can be adjusted, but the semantic surface should remain stable.

### 7.1 Kernel and launch identity

```mlir
vc4tile.program_id     // logical independent-vector request or logical block tile
vc4tile.block_id       // logical cooperative/grid block ID
vc4tile.grid_dim
vc4tile.block_dim
vc4tile.warp_id        // logical warp ID within cooperative block
vc4tile.warps_per_block
vc4tile.lane_range     // vector<16xi32> = [0, 1, ..., 15]
vc4tile.thread_id      // derived CUDA-like thread ID, optional helper
vc4tile.global_id      // derived block_id * block_dim + thread_id, optional helper
```

Recommendation: include both primitive and derived forms, but canonicalize derived forms to primitives plus arithmetic before lowering to SSAVC4.

### 7.2 Global memory

```mlir
vc4tile.masked_load_global
vc4tile.masked_store_global
```

Conceptual shape:

```mlir
%v = vc4tile.masked_load_global %base[%offsets], %mask
       : !vc4tile.buffer<f32>, vector<16xi32>, vector<16xi1> -> vector<16xf32>

vc4tile.masked_store_global %base[%offsets], %value, %mask
       : !vc4tile.buffer<f32>, vector<16xi32>, vector<16xf32>, vector<16xi1>
```

Initial global memory policy:

```text
loads:
  per-lane 32-bit loads through TMU direct memory lookup

stores:
  coalesced/affine vector stores through VPM/VDW

reject initially:
  arbitrary scatter stores
  atomics
  uncoalesced byte stores
```

The store verifier should know the difference between a coalesced affine store and arbitrary vector scatter. If a future slow path exists, it should be explicit and not silently selected.

### 7.3 Shared VPM memory

```mlir
vc4tile.shared_alloc
vc4tile.shared_load
vc4tile.shared_store
```

The dialect should expose shared memory as **VC4 shared tiles**, not as arbitrary CUDA shared scalar SRAM.

Supported initial patterns:

```text
row-granular 32-bit shared tiles
affine row/column indexing
one row = one 16-lane vector where possible
block reductions using row staging
transpose-like patterns using horizontal/vertical VPM access
```

Rejected initially:

```text
arbitrary per-lane shared scatter/gather
shared-memory atomics
byte-addressed alias-heavy shared memory
dynamic shared memory
```

A shared allocation must carry enough information to compute:

```text
vpm_rows_per_block
vpm_bytes_per_block
hidden compiler rows if required
```

### 7.4 Barrier

```mlir
vc4tile.barrier
```

Semantics:

```text
block-wide synchronization among all logical QPU warps in a cooperative block
memory ordering for block-shared VPM accesses before/after the barrier
requires uniform participation by all logical warps in the block
requires full residency
```

Lowering:

```text
vc4tile.barrier
  → ssavc4 barrier/semaphore sequence
  → validated four-semaphore reusable QPU barrier
```

The producer should never emit raw semaphore ops for normal barriers.

### 7.5 Warp-local operations

```mlir
vc4tile.rotate
vc4tile.reduce
```

These represent lane/subgroup-local operations over the 16-lane QPU tile.

Examples:

```text
rotate vector lanes by constant/supported offset
reduce_add over 16 lanes
reduce_max / reduce_min if supported later
mask reductions if needed
```

This tier should be exposed early because it is central to reductions and because SSAVC4 already has rotate/reduction-oriented support.

### 7.6 Arithmetic

Use standard MLIR `arith` where possible:

```text
arith.addi / subi / muli
arith.addf / subf / mulf
arith.cmpi / cmpf
arith.select
arith.constant
```

The dialect should not duplicate ordinary arithmetic unless VC4-specific semantics require it.

### 7.7 Loops

Use constrained `scf`/`cf` where possible:

```text
scf.for for uniform loops
scf.if for uniform conditions
cf.br / cf.cond_br after canonicalization
```

Per-lane divergent conditionals should canonicalize to masks/selects where possible.

---

## 8. Resource metadata and lowering to SSAVC4

### 8.1 Required SSAVC4 output

`vc4tile → ssavc4` must emit:

```text
ssavc4.func
  with vc4.launch_abi metadata
  with vc4.resource metadata
  using i32/f32/vector<16xi32>/vector<16xf32>
  using SSAVC4 block arguments for CFG merges
  using SSAVC4 effects/tokens for TMU/VPM/VDW/barrier/uniform/thread-end
```

### 8.2 Schedule mode derivation

Derive:

```text
independent_vector
```

when:

```text
no vc4tile.shared_* ops
no vc4tile.barrier
no cooperative block semantics
```

Derive:

```text
cooperative_block
```

when:

```text
shared VPM is used
barrier is used
cross-warp cooperation is used
```

### 8.3 Cooperative resource derivation

For cooperative kernels, compute:

```text
warps_per_block_max
semaphores_per_block
vpm_rows_per_block
vpm_bytes_per_block
max_resident_blocks
require_full_block_residency
uses_barrier
uses_shared_vpm
```

Barrier kernels should use:

```text
semaphores_per_block = 4
require_full_block_residency = true
```

unless a later verified barrier protocol changes this.

### 8.4 Uniform ABI

The dialect should define a stable ABI concept for builtins, but the exact packed uniform order may be finalized in the lowering.

Required logical builtin information:

```text
logical_request
total_requests
logical_block_id
logical_warp_id
warps_per_block
vpm_base_row
vpm_rows
barrier_arrive_sem
barrier_go_sem
barrier_depart_sem
barrier_reset_sem
resident_request_id
spill_frame_base
spill_frame_bytes
spill_frame_stride_bytes
kernel buffer/scalar arguments
```

These are already represented in `vc4_launch_request_info`. `vc4tile` should map kernel builtins to these runtime-provided fields.

### 8.5 Spill interaction

Current runtime metadata includes spill-frame fields. `vc4tile` should not expose public spill operations. Spilling remains a lowering-private allocator feature in `ssavc4 → vc4`. However, `vc4tile → ssavc4` must preserve enough metadata for SSAVC4/VC4 lowering and artifact emission to describe spill requirements to the runtime.

---

## 9. Feature tiers

The dialect should define the full architecture up front, but implementation and verification should proceed in tiers.

### Tier 0: independent-vector tile kernels

Purpose: prove the basic tile/kernel model.

Features:

```text
vc4tile.kernel
program_id / lane_range
vector<16xi32> / vector<16xf32>
scalar uniforms
masks
masked global load
masked coalesced global store
arith
tail-safe code
independent_vector metadata
```

Example target kernels:

```text
vector add
SAXPY
elementwise affine map
simple masked tail store
```

### Tier 1: warp-local vector operations

Purpose: expose operations that use one QPU’s 16 lanes cooperatively.

Features:

```text
rotate
horizontal reduce
mask reductions
subgroup/warp-level operations
```

Example target kernels:

```text
warp reduce sum
row-wise small reduction
subgroup rotate test
```

### Tier 2: cooperative block structure

Purpose: represent CUDA-like blocks as 1..12 QPU warp tiles.

Features:

```text
block_id
warp_id
warps_per_block
thread_id derived from warp_id/lane
cooperative_block schedule mode
full-residency resource flags
```

Example target kernels:

```text
cooperative ID smoke
block-thread indexing test
multi-warp block write pattern
```

### Tier 3: VPM shared tile memory

Purpose: expose block-local shared memory through VC4-compatible VPM tile abstractions.

Features:

```text
shared_alloc
shared_load
shared_store
VPM row layout metadata
row/column affine indexing
hidden row accounting
```

Example target kernels:

```text
shared 16x16 transpose
block-local staging
shared row exchange
```

### Tier 4: block barrier

Purpose: expose CUDA-like `__syncthreads()` at the tile-kernel level.

Features:

```text
vc4tile.barrier
four-semaphore reusable barrier lowering
uniform participation verifier
full-residency verifier
VPM visibility across barrier
```

Example target kernels:

```text
barrier smoke
full 12-warp block barrier
two resident blocks with VPM/semaphore partitioning
```

### Tier 5: tiled math and matmul/conv extension points

Purpose: prepare for real model coverage.

Features:

```text
block-tile load/store patterns
accumulator tiles
shared tile staging
reduction loops
layout annotations sufficient for matmul/conv-like kernels
```

Caution: do not over-specify a final matmul microkernel before inspecting IREE/Triton producer shapes. Define extension points and simple patterns first; let real producer examples inform the final microkernel forms.

---

## 10. Six-stage project plan

The project should proceed in six stages.

### Stage 1: define the full VC4Tile architecture

Deliverables:

```text
VC4Tile design document
dialect name/mnemonic/namespace locked
op surface and feature tiers specified
resource model specified
uniform ABI model specified
mapping to SSAVC4 specified
producer strategy specified
unsupported-feature policy specified
verification strategy specified
```

This stage should define the full dialect architecture, not only the Tier 0 independent-vector subset. The contract should include independent-vector kernels, warp-local reductions/rotates, cooperative blocks, shared VPM tiles, and barriers from the beginning.

However, “defined” does not mean fully implemented. The architecture should be stable enough that later producer wiring does not require renaming or rethinking the dialect.

### Stage 2: implement standalone `vc4tile → ssavc4` vertical slices

Implement and hardware-verify `vc4tile → ssavc4` in feature slices:

```text
2a. dialect scaffold / parser-printer / verifier
2b. independent vector add/SAXPY
2c. tail masks and coalesced global stores
2d. warp-local rotate/reduction
2e. cooperative block ID / warp ID
2f. VPM shared load/store
2g. barrier
2h. shared transpose or block reduction
```

Producer adapters are not required yet. The fixtures are handwritten `vc4tile` files. This stage proves the core lower-half integration.

### Stage 3: producer-shape probes

Before deep wiring, run targeted producer-shape probes:

```text
Triton TTIR vector add
Triton TTIR simple reduction shape
IREE elementwise dispatch executable
IREE reduction/shared-memory dispatch executable
```

The goal is to confirm that `vc4tile` is reachable from real producers. These probes may be diagnostic-only at first. They should inform any final verifier or op-shape adjustments before deep adapters are written.

### Stage 4: Triton TTIR to VC4Tile adapter

Implement:

```text
Triton TTIR subset → vc4tile
```

Initial supported TTIR concepts:

```text
tt.func
tt.get_program_id
tt.make_range
tt.splat
tt.addptr
tt.load with mask
tt.store with mask
arith elementwise ops
static tile sizes, initially 16 or multiples of 16
```

Do not start from late CUDA/AMD-specific TTGIR. Late TTGIR can already contain target layout decisions that do not match VC4’s 16-lane QPU model.

### Stage 5: IREE compiler-side VC4 path

Implement a compiler-only IREE path:

```text
StableHLO / Torch-MLIR input
  → IREE Flow/Stream/HAL/codegen
  → hal.executable.variant target("vc4")
  → VC4 target config and translation
  → vc4tile
  → ssavc4
  → vc4 artifacts
```

This does not require a full IREE runtime/HAL driver yet. Host execution at this stage can still use:

```text
generated/handwritten C harness
vc4ProgramCreateFromImage
vc4Malloc
vc4MemcpyHtoD
kernel_launch wrapper / vc4LaunchKernel
vc4MemcpyDtoH
```

However, this is more than “dump arbitrary IREE IR.” IREE needs a VC4 compiler target configuration so that tiling/distribution choices respect VC4:

```text
subgroup size = 16
max workgroup threads = 192
shared memory budget = 4096 bytes minus reserves
limited scatter/atomics
coalesced stores preferred
limited divergent side-effecting control flow
```

### Stage 6: IREE HAL runtime/device wrapper around `vc4_runtime`

Implement enough IREE runtime integration for full JAX/PyTorch programs to execute without handwritten host code.

Mapping:

```text
IREE executable load      → vc4ProgramCreateFromImage
IREE buffer allocation    → vc4Malloc
IREE buffer free          → vc4Free
IREE host→device copy     → vc4MemcpyHtoD
IREE device→host copy     → vc4MemcpyDtoH
IREE device→device copy   → vc4MemcpyDtoD
IREE fill                 → vc4MemsetD8
IREE dispatch             → vc4LaunchKernel
```

At this stage, full end-to-end supported programs become possible:

```text
JAX → StableHLO → IREE → VC4 compiler target → VC4 HAL runtime → hardware
PyTorch → Torch-MLIR/IREE or StableHLO/IREE → VC4 compiler target → VC4 HAL runtime → hardware
Triton TTIR → vc4tile → ssavc4 → hardware
```

Initial program coverage should still be limited to supported static-shape/simple-kernel subsets.

---

## 11. Producer integration details

### 11.1 Triton path

The initial Triton path should be:

```text
Triton Python kernel
  → TTIR / early target-neutral Triton MLIR
  → Triton-to-vc4tile adapter
  → vc4tile
  → ssavc4
  → vc4
  → artifacts/runtime
```

Expected mapping:

```text
tt.func                   → vc4tile.kernel
tt.get_program_id         → vc4tile.program_id
tt.make_range / arange    → vc4tile.lane_range or tile_range
tt.splat                  → arith/vector splat or canonicalized uniform splat
tt.addptr                 → buffer base + vector offset arithmetic
tt.load(..., mask=...)    → vc4tile.masked_load_global
tt.store(..., mask=...)   → vc4tile.masked_store_global
arith.addf/addi/muli/etc  → arith ops retained
tensor<NxT> tile values   → vector<NxT> or stripmined vector<16xT> chunks
```

Initial Triton restrictions:

```text
TTIR only or early target-neutral form
static tile sizes
f32/i32 initially
1-D or simple affine pointer arithmetic
masked coalesced stores
no late CUDA TTGIR layout assumptions
no arbitrary dot/matmul until VC4Tile Tier 5 is ready
```

### 11.2 IREE path

The initial IREE path should be compiler-side only:

```text
JAX/PyTorch input
  → IREE pipeline
  → one or more hal.executable.variant ops for target("vc4")
  → VC4 codegen translation
  → vc4tile
```

The adapter should normalize late executable bodies into `vc4tile`, not accept arbitrary IREE internal IR directly.

Likely source dialect concepts after IREE codegen may include:

```text
arith
memref
scf/cf
vector
IREE codegen attributes
IREE GPU/common codegen helper concepts
HAL executable interfaces
workgroup ID/count/size operations or equivalent
```

The VC4 target path should reject or normalize away producer-private operations before the core `vc4tile → ssavc4` lowering.

### 11.3 JAX path

Recommended initial JAX path:

```text
JAX
  → jax.export / StableHLO
  → IREE StableHLO input
  → IREE VC4 compiler target
  → vc4tile
  → hardware
```

JAX is likely the cleaner first full-program path because JAX export to StableHLO is a first-class route, StableHLO is a stable ML compiler interchange format, and IREE uses StableHLO as an input format.

### 11.4 PyTorch path

Recommended PyTorch paths:

```text
PyTorch
  → IREE Turbine / Torch-MLIR
  → IREE
  → VC4 target
```

and/or:

```text
PyTorch
  → torch.export / PyTorch-XLA StableHLO export
  → StableHLO
  → IREE
  → VC4 target
```

PyTorch will likely have more frontend friction than JAX because capture/export/decomposition/dynamic-shape issues are more complex. That should not change the target architecture: PyTorch should still go through IREE and eventually produce VC4-compatible dispatch kernels.

---

## 12. Host-side model by stage

### 12.1 Before IREE runtime integration

Compiler-only IREE support means:

```text
IREE helps generate or expose device dispatch kernels.
The VC4 backend lowers those kernels to vc4tile/ssavc4/vc4 artifacts.
Host execution still uses the existing VC4 runtime and generated/manual C harness.
```

For example:

```c
struct vc4_program *program = NULL;
vc4ProgramCreateFromImage(&program, &my_module_image, heap_bytes);

vc4_deviceptr_t x_dev, y_dev, out_dev;
vc4Malloc(program, &x_dev, bytes);
vc4Malloc(program, &y_dev, bytes);
vc4Malloc(program, &out_dev, bytes);

vc4MemcpyHtoD(program, x_dev, x_host, bytes);
vc4MemcpyHtoD(program, y_dev, y_host, bytes);

my_kernel_launch(program, grid, block, x_dev, y_dev, out_dev, n);

vc4MemcpyDtoH(program, out_host, out_dev, bytes);
vc4_program_destroy(program);
```

This is appropriate for Stages 1–5 compiler bring-up.

### 12.2 After IREE HAL runtime integration

Full JAX/PyTorch program execution requires IREE runtime integration. At that point, the user-facing host code is IREE-style module invocation; the VC4 runtime is called internally by the VC4 HAL device.

The wrapper should not rewrite the lower half. It should map IREE concepts onto current runtime functions:

```text
IREE HAL buffer          → vc4_deviceptr_t allocated by vc4Malloc
IREE HAL executable      → vc4_module_image loaded by vc4ProgramCreateFromImage
IREE HAL dispatch        → vc4LaunchKernel
IREE command sequencing  → synchronous or serialized VC4 launch behavior initially
```

### 12.3 Runtime issue to fix for full programs

The current runtime rejects `vc4_module_image` values whose `kernel_count` exceeds `VC4_RUNTIME_MAX_QPUS`. This is acceptable for early fixtures but likely too small for full IREE-compiled programs, which may contain more than 12 dispatch kernels. The program image should eventually decouple:

```text
number of kernels in program image
```

from:

```text
number of active QPU slots
```

This is a runtime/artifact robustness item for the IREE HAL integration stage.

---

## 13. Expected JAX/PyTorch coverage

Initial `vc4tile` support does not imply arbitrary JAX/PyTorch support.

Coverage depends on four layers:

```text
frontend import coverage
IREE middle-end/codegen coverage
VC4Tile adapter/lowering coverage
runtime/HAL whole-program coverage
```

### 13.1 Initial coverage

With Tier 0/Tier 1 support, expect:

```text
simple elementwise kernels
SAXPY/vector add
simple broadcast-like indexing
masked tail stores
simple reductions after Tier 1
```

This is enough to prove the path, but not enough for general neural networks.

### 13.2 Coverage expansion ladder

Coverage grows by adding dispatch/kernel patterns:

```text
Level 0: elementwise unary/binary
Level 1: elementwise + broadcast + select/masks
Level 2: reductions
Level 3: cooperative VPM/barrier
Level 4: tiled matmul / dot_general
Level 5: convolution/pooling
Level 6: softmax/attention pieces
Level 7: gather/scatter/atomics/dynamic shapes/quantization/custom ops
```

The biggest coverage unlocks are:

```text
reductions
cooperative shared VPM + barrier
tiled matmul/dot_general
convolution or conv→matmul strategy
full IREE HAL runtime wrapper
multi-kernel/multi-dispatch program image robustness
```

### 13.3 Supported-program language

Initial Stage 6 success should be described as:

```text
supported static-shape JAX/PyTorch programs run end-to-end
```

not:

```text
all JAX/PyTorch programs work
```

This is important. The architecture is right, but broad model coverage requires many kernel patterns and runtime maturity.

---

## 14. Verification strategy

The verification strategy should mirror the discipline used for SSAVC4 and scheduled VC4.

Each feature tier should have:

```text
1. dialect roundtrip tests
2. invalid verifier tests
3. vc4tile → ssavc4 conversion tests
4. scheduled vc4 output checks
5. artifact emission checks
6. hardware fixture where the feature affects hardware behavior
```

### 14.1 Stage 1/2 fixtures

Recommended handwritten `vc4tile` fixtures:

```text
vc4tile_vector_add.mlir
vc4tile_saxpy.mlir
vc4tile_tail_mask_store.mlir
vc4tile_warp_reduce_sum.mlir
vc4tile_cooperative_id_smoke.mlir
vc4tile_shared_transpose_16x16.mlir
vc4tile_barrier_smoke.mlir
vc4tile_two_resident_blocks_barrier.mlir
```

### 14.2 Producer probe fixtures

Recommended producer-shape fixtures:

```text
triton_ttir_vector_add_to_vc4tile.mlir
triton_ttir_saxpy_to_vc4tile.mlir
triton_ttir_reduce_to_vc4tile.mlir

iree_elementwise_dispatch_to_vc4tile.mlir
iree_reduction_dispatch_to_vc4tile.mlir
iree_shared_memory_dispatch_to_vc4tile.mlir
```

### 14.3 Runtime/HAL fixtures

Recommended Stage 6 fixtures:

```text
iree_vc4_hal_buffer_alloc_copy_smoke
iree_vc4_hal_single_dispatch_vector_add
iree_vc4_hal_two_dispatch_pipeline
iree_vc4_hal_multi_kernel_module_image
jax_stablehlo_elementwise_end_to_end
pytorch_torch_mlir_elementwise_end_to_end
```

---

## 15. Unsupported-feature policy

The compiler should reject unsupported producer features with precise diagnostics.

Initial unsupported features:

```text
arbitrary scatter stores
global atomics
shared-memory atomics
dynamic shared memory
arbitrary divergent side-effecting control flow
divergent barrier participation
i64/f64/f16/bf16/i8 unless explicitly supported
late target-specific Triton TTGIR layouts
NVVM/ROCDL/SPIR-V dialects as input to vc4tile
MMA/tensor-core ops
device recursion / non-inlined device calls
async dependency graphs unless explicitly integrated
large private arrays requiring unsupported spilling behavior
```

Unsupported features should not silently fall back to wrong or inefficient semantics. For early bring-up, deterministic rejection is better than accidental wrong-code.

---

## 16. Design constraints that must not be violated

The following constraints are locked unless later hardware tests prove a better implementation:

```text
1. Warp size is 16.
2. Physical QPU number is not logical warp ID.
3. The device is modeled as one SM, not three slice-SMs.
4. Shared memory is one 4 KiB global VPM pool.
5. Barrier-enabled blocks require full residency.
6. First barrier lowering uses the validated four-semaphore reusable protocol.
7. VPM shared memory is row/tile oriented, not arbitrary scalar SRAM.
8. Global stores prefer coalesced/affine vector forms through VPM/VDW.
9. Arbitrary scatter and atomics are not supported initially.
10. `vc4tile` does not expose TMU/VDW/semaphore implementation details directly.
11. `ssavc4` remains the pre-RA/pre-scheduling target machine IR.
12. scheduled `vc4` remains the QASM-near artifact sink.
13. The lower-half artifact/runtime path remains the execution backend.
```

---

## 17. Example VC4Tile sketches

These are illustrative only. Exact syntax should be defined in ODS.

### 17.1 Independent vector add

```mlir
vc4tile.kernel @vadd(%a: !vc4tile.buffer<f32>,
                     %b: !vc4tile.buffer<f32>,
                     %c: !vc4tile.buffer<f32>,
                     %n: i32)
    attributes {
      schedule_mode = "independent_vector",
      warp_size = 16 : i32,
      tail_policy = "masked_tail"
    } {
  %pid  = vc4tile.program_id x : i32
  %lane = vc4tile.lane_range : vector<16xi32>

  %sixteen = arith.constant 16 : i32
  %base = arith.muli %pid, %sixteen : i32
  %basev = vc4tile.splat %base : i32 -> vector<16xi32>
  %offs = arith.addi %basev, %lane : vector<16xi32>

  %nv = vc4tile.splat %n : i32 -> vector<16xi32>
  %mask = arith.cmpi ult, %offs, %nv : vector<16xi32>

  %av = vc4tile.masked_load_global %a[%offs], %mask
        : !vc4tile.buffer<f32>, vector<16xi32>, vector<16xi1> -> vector<16xf32>
  %bv = vc4tile.masked_load_global %b[%offs], %mask
        : !vc4tile.buffer<f32>, vector<16xi32>, vector<16xi1> -> vector<16xf32>

  %sum = arith.addf %av, %bv : vector<16xf32>

  vc4tile.masked_store_global %c[%offs], %sum, %mask
        : !vc4tile.buffer<f32>, vector<16xi32>, vector<16xf32>, vector<16xi1>
  vc4tile.return
}
```

### 17.2 Cooperative shared-memory barrier sketch

```mlir
vc4tile.kernel @shared_exchange(%out: !vc4tile.buffer<i32>)
    attributes {
      schedule_mode = "cooperative_block",
      warp_size = 16 : i32,
      warps_per_block_max = 4 : i32,
      uses_shared_vpm,
      uses_barrier,
      requires_full_block_residency
    } {
  %warp = vc4tile.warp_id : i32
  %lane = vc4tile.lane_range : vector<16xi32>

  %shared = vc4tile.shared_alloc @tile
      {rows = 4 : i32, element_type = i32}

  %value = ... : vector<16xi32>
  vc4tile.shared_store %shared[%warp, %lane], %value
      : !vc4tile.shared<i32>, i32, vector<16xi32>, vector<16xi32>

  vc4tile.barrier

  %other = ...
  %loaded = vc4tile.shared_load %shared[%other, %lane]
      : !vc4tile.shared<i32>, i32, vector<16xi32> -> vector<16xi32>

  ...
  vc4tile.return
}
```

---

## 18. Acceptance criteria for the VC4Tile milestone

A strong first VC4Tile milestone is accepted only when:

```text
1. The VC4 Tile dialect is present as vc4tile.
2. vc4tile parses, prints, verifies, and round-trips.
3. The design doc defines independent-vector, warp-local, cooperative, VPM, and barrier tiers.
4. Tier 0 independent-vector kernels lower to SSAVC4 and then to scheduled VC4.
5. At least one Tier 0 hardware fixture passes.
6. Tail masks and coalesced global stores are verified.
7. The lowering emits existing vc4.launch_abi / vc4.resource metadata without breaking the runtime/artifact path.
8. The dialect verifier rejects unsupported scatter/atomics/dynamic shared memory/divergent barriers.
9. At least one producer-shape probe exists for Triton TTIR and one for IREE executable IR, even if diagnostic-only.
10. The plan for Stage 4 Triton adapter and Stage 5 IREE compiler-side adapter is encoded in milestone artifacts.
```

A later full ecosystem milestone is accepted only when:

```text
1. Triton TTIR vector add/SAXPY lowers through vc4tile to hardware.
2. An IREE-generated elementwise dispatch lowers through vc4tile to hardware.
3. The IREE VC4 compiler target path can produce VC4 artifacts for a simple full program.
4. The VC4 HAL runtime wrapper maps IREE buffers/executables/dispatches to vc4_runtime.
5. At least one JAX StableHLO program runs end-to-end.
6. At least one PyTorch/Torch-MLIR or PyTorch/StableHLO program runs end-to-end.
```

---

## 19. Final locked plan

The plan going forward is:

```text
1. Define the full VC4 Tile dialect architecture.
   Include independent vector, warp-local ops, cooperative blocks,
   VPM shared tiles, barriers, reductions, resource metadata, and feature tiers.

2. Implement vc4tile → ssavc4 in standalone feature slices.
   Use handwritten vc4tile fixtures and hardware verification.

3. Run producer-shape probes.
   Use real or representative Triton TTIR and IREE executable snippets to validate
   that vc4tile is reachable without redesign.

4. Implement Triton TTIR → vc4tile.
   Start with vector add/SAXPY-style tile kernels, then reductions/matmul-like patterns.

5. Implement IREE compiler-side VC4 target path → vc4tile.
   Reuse IREE dispatch formation, bufferization, tiling, distribution, and codegen
   where possible, but configure the backend for VC4’s 16-lane, 12-QPU, 4 KiB VPM model.

6. Wrap vc4_runtime as an IREE HAL runtime/device.
   This enables full JAX/PyTorch program execution without handwritten per-program host code.
```

The key project invariant is:

```text
vc4tile is the shared tile-kernel boundary.

Triton and IREE lower into vc4tile.
vc4tile lowers once into ssavc4.
ssavc4 lowers into scheduled vc4.
scheduled vc4 emits the existing artifact/runtime path.
```

---

## Appendix A: Proposed repo layout

```text
compiler/include/vc4/Dialect/VC4Tile/IR/
  VC4TileDialect.h
  VC4TileOps.td
  VC4TileTypes.td
  VC4TileAttrs.td
  VC4TileInterfaces.td

compiler/lib/Dialect/VC4Tile/IR/
  VC4TileDialect.cpp
  VC4TileOps.cpp
  VC4TileTypes.cpp
  VC4TileAttrs.cpp

compiler/include/vc4/Conversion/VC4TileToSSAVC4/
  VC4TileToSSAVC4.h

compiler/lib/Conversion/VC4TileToSSAVC4/
  VC4TileToSSAVC4.cpp

compiler/test/Dialect/VC4Tile/
  roundtrip.mlir
  invalid.mlir
  types_attrs.mlir
  independent_vector.mlir
  cooperative_resource_verifier.mlir

compiler/test/Conversion/VC4TileToSSAVC4/
  vector_add.mlir
  saxpy.mlir
  tail_mask.mlir
  warp_reduce.mlir
  cooperative_ids.mlir
  shared_vpm.mlir
  barrier.mlir

compiler/test/Integration/VC4Tile/
  vector_add_vc4tile
  saxpy_vc4tile
  shared_transpose_vc4tile
  barrier_vc4tile
```

---

## Appendix B: Suggested milestone slice names

```text
vc4tile-00-milestone-package
vc4tile-01-dialect-scaffold
vc4tile-02-type-model-and-core-ops
vc4tile-03-independent-vector-lowering
vc4tile-04-global-load-store-and-tail-masks
vc4tile-05-first-hardware-vector-add
vc4tile-06-warp-rotate-reduce
vc4tile-07-cooperative-block-ids
vc4tile-08-shared-vpm-tiles
vc4tile-09-barrier
vc4tile-10-producer-shape-probes
vc4tile-11-triton-ttir-adapter-v0
vc4tile-12-iree-compiler-side-adapter-v0
vc4tile-13-final-acceptance
```

If the milestone is kept narrower, slices 11 and 12 can be moved into the next milestone. The dialect architecture should still be designed with them in mind.

---

## Appendix C: Upstream reference links

These references are for the ecosystem and producer-side claims in this document.

1. MLIR GPU dialect documentation  
   <https://mlir.llvm.org/docs/Dialects/GPU/>

2. MLIR GPU dialect source documentation for `gpu.launch` / `gpu.launch_func` outlining semantics  
   <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/GPU/IR/GPUOps.td>

3. IREE HAL dialect documentation  
   <https://iree.dev/reference/mlir-dialects/HAL/>

4. IREE HAL pass documentation, including executable variant translation  
   <https://iree.dev/reference/mlir-passes/HAL/>

5. IREE Codegen dialect documentation  
   <https://iree.dev/reference/mlir-dialects/IREECodegen/>

6. IREE common GPU codegen pass documentation  
   <https://iree.dev/reference/mlir-passes/CodegenCommonGPU/>

7. Triton MLIR dialect documentation  
   <https://triton-lang.org/main/dialects/dialects.html>

8. Triton ops documentation  
   <https://triton-lang.org/main/dialects/TritonOps.html>

9. PyTorch blog: Triton kernel compilation stages  
   <https://pytorch.org/blog/triton-kernel-compilation-stages/>

10. StableHLO overview  
    <https://openxla.org/stablehlo>

11. StableHLO specification  
    <https://openxla.org/stablehlo/spec>

12. JAX StableHLO export tutorial  
    <https://openxla.org/stablehlo/tutorials/jax-export>

13. IREE PyTorch guide  
    <https://iree.dev/guides/ml-frameworks/pytorch/>

14. Torch-MLIR architecture / backend contract  
    <https://github.com/llvm/torch-mlir/blob/main/docs/architecture.md>

15. IREE Turbine repository  
    <https://github.com/iree-org/iree-turbine>
