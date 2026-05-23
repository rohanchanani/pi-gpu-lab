# VC4 Tile Dialect (`vc4tile`) Design for M4

**Status:** Milestone 4 implementation specification.
**Dialect name:** VC4 Tile dialect.
**Mnemonic:** `vc4tile`.
**C++ namespace:** `::mlir::vc4tile`.
**ODS prefix:** `VC4Tile`.
**Repository directory:** `compiler/include/vc4/Dialect/VC4Tile/IR` and `compiler/lib/Dialect/VC4Tile/IR`.

## 1. Locked milestone scope

M4 defines a new MLIR dialect, `vc4tile`, and lowers it to the already-implemented `ssavc4` dialect.

```text
future producer kernel IR: Triton TTIR / IREE late executable IR / MLIR-GPU-like normalized IR
  ↓ future producer adapters, not M4
vc4tile
  ↓ M4: --convert-vc4tile-to-ssavc4
ssavc4
  ↓ existing M3: --convert-ssavc4-to-vc4
scheduled vc4
  ↓ existing M2 artifact emitter
QASM / shader arrays / kernel_launch.c/h / manifest/layout
  ↓ existing vc4_runtime
VideoCore IV hardware
```

M4 must not implement Triton, IREE, StableHLO, Torch-MLIR, JAX, PyTorch, or upstream MLIR `gpu` producer lowering. M4 must not lower `vc4tile` directly to scheduled `vc4`.

## 2. Why this dialect exists

`ssavc4` is the pre-register-allocation/pre-scheduling machine-SSA layer. It exposes VC4 machine resources such as TMU, VPM, VDW, semaphores, barriers, flags, branches, and thread termination. That is exactly right for the lower half, but too low as the shared target of future Triton/IREE producer adapters.

`vc4tile` is the producer-facing VC4 kernel/tile boundary. It represents the work of one logical 16-lane QPU tile and optional groups of those tiles into cooperative blocks. It should be target-specific enough to express VC4's real SIMD/VPM/barrier constraints, but high enough that producers do not need to know TMU request token sequencing, VDW setup words, raw VPM setup registers, semaphore micro-protocols, scheduled hazards, physical register allocation, or QASM.

## 3. Target model embedded in the dialect

The dialect assumes the VC4-as-CUDA model:

```text
one CUDA-like SM                 = the whole VC4 QPU execution domain
physical warp slots              = 12 QPUs
lanes per warp/tile              = 16 SIMD lanes
max logical resident threads     = 192
shared memory                    = one global 4 KiB VPM user window
semaphores                       = 16 counting semaphores
barrier implementation           = validated four-semaphore reusable protocol
independent-vector schedule mode = no shared VPM/barrier/cross-warp cooperation
cooperative-block schedule mode  = 1..12 logical QPU warps per block
```

Physical `QPU_NUMBER` is never a logical identity source. Logical request, block, and warp IDs come from runtime-provided launch metadata/uniforms. Lane ID comes from QPU `ELEMENT_NUMBER` after lowering to SSAVC4.

The physical limits remain binding on every logical model in this dialect: there are 12 physical QPU warp slots, each QPU warp has 16 lanes, independent-vector launches run in waves of at most the active QPUs, and cooperative blocks are constrained by active QPUs, `warps_per_block`, VPM rows/bytes, semaphores, and full-residency requirements. Logical request, warp, and block IDs are software/runtime metadata defined under that residency model; they do not imply unlimited resident workers.

## 4. Dialect design principles

1. **Tile-first.** The central unit is a 16-lane QPU tile, not a scalar CUDA thread and not a high-level tensor.
2. **Single-dispatch kernel IR.** `vc4tile` describes one kernel body and its resource requirements; it does not represent model-level host scheduling, Flow/Stream/HAL, StableHLO, Torch, or Linalg.
3. **Lower to SSAVC4 only.** `vc4tile` is not an artifact emitter input.
4. **Full semantic contract early.** The op/type/attr surface should include independent-vector and cooperative concepts early. Lowering support comes online in vertical slices.
5. **Precise rejection.** Unsupported forms must fail with deterministic diagnostics, not silently lower incorrectly.
6. **Hardware as gold standard.** Every executable feature eventually needs hardware/reference verification from `vc4tile` input.

## 5. Region and dialect dependencies

A `vc4tile.kernel` owns a single region. The region may use:

```text
vc4tile ops for kernel/tile/resource/memory/barrier semantics
arith for scalar/vector arithmetic and comparisons
cf for explicit branches and block arguments
scf only for simple structured forms that the lowering can canonicalize or directly handle
vector for vector<16x...> data and masks where useful
builtin/index types with explicit M4 lowering policy
```

The dialect must not accept tensors, linalg, StableHLO, Torch, Triton dialect ops, IREE-private ops, NVVM, ROCDL, SPIR-V, or upstream `gpu` ops as M4 inputs unless they have already been normalized by a future producer adapter.

## 6. Core values and types

M4 should use standard MLIR scalar/vector types where possible:

```text
i1                         uniform boolean
index                      accepted only where the lowering can convert to 32-bit index arithmetic
i32                        scalar uniform integer / pointer word / offset
f32                        scalar uniform float
vector<16xi1>              lane mask
vector<16xi32>             16-lane integer tile value
vector<16xf32>             16-lane float tile value
```

Unsupported initially:

```text
i64 / f64
f16 / bf16 / i8 quantized arithmetic
vector widths other than 16 in the core lowering form
tensor values inside vc4tile.kernel
arbitrary pointer/address-space dialect types not normalized to buffer + offset semantics
```

Larger tiles may be introduced later as a producer-normalization convenience, but the core `vc4tile -> ssavc4` lowering must operate on `vector<16xT>` chunks.

## 7. Required op families

The exact ODS names may be refined during implementation, but M4 must provide these semantic families.

### 7.1 Kernel and terminator

```mlir
vc4tile.kernel @name(...) attributes { ... } { ... }
vc4tile.return
```

`vc4tile.kernel` is isolated from above, has a symbol name/public name, owns a region, and carries or derives schedule/resource metadata. It must be possible to generate existing `vc4.launch_abi` and `vc4.resource` dictionaries from the kernel.

### 7.2 Identity and lane operations

```mlir
vc4tile.program_id        // logical independent-vector request/program id
vc4tile.block_id          // logical cooperative block id
vc4tile.warp_id           // logical warp id inside cooperative block
vc4tile.lane_id           // scalar lane id if needed
vc4tile.lane_range        // vector<16xi32> [0, 1, ..., 15]
vc4tile.thread_id         // logical thread id within block/tile when useful
```

Lowering policy:

```text
lane_range / lane_id -> ssavc4.element_number-derived values
program_id           -> launch ABI uniform / logical_request
block_id             -> launch ABI uniform / logical_block_id
warp_id              -> launch ABI uniform / logical_warp_id
thread_id            -> warp_id * 16 + lane_id, with masks for inactive lanes
```

## VC4Tile ABI value categories

VC4Tile has three distinct ABI value categories. They may share the same physical uniform transport after lowering, but their semantic ownership is different and must not be collapsed.

**Kernel arguments.** User/caller-supplied kernel inputs are formal arguments of `vc4tile.kernel`. Examples include output pointers, input pointers, `n`, `alpha`, strides, and scalar parameters. These values lower to `vc4.launch_abi.args[]`, appear in generated launch wrapper signatures, and occupy physical uniform stream slots only because the VC4 uniform stream is the transport mechanism.

**Runtime builtins.** Runtime/scheduler-provided execution metadata is represented in VC4Tile by zero-operand identity ops such as `vc4tile.program_id`, `vc4tile.block_id`, and `vc4tile.warp_id`, plus lower-half runtime metadata needs. These values lower to `vc4.launch_abi.builtins[]`, are populated from `vc4_launch_request_info`, and occupy physical uniform stream slots after lowering. They are not user launch arguments. Runtime builtin examples include `logical_request`, `total_requests`, `logical_block_id`, `logical_warp_id`, `warps_per_block`, `vpm_base_row`, `vpm_rows`, `semaphore_base`, `barrier_arrive_sem`, `barrier_go_sem`, `barrier_depart_sem`, `barrier_reset_sem`, `resident_request_id`, `spill_frame_base`, `spill_frame_bytes`, `spill_frame_stride_bytes`, and `spill_vpm_row`.

**Lane/register builtins.** `vc4tile.lane_id` and `vc4tile.lane_range` lower to `ssavc4.element_number`-derived values. `ssavc4.element_number` remains a first-class SSAVC4 hardware lane identity op. These values are hardware/register-derived, are not entries in `vc4.launch_abi.builtins[]`, and are not carried in the uniform stream.

`vc4tile.program_id`, `vc4tile.block_id`, and `vc4tile.warp_id` must remain zero-operand runtime identity ops. They must never carry `uniform_index`; forms such as `vc4tile.program_id {uniform_index = ...}` are invalid. They must never be used to represent user arguments such as output pointer, input pointer, `n`, `alpha`, strides, or any other caller-supplied scalar. Such values must be modeled as `vc4tile.kernel` formal arguments.

Future lower-ABI cleanup must remove the stale builtin categories `#vc4.builtin_kind<qpu_num>`, `#vc4.builtin_kind<num_qpus>`, `#vc4.builtin_kind<elem_num>`, and string kind `"hidden_runtime"`. Do not keep aliases for those names. Tests and fixtures that still use them must be modernized during the ABI refactor rather than preserved for compatibility.

Multi-kernel programs continue to use exactly one top-level `vc4.module` containing one or more scheduled kernels. `vc4-codegen` requiring exactly one top-level `vc4.module` is compatible with multi-kernel programs and must not be weakened into multiple top-level modules.

### 7.3 Global memory operations

```mlir
vc4tile.masked_load_global
vc4tile.masked_store_global
```

Initial legal subset:

```text
32-bit element type only: i32 or f32
vector<16xT> value for varying data
vector<16xi32> byte or element offsets after normalization
vector<16xi1> lane mask
coalesced/affine contiguous stores for the hardware-lowered VDW path
```

Lowering policy:

```text
masked_load_global  -> ssavc4.tmu.request / ssavc4.tmu.read
masked_store_global -> ssavc4.vdw.store using VPM/VDW staging
```

Arbitrary per-lane global scatter stores are not part of the initial M4 hardware-lowered subset. They must reject unless a deliberate slow path is implemented and verified.

### 7.4 Warp-local operations

```mlir
vc4tile.rotate
vc4tile.reduce
```

These are warp/tile-local, not block-wide. Supported reductions initially should be limited to operations already representable by SSAVC4 arithmetic/rotate support, especially `add` on `i32`/`f32` where the lower half is proven.

### 7.5 Shared VPM operations

```mlir
vc4tile.shared_alloc
vc4tile.shared_load
vc4tile.shared_store
```

The initial shared-memory model is row-granular and vector/tile-oriented. VPM is a structured 64-row × 16-word window, not arbitrary scalar SRAM. Supported first:

```text
row-granular allocations
affine row/column indexing
per-warp contiguous rows
transpose/reduction-style patterns
32-bit element modes
```

Unsupported initially:

```text
shared-memory atomics
arbitrary per-lane shared scatter/gather
byte-addressed alias-heavy shared memory
dynamic shared memory
```

### 7.6 Barrier

```mlir
vc4tile.barrier
```

`vc4tile.barrier` is only legal in cooperative-block kernels. It requires full block residency and uniform participation by all logical warps in the block. Lowering must produce SSAVC4 barrier/semaphore operations and resource metadata requiring four semaphores per resident block.

## 8. Metadata model

M4 must preserve the existing lower-half metadata conventions rather than inventing a parallel launch system.

`vc4tile -> ssavc4` must generate or carry:

```text
vc4.launch_abi
vc4.resource
```

Independent-vector kernels:

```text
schedule_mode = independent_vector
warps_per_block_max = 1
uses_shared_vpm = false
uses_barrier = false
semaphores_per_block = 0
vpm_rows_per_block = 0
```

Cooperative-block kernels:

```text
schedule_mode = cooperative_block
1 <= warps_per_block_max <= 12
vpm_rows_per_block <= 64
vpm_bytes_per_block <= 4096
semaphores_per_block <= 16
uses_barrier => semaphores_per_block = 4
uses_barrier => require_full_block_residency = true
uses_shared_vpm => vpm rows/bytes nonzero
```

Spill metadata remains an SSAVC4/VC4 lower-half concern. M4 should generate low-pressure IR where practical, but it may rely on the current spill-aware SSAVC4 lower half.

## 9. Control flow and block arguments

`vc4tile` may use `cf`/simple `scf` control flow. The M4 lowering must map supported control flow to SSAVC4 branch operations with successor operands. There is no public `vc4tile.phi` or `ssavc4.phi`; block arguments are the phi-like mechanism.

Supported initially:

```text
uniform control flow
simple loops/merges with block arguments
side-effect-free lane-varying select/predication where explicitly implemented
tail masks
```

Unsupported initially:

```text
general divergent side-effecting branches
divergent barrier participation
early returns around barriers unless proven uniform
```

## 10. Verification invariant

Every implemented `vc4tile` feature must pass the canonical M4 layers:

```text
dialect contract
invalid diagnostic contract
vc4tile -> ssavc4 lowered-IR contract
ssavc4 -> vc4 scheduled/artifact contract
hardware CPU/reference contract when executable on hardware
```

M4 final acceptance also requires a global implementation-integrity audit and lower-half M3/M2 regression.
