# VC4-as-CUDA Mapping Guide for an MLIR Backend

**Status:** Working and now partially hardware-validated design for a CUDA-like abstraction over Raspberry Pi VideoCore IV / VC4 general-purpose QPU execution.

**Intended reader:** Compiler/runtime implementer building an MLIR backend that wants VC4 kernels to *look* as much like CUDA kernels as possible while preserving VC4 correctness.

**Core conclusion:** Model the Raspberry Pi VC4 GPU as **one tiny CUDA-like SM** with **12 physical warp slots**, **16 SIMD lanes per warp**, and **one global 4 KiB user-visible shared-memory window** backed by VPM.

**Current validation status:**

```text
Targeted qpu_num test:              PASS
VPM slice visibility test:          PASS
QPU semaphore barrier / syncthreads: PASS
```

The `__syncthreads()` implementation described here is now considered **locked for first compiler implementation**, subject to the invariants in Sections 10 and 20.

---

## 1. Ground facts this design assumes

### 1.1 Hardware facts from the VC4 architecture guide

The VideoCore IV QPU is a **16-way SIMD processor**. The guide says a QPU can be treated “for all intents and purposes” as a 16-way 32-bit SIMD processor, even though it is physically implemented as a 4-way SIMD processor multiplexed over four cycles.

QPUs are grouped into **slices** of up to four QPUs. A slice shares resources such as:

- instruction cache,
- Special Functions Unit,
- one or two Texture and Memory Lookup Units,
- varying interpolation hardware.

The guide does **not** describe VPM storage as private per slice. In the system block diagram, VPM appears outside the QPU slices as a separate shared system block.

The VPM section states that, from a QPU perspective, the VPM window is a two-dimensional array of 32-bit words:

```text
width:       16 words
max height:  64 rows
word size:   32 bits
```

Therefore the user-visible generic QPU window is:

```text
64 rows × 16 words/row × 4 bytes/word = 4096 bytes
```

The guide also defines `V3D_VPMBASE.VPMURSV` as the amount of VPM reserved **for all user programs**, in units of 256 bytes.

The guide documents sixteen system-wide 4-bit counting semaphores. Each QPU can increment or decrement one of those semaphores using the semaphore instruction. A decrement stalls if the semaphore count is zero; an increment stalls if the count is already fifteen.

The guide also documents one global QPU mutex shared between all QPUs. This guide uses that mutex conservatively for VPM/VDW setup and DMA-store sequences until a dedicated setup-clobber test proves which setup states are private and which are shared.

### 1.2 Measured facts from the current Raspberry Pi VC4 test setup

The `vpm_slice_visibility` hardware test observed this topology:

```text
V3D_IDENT1 = 0xc1102431
VPMSZ      = 12 KiB physical VPM
QUPS       = 4 QPUs per slice
NSLC       = 3 slices
num_qpus   = 12
VPMBASE    = 16 reservation units = 4096 bytes
```

It then tested same-slice and cross-slice VPM visibility/collision behavior. Cross-slice pairs saw each other’s writes, and cross-slice same-row collisions behaved as “last writer wins.” Therefore the user-visible VPM storage behaves as:

```text
one global 4 KiB user-visible VPM window
```

not:

```text
three independent 4 KiB windows, one per 4-QPU slice
```

The corrected `qpu_num` test established that the physical QPUs are individually runnable when the host uses `V3D_SQRSV0/1` to reserve all QPUs except the target. Do not infer physical QPU assignment from the order in which short user-program requests are queued.

### 1.3 Measured facts from the `qpu_barrier_syncthreads` hardware test

The `qpu_barrier_syncthreads` hardware test observed the same topology:

```text
V3D_IDENT1 = 0xc1102431
VPMSZ      = 12 KiB physical VPM
QUPS       = 4 QPUs per slice
NSLC       = 3 slices
num_qpus   = 12
NSEM       = 16 semaphores
VPMBASE    = 16 reservation units = 4096 bytes
```

It validated a reusable QPU semaphore barrier intended to implement CUDA-like `__syncthreads()`.

The test passed all of these cases:

```text
same_slice_smoke:
    blocks=1
    warps_per_block=2
    iterations=4
    expected_qpu_mask=0x3
    observed_qpu_mask=0x3
    mismatches=0
    timeout=0

cross_slice_0_1:
    blocks=1
    warps_per_block=2
    iterations=4
    expected_qpu_mask=0x11
    observed_qpu_mask=0x11
    mismatches=0
    timeout=0

cross_slice_0_2:
    blocks=1
    warps_per_block=2
    iterations=4
    expected_qpu_mask=0x101
    observed_qpu_mask=0x101
    mismatches=0
    timeout=0

full_block_stress:
    blocks=1
    warps_per_block=12
    total_requests=12
    iterations=64
    expected_qpu_mask=0xfff
    observed_qpu_mask=0xfff
    every warp all_seen=0xfff
    every warp always_seen=0xfff
    mismatches=0
    timeout=0

two_block_partition:
    blocks=2
    warps_per_block=4
    total_requests=8
    iterations=32
    expected_qpu_mask=0xff
    observed_qpu_mask=0xff
    every block-local warp all_seen=0xf
    every block-local warp always_seen=0xf
    mismatches=0
    timeout=0
```

Final result:

```text
VC4_TEST_RESULT name=qpu_barrier_syncthreads status=PASS
same_slice_pass=1
cross_slice_0_1_pass=1
cross_slice_0_2_pass=1
full_block_pass=1
multi_block_pass=1
qpu_mismatches=0
data_mismatches=0
timeouts=0
invalid_topology=0
errstat_relevant_changed=0
```

Interpretation:

```text
The semaphore barrier is valid as the first implementation of CUDA-like
__syncthreads() for resident logical QPU warps in one block.
```

More precisely:

```text
The barrier synchronizes same-slice QPUs.
The barrier synchronizes cross-slice QPUs.
The barrier works for a full 12-QPU block.
The barrier works across many repeated generations.
The barrier works for multiple resident blocks when VPM rows and semaphore IDs are partitioned per block.
VPM writes before the barrier are visible to participating block warps after the barrier under the tested protocol.
```

The test does **not** prove that oversubscribed blocks can safely wait at barriers. They cannot. The runtime must only schedule barrier-participating blocks when all their logical warps can be resident.

---

## 2. High-level CUDA-like device model

Expose this target as a CUDA-like device with one multiprocessor:

```text
multiProcessorCount:              1
warpSize / subgroup size:         16
max physical warps per SM:        12
max logical threads per SM:       192
max logical threads per block:    192
shared memory per SM:             4096 bytes
shared memory per block:          up to 4096 bytes, subject to occupancy
semaphores per SM:                16 hardware counting semaphores
recommended semaphores per block: 4 if __syncthreads() is used
```

This is the central abstraction:

```text
CUDA-ish device   -> one VC4 V3D/QPU complex
CUDA SM           -> the whole VC4 general-QPU execution domain
CUDA warp         -> one QPU user program, 16 SIMD lanes
CUDA lane         -> one QPU SIMD element
CUDA thread       -> one SIMD lane in one logical QPU warp
CUDA thread block -> a cooperative group of 1..12 logical QPU warps
CUDA shared mem   -> a software allocation in the single global 4 KiB VPM window
CUDA syncthreads  -> a per-block QPU semaphore barrier among resident logical warps
```

Do **not** map VC4 slices to CUDA SMs. Slices are real performance topology, but they are not independent shared-memory domains.

---

## 3. Primitive mapping table

| CUDA / MLIR GPU concept | VC4 mapping | Required compiler/runtime behavior |
|---|---|---|
| `threadIdx.x` | `logical_warp_id * 16 + ELEMENT_NUMBER` | `logical_warp_id` comes from uniforms; `ELEMENT_NUMBER` is QPU A-read register 38. |
| `lane_id` | `ELEMENT_NUMBER` | 0..15. |
| CUDA warp | one QPU user-program request | Warp size is 16, not 32. |
| `warp_id` within block | uniform | Never use physical `QPU_NUMBER` for logical warp identity. |
| `QPU_NUMBER` | physical QPU id | Debug/profiling/topology only. B-read register 38. |
| thread block / CTA | group of 1..12 QPU requests | All barrier-participating warps must be resident together. |
| SM | whole VC4 user-QPU system | Treat as one SM for correctness. |
| slice | group of 4 QPUs sharing I-cache/SFU/TMU/VRI | Performance topology only. Not an SM. |
| `__shared__` memory | VPM row allocation | One global 4 KiB pool, partitioned by runtime among resident blocks. |
| `__syncthreads()` | reusable QPU semaphore barrier | Hardware-validated for same-slice, cross-slice, full 12-warp block, repeated iterations, and multi-block partitioning. |
| global load | TMU direct memory lookup | Natural path for 32-bit reads. |
| global store | QPU register → VPM → VDW DMA store | Efficient mainly for coalesced/affine vector stores. |
| private registers | QPU vector registers and accumulators | Respect QPU register hazards and delay restrictions. |
| atomics | not native in this model | Reject, emulate slowly with global mutex, or lower only special cases. |
| divergent SIMT control flow | SIMD masks/predication | No native CUDA-style per-lane PCs/reconvergence. |

---

## 4. QPU as CUDA warp

### 4.1 Warp size

Set:

```c
#define VC4_WARP_SIZE 16
```

A QPU instruction operates over 16 SIMD elements. These map naturally to CUDA-like lanes:

```text
lane = ELEMENT_NUMBER;   // 0..15
```

### 4.2 Logical thread id

For a one-dimensional block:

```c
uint32_t lane = element_number();
uint32_t tid  = logical_warp_id * 16 + lane;
uint32_t active = tid < block_dim_x;
```

For multidimensional blocks, first compute the flattened logical thread id, then unflatten:

```c
uint32_t flat = logical_warp_id * 16 + lane;

threadIdx.x = flat % blockDim.x;
threadIdx.y = (flat / blockDim.x) % blockDim.y;
threadIdx.z = flat / (blockDim.x * blockDim.y);

active = flat < blockDim.x * blockDim.y * blockDim.z;
```

Boundary lanes must be predicated off. For a block with 100 logical threads, the compiler launches:

```text
ceil(100 / 16) = 7 QPU warps
```

and masks out lanes with:

```text
flat >= 100
```

### 4.3 Do not use physical QPU number as logical warp id

The hardware QPU scheduler chooses physical QPUs automatically. A short program may run repeatedly on the same physical QPU. Therefore:

```text
logical_warp_id  -> uniform supplied by runtime
physical_qpu_id  -> QPU_NUMBER, only for diagnostics/performance
```

The corrected `qpu_num` test used QPU reservations to target individual QPUs. That is useful for tests, not for normal kernel indexing.

The barrier test reinforced this: in the full-block stress run, logical warp IDs were not identical to physical QPU numbers. The logical warp ID came from the test/runtime metadata, while physical QPU numbers were diagnostic observations.

---

## 5. SM model and slices

### 5.1 Correctness model

For correctness, model the device as:

```text
one SM
12 QPU warp slots
one global VPM user window
16 global QPU semaphores
one global QPU mutex
```

The runtime’s block scheduler should act like a CUDA occupancy scheduler for a single SM.

### 5.2 Performance model

VC4 slices still matter for performance:

```text
slice 0: QPU 0, 1, 2, 3
slice 1: QPU 4, 5, 6, 7
slice 2: QPU 8, 9, 10, 11
```

Each slice shares I-cache, SFU, TMU, and VRI-like resources. This can affect instruction-cache locality, TMU pressure, SFU-heavy code, and possibly resource arbitration. However, slices do not define separate shared-memory scopes and do not define separate block scheduling domains.

### 5.3 Rule

Do not expose:

```text
3 SMs × 4 QPUs/slice × 4 KiB shared memory
```

Expose:

```text
1 SM × 12 QPUs × 4 KiB shared memory
```

---

## 6. Thread block / CTA mapping

A CUDA-like thread block maps to a set of logical QPU warps:

```c
warps_per_block = ceil(block_threads / 16);
```

Hard limits:

```text
1 <= warps_per_block <= 12
block_threads <= 192
```

If a block requires barriers or shared memory, the runtime must ensure all of that block’s QPU warps are resident together.

### 6.1 Block uniforms

Each QPU warp request should receive a uniform stream with at least:

```c
struct vc4_warp_uniforms {
    uint32_t block_id_x;
    uint32_t block_id_y;
    uint32_t block_id_z;

    uint32_t grid_dim_x;
    uint32_t grid_dim_y;
    uint32_t grid_dim_z;

    uint32_t block_dim_x;
    uint32_t block_dim_y;
    uint32_t block_dim_z;

    uint32_t logical_warp_id;       // 0..warps_per_block-1
    uint32_t warps_per_block;

    uint32_t vpm_base_row;          // start row of this block's shared allocation
    uint32_t vpm_rows;              // rows allocated to this block

    uint32_t barrier_arrive_sem;    // only if barriers are used
    uint32_t barrier_release_sem;
    uint32_t barrier_depart_sem;
    uint32_t barrier_reset_sem;

    // kernel argument pointers/scalars follow
};
```

The exact layout can be backend-specific, but it must be stable and documented because every QPU request consumes its uniforms sequentially.

### 6.2 Partial last warp

If `block_threads` is not a multiple of 16, the last logical warp still exists and still participates in block-level synchronization. Inactive lanes are masked for computation and memory accesses, but the QPU program representing that logical warp must still execute `__syncthreads()`.

Example:

```text
block_threads = 130
warps_per_block = ceil(130 / 16) = 9
warp 8 active lanes = lanes 0..1
warp 8 inactive lanes = lanes 2..15
warp 8 still participates in every __syncthreads()
```

---

## 7. Shared memory as VPM

### 7.1 Global VPM pool

Use:

```c
#define VC4_VPM_USER_BYTES 4096
#define VC4_VPM_ROWS       64
#define VC4_VPM_ROW_BYTES  64
#define VC4_VPM_ROW_WORDS  16
```

At runtime initialization for user-QPU kernels, reserve the 4 KiB user window:

```c
V3D_VPMBASE = 16;   // 16 × 256 bytes = 4096 bytes
```

The guide says this register can only be written when V3D is idle before shading has commenced. Treat `V3D_VPMBASE` as a global device configuration for this backend.

### 7.2 Per-block shared allocation

CUDA has a shared-memory pool per SM and partitions it among resident blocks. Do the same:

```text
4 KiB VPM pool
  = block 0 shared rows
  + block 1 shared rows
  + ...
  + compiler scratch rows
  + VDW store-staging rows
```

A one-block-at-a-time launch can allocate all rows to the block:

```text
block 0: rows 0..63
```

A multi-resident launch must partition:

```text
block 0: rows  0..15
block 1: rows 16..31
block 2: rows 32..47
block 3: rows 48..63
```

or any equivalent runtime-chosen row layout.

The `qpu_barrier_syncthreads` two-block partition run validated this idea for two resident blocks with four warps per block. Each block observed only its own four-warp block-local mask after repeated barriers.

### 7.3 Row layout

The most natural 32-bit shared-memory layout is:

```text
VPM row    = one 16-lane vector
VPM column = lane id
```

A shared array:

```c
__shared__ uint32_t s[N];
```

maps to:

```c
row = vpm_base_row + index / 16;
col = index % 16;
```

A full-warp contiguous store:

```c
s[logical_warp_id * 16 + lane] = value;
```

becomes one horizontal 32-bit VPM vector write to:

```c
row = vpm_base_row + logical_warp_id;
```

### 7.4 Shared-memory allocation accounting

For 32-bit shared data:

```c
rows = ceil(num_u32_elements / 16);
bytes_charged = rows * 64;
```

For byte/halfword data, VC4 has 8-bit and 16-bit VPM modes, but the initial backend should still allocate conservatively at row granularity unless it has a tested packed-lane lowering.

### 7.5 VPM is not arbitrary scalar SRAM

CUDA shared memory supports arbitrary scalar per-thread addressing. VC4 VPM is structured vector memory. Accesses are programmed using setup registers and then read/written as horizontal or vertical vectors.

Initially support these shared-memory patterns:

```text
contiguous per-warp rows
simple affine row/column accesses
block reductions using row-granular staging
transpose-like patterns that map to horizontal/vertical VPM access
```

Initially reject or slow-path:

```text
arbitrary per-lane shared-memory scatter/gather
large irregular shared arrays
shared-memory atomics
byte-addressed alias-heavy shared memory
```

### 7.6 Hidden VPM rows

Do not assume the entire 4 KiB can always be user-visible `__shared__` memory. The compiler/runtime may need hidden VPM rows for:

- VDW global-store staging,
- reductions,
- temporary transposes,
- spill staging,
- result marshaling in tests.

Two reasonable policies:

```text
Policy A: expose less than 4096 bytes as max user shared memory, reserving hidden rows.
Policy B: expose 4096 bytes but reject kernels whose codegen also needs hidden VPM rows.
```

For early compiler bring-up, Policy A is safer.

---

## 8. Occupancy model

### 8.1 Resource variables

For each kernel/block shape:

```c
warp_size         = 16;
warps_per_block   = ceil(block_threads / 16);

user_shared_rows  = ceil(user_shared_bytes / 64);
compiler_rows     = rows needed for VDW staging, reductions, spills, etc.;
rows_per_block    = user_shared_rows + compiler_rows;

barrier_semas_per_block = needs_barrier ? 4 : 0;
```

The locked reusable barrier uses four semaphores per block:

```text
arrive
release/go
depart
reset
```

This is more conservative than a one-shot two-semaphore barrier, but it avoids generation races when `__syncthreads()` appears multiple times or inside a loop.

### 8.2 Resident block count

For a single-SM model:

```c
resident_blocks = min(
    floor(12 / warps_per_block),
    floor(64 / rows_per_block),
    needs_barrier ? floor(16 / 4) : UINT_MAX
);
```

Also require:

```text
resident_blocks >= 1
```

If not, reject the kernel/block configuration or lower it using a special slow path.

### 8.3 Barrier safety rule

For any kernel using `__syncthreads()`:

```text
All warps of every resident block must be resident before any of them can wait at a barrier.
```

Never enqueue more barrier-participating blocks than can fit by QPU slots, VPM rows, and semaphore IDs.

A deadlocking bad schedule looks like:

```text
block A has 8 warps; only 6 start
those 6 reach barrier and occupy 6 QPUs
remaining 2 warps of block A are queued behind other work
barrier can never complete
```

The runtime must avoid this by scheduling whole resident block waves.

### 8.4 Semaphore count range

Each semaphore is 4-bit, so its count range is:

```text
0..15
```

The block maximum is 12 warps, so the barrier releases at most:

```text
N - 1 <= 11
```

tokens per phase. That is within the semaphore count range.

The runtime must still ensure semaphores are initially drained to zero before assigning them to a block and must not reuse a semaphore set until all QPU requests for the owning resident block have completed.

---

## 9. Software scheduler model

### 9.1 What CUDA does

A CUDA SM has an instruction-level warp scheduler. When one warp stalls, the SM can issue instructions from another resident warp.

### 9.2 What VC4 does

VC4 does not expose that kind of software-controlled warp issue. A QPU user program is queued with:

```text
SRQUA = uniforms address
SRQPC = program counter
```

The hardware QPU scheduler picks a physical QPU. Once a QPU starts a program, it runs that program until it terminates or stalls internally. Your software scheduler cannot preempt it and run another warp on the same QPU.

Therefore:

```text
software block scheduler -> chooses block waves and uniform streams
hardware QPU scheduler   -> assigns queued warp programs to physical QPUs
QPU program              -> one logical warp, runs to completion
```

### 9.3 Runtime scheduling modes

Implement two modes.

#### Mode A: independent-vector mode

Use for kernels with:

```text
no shared memory
no cross-warp barrier
no block-level cooperation
```

Schedule many independent QPU warp tasks. Each request processes a vector tile:

```c
global_base = logical_request_id * 16;
lane        = ELEMENT_NUMBER;
global_id   = global_base + lane;
```

This mode is appropriate for:

- elementwise kernels,
- map operations,
- simple affine loads/stores,
- staged reductions where each pass is separate,
- MLIR `linalg` tiles with no workgroup memory.

#### Mode B: cooperative-block mode

Use for kernels with:

```text
workgroup/shared memory
__syncthreads()
cross-warp reductions
block-level tiling
```

For each resident wave:

```text
1. Select resident blocks according to occupancy.
2. Allocate VPM rows and semaphore IDs for each block.
3. Queue all QPU warp requests for those blocks.
4. Wait for all requests in the wave to complete.
5. Reuse resources for the next wave.
```

For initial correctness, a one-block-at-a-time cooperative-block mode is acceptable:

```text
resident_blocks = 1
```

The barrier test now supports moving beyond one block: the `two_block_partition` run validated two resident blocks with disjoint VPM/semaphore resources. More multi-block combinations should still be added as regression coverage before relying on every possible occupancy shape.

### 9.4 User request FIFO

The QPU user-program request FIFO is 16 requests deep. The cooperative-block resource model caps resident warps at 12, so a full resident wave fits within the request FIFO. This is useful: a barrier-enabled resident wave can be enqueued in full without overflowing the scheduler FIFO.

---

## 10. Barrier mapping: `__syncthreads()`

### 10.1 Locked implementation scope

The following implementation is now the **locked first implementation** of CUDA-like `__syncthreads()`:

```text
A reusable four-semaphore barrier among all resident logical QPU warps of one block.
```

It is valid when:

```text
1. all participating block warps are resident,
2. all participating block warps execute the same barrier generation,
3. each resident block owns distinct semaphore IDs,
4. each resident block owns a distinct VPM row range,
5. the semaphore set is initially drained to zero,
6. the semaphore set is not reused until all QPU requests for that block complete.
```

It is not valid for:

```text
oversubscribed blocks,
barriers in non-uniform/divergent control flow,
blocks whose warps are not all enqueued as part of the same resident wave,
blocks sharing semaphore IDs,
blocks sharing VPM rows by mistake.
```

### 10.2 Hardware primitive

VC4 provides sixteen system-wide 4-bit counting semaphores. A semaphore increment stalls if the count is 15; a decrement stalls if the count is 0.

The semaphore instruction itself may stall due to external arbitration. It must not be combined with writes to closely coupled peripherals that can also stall. In code generation, emit semaphore operations as standalone synchronization instructions with safe destinations and no simultaneous VPM/TMU/TLB/SFU/mutex side effects.

### 10.3 Recommended reusable barrier

For a block with `N` logical QPU warps:

```text
N = warps_per_block
leader = logical_warp_id == 0
```

Use four semaphores per resident block:

```text
arrive_sem
release_sem
depart_sem
reset_sem
```

Barrier protocol:

```text
if N == 1:
    return

if logical_warp_id != 0:
    sem_inc(arrive_sem)        // I arrived at barrier
    sem_dec(release_sem)       // wait for leader to release phase 1

    sem_inc(depart_sem)        // I consumed my release token
    sem_dec(reset_sem)         // wait until leader confirms barrier generation reset
else:
    repeat N-1 times:
        sem_dec(arrive_sem)    // wait for all non-leaders

    repeat N-1 times:
        sem_inc(release_sem)   // release all non-leaders

    repeat N-1 times:
        sem_dec(depart_sem)    // wait until release tokens consumed

    repeat N-1 times:
        sem_inc(reset_sem)     // allow non-leaders into next generation
```

This protocol is intentionally more conservative than the minimal one-shot barrier. It is safe for repeated `__syncthreads()` calls and loops because non-leaders cannot enter the next barrier generation until the leader has observed that the previous release tokens were consumed.

### 10.4 Compiler lowering

Lower:

```mlir
gpu.barrier
```

or CUDA-like:

```c
__syncthreads();
```

to the above semaphore protocol.

### 10.5 Required scheduling invariant

The barrier only works if all participating logical QPU warps are resident. The runtime must guarantee:

```text
warps_per_block <= available QPU slots assigned to resident blocks
```

For a block with 12 warps, this means no other barrier-participating block can be resident at the same time.

### 10.6 What the barrier hardware test proved

The barrier test proved the following directly:

```text
same-slice synchronization works
cross-slice synchronization works
full 12-QPU block synchronization works
64 repeated barrier generations work
VPM writes before the barrier are visible after the barrier
multi-block partitioning works for 2 blocks × 4 warps/block
```

The full-block stress case is the key compiler-enabling case:

```text
1 block × 12 QPU warps × 64 iterations
observed_qpu_mask = 0xfff
every logical warp always saw all 12 block participants
mismatches = 0
timeouts = 0
```

The multi-block case is the key occupancy-enabling case:

```text
2 blocks × 4 QPU warps/block × 32 iterations
observed_qpu_mask = 0xff
each block saw its own 4 block-local warps
data_mismatches = 0
```

### 10.7 Barrier memory-ordering interpretation

For the compiler’s first shared-memory implementation, assume:

```text
VPM writes before __syncthreads() are visible to all block warps after __syncthreads().
```

This interpretation is valid for the tested pattern:

```text
all block warps write distinct VPM rows
__syncthreads()
all block warps read the block's VPM rows
```

Do not generalize this to arbitrary global memory operations yet. The test validates VPM shared-memory synchronization, not global memory fencing.

### 10.8 Divergent barriers are not supported

CUDA requires `__syncthreads()` to be reached by all non-exited threads in the block. The VC4 implementation is even stricter operationally:

```text
Every logical QPU warp in the block must execute every barrier generation.
```

The compiler should reject or conservatively transform barriers inside data-dependent divergent control flow unless it can prove uniform participation.

---

## 11. Global memory lowering

### 11.1 Loads

The natural path for global memory loads is TMU direct memory lookup. For direct memory lookup:

```text
write address to TMU s register
signal TMU read
consume result from r4
```

Use this for per-lane 32-bit loads. Optimize coalescing and uniform address patterns later.

### 11.2 Stores

VC4 has no CUDA-like native scalar global store instruction. The usual path is:

```text
QPU registers -> VPM -> VDW DMA store -> memory
```

This favors:

```text
coalesced contiguous vector stores
affine strided stores
row/column stores through VPM
```

Initially reject or slow-path:

```text
arbitrary per-lane scatter stores
uncoalesced byte stores
atomics
```

### 11.3 VDW setup serialization

VDW/VPM setup registers are shared enough that unprotected concurrent setup/store sequences are unsafe unless proven otherwise by a setup-clobber test.

Initial backend rule:

```text
Protect VPM setup + VPM access + VDW setup + VDW address + VDW wait sequences with the global QPU mutex.
```

This is conservative and may reduce performance, but it avoids false correctness failures while the compiler is being built.

Later, after a dedicated setup-clobber test, relax this rule if safe.

### 11.4 Global memory ordering

The `__syncthreads()` barrier should be treated as a VPM shared-memory barrier for now. Do not treat it as a complete global memory fence until separate tests establish the desired ordering for:

```text
TMU loads
VDW stores
host-visible memory
multiple QPUs writing disjoint global regions
multiple QPUs writing adjacent/global coalesced regions
```

---

## 12. Divergence and control flow

A QPU is SIMD, not CUDA SIMT with independent per-lane program counters.

Lower divergent control flow using:

```text
active masks
predicated ALU writes
predicated VPM/global stores
if-conversion where profitable
```

Uniform control flow can branch normally. Per-lane divergent control flow should be represented as masks.

Do not claim full CUDA SIMT semantics unless the backend implements a robust reconvergence/masking scheme.

### 12.1 Barrier-specific divergence rule

The compiler must ensure that `__syncthreads()` is executed uniformly by all logical warps in the block. A barrier inside a per-lane conditional is not legal unless the condition is proven uniform across the whole block or rewritten so all warps execute the barrier.

---

## 13. Register/private memory model

### 13.1 Private values

Per-thread private scalar values become vector registers:

```text
one logical scalar per CUDA thread -> one 16-lane QPU vector value
```

### 13.2 Register hazards

The guide documents QPU instruction restrictions, including no immediate read from a physical regfile location written by the previous instruction. Accumulators avoid some of these hazards.

The code generator must include a VC4 hazard scheduler that handles:

- physical regfile read-after-write restrictions,
- branch delay slots,
- thread-end delay slots,
- SFU result latency and `r4` restrictions,
- TMU result latency and `r4` use,
- VPM read setup latency,
- final instructions not accessing uniforms/VPM/VDW/VDR,
- semaphore instructions not simultaneously targeting closely-coupled peripherals that can stall.

### 13.3 Spills

Do not initially promise large private memory. Spills are expensive and likely require either:

- global memory via TMU/VDW sequences,
- VPM scratch rows,
- or recomputation.

For an early backend, reject kernels whose register pressure requires spilling.

---

## 14. MLIR lowering recommendations

### 14.1 Target properties

Expose or internally assume:

```text
subgroup_size = 16
max_workgroup_size = 192
max_workgroup_memory = 4096 bytes minus compiler reserve
num_multiprocessors = 1
```

If a public CUDA-like API requires a `warpSize` property, expose:

```text
warpSize = 16
```

Do not pretend the warp size is 32 unless the compiler explicitly emulates a 32-lane warp as two QPU warps.

### 14.2 Map MLIR GPU constructs

| MLIR construct | VC4 lowering |
|---|---|
| `gpu.thread_id x/y/z` | derived from `logical_warp_id`, `ELEMENT_NUMBER`, and block dimensions |
| `gpu.block_id x/y/z` | uniform |
| `gpu.block_dim x/y/z` | uniform or compile-time constant |
| `gpu.grid_dim x/y/z` | uniform |
| `gpu.barrier` | locked four-semaphore reusable QPU barrier |
| workgroup memory attribution | VPM row allocation |
| private memory attribution | QPU registers, reject/spill if too large |
| subgroup operations | QPU horizontal vector operations/rotates where possible |
| global loads | TMU direct memory lookup |
| global stores | VPM + VDW DMA store |

### 14.3 Preferred kernel shapes

Good first targets:

```text
elementwise tensor ops
coalesced loads/stores
affine linalg tiles
small reductions
row-wise reductions
stencils with simple shared-memory halos
matrix/vector kernels with explicit VPM tiling
block reductions using __syncthreads()
```

Hard targets:

```text
arbitrary CUDA C
heavy divergence
large shared memory
shared-memory atomics
global atomics
uncoalesced scatter stores
large private arrays
divergent barriers
```

---

## 15. Runtime resource allocation

For each cooperative kernel launch:

```text
1. Decode hardware topology from V3D_IDENT1.
2. Assert qpus_per_slice * num_slices >= required active QPUs.
3. Set or verify V3D_VPMBASE = 16 while idle.
4. Compute warps_per_block.
5. Compute rows_per_block.
6. Compute semaphores_per_block.
7. Compute resident_blocks.
8. For each resident block:
       assign VPM row range
       assign semaphore IDs
       prepare one uniform stream per logical QPU warp
9. Queue all QPU requests in the resident wave.
10. Wait for completions.
11. Reclaim VPM rows/semaphores.
12. Launch next resident wave.
```

For kernels without barriers/shared memory, the runtime can be more relaxed and stream QPU requests through the hardware scheduler.

### 15.1 Resource reclamation rule

For cooperative blocks, reclaim resources only after every QPU request in the resident wave has completed:

```text
VPM rows:       reusable after block completion
semaphore IDs:  reusable after block completion
uniform memory: reusable after block completion
output staging: reusable after block completion
```

Do not recycle semaphore sets early. A reusable barrier depends on generation counts returning to the expected drained state.

### 15.2 Error/status checking

Tests should continue checking:

```text
V3D_SRQCS completion count
V3D_SRQCS queue error bit
V3D_ERRSTAT relevant error bits
QPU host interrupt mask, where useful
reported physical QPU masks, where useful
```

The hardware tests have observed `V3D_ERRSTAT = 0x1000` before and after runs. This corresponds to the VCD idle bit in the documented error/status register layout and is not by itself a relevant error. The test harnesses correctly check that relevant error bits do not change.

---

## 16. Conservative implementation rules

Use these rules until more tests prove they can be relaxed:

```text
1. Warp size is always 16.
2. Physical QPU number is never logical warp id.
3. Shared memory is one 4 KiB global VPM pool.
4. Slices are not SMs.
5. Barrier-enabled blocks must be fully resident.
6. Reusable barriers use four semaphores per resident block.
7. VPM/VDW setup and access sequences are protected by the global mutex.
8. Workgroup memory is row-granular and initially 32-bit vector-oriented.
9. Arbitrary shared-memory scatter/gather is rejected or slow-pathed.
10. Global stores are limited to coalesced/affine patterns initially.
11. Kernels requiring native atomics are rejected or explicitly emulated.
12. Kernels requiring spills are rejected until spill lowering is implemented.
13. Barriers inside divergent control flow are rejected unless uniform participation is proven.
14. Semaphore IDs are allocated per resident block and not reused until completion.
15. VPM rows are allocated per resident block and not shared between blocks except by explicit runtime design.
```

---

## 17. Tests that gate compiler features

### 17.1 Already established

#### 1. Targeted `qpu_num` test

Confirms:

```text
individual physical QPUs are runnable
QPU_NUMBER is read correctly when QPU reservations target one QPU
short-kernel launch order is not physical QPU order
```

Compiler consequence:

```text
Use uniforms for logical warp_id.
Use QPU_NUMBER only for diagnostics/profiling/tests.
```

#### 2. VPM slice visibility test

Confirms:

```text
user-visible VPM storage is global across slices
cross-slice same-row collisions behave as global last-writer-wins
there are not independent 4 KiB VPM windows per slice
```

Compiler consequence:

```text
Model shared memory as one 4 KiB per-SM pool.
Do not map slices to CUDA SMs.
Partition VPM rows in software across resident blocks.
```

#### 3. Semaphore barrier / `__syncthreads()` test

Confirms:

```text
same-slice barrier works
cross-slice barrier works
full 12-QPU block barrier works
64 repeated barrier generations work
2 resident blocks with partitioned semaphores/VPM rows work
VPM writes before barrier are visible after barrier under the tested protocol
```

Compiler consequence:

```text
Lock the four-semaphore reusable barrier as the first __syncthreads() implementation.
Barrier-enabled blocks must be scheduled as fully resident resident-block waves.
Multiple resident barrier blocks are allowed when QPU slots, VPM rows, and semaphore IDs are partitioned.
```

### 17.2 Required next tests

#### 1. VPM setup-clobber test

Purpose:

```text
Determine whether VPM read/write setup state is per-QPU, per-slice, or global.
```

Sketch:

```text
QPU A writes VPM setup for row A.
QPU B writes VPM setup for row B.
QPU A performs VPM_WRITE without reprogramming setup.
Check whether row A or row B receives QPU A's data.
```

Until this passes, keep mutex serialization around VPM setup/access and VDW setup/store sequences.

#### 2. Global-store correctness test

Purpose:

```text
Validate coalesced vector stores through VPM+VDW under the compiler's chosen serialization protocol.
```

Required cases:

```text
single QPU stores one row
multiple QPUs store disjoint rows
multiple QPUs store adjacent rows
multiple resident blocks store disjoint output regions
barrier before store
barrier after store if needed
```

#### 3. TMU global-load test

Purpose:

```text
Validate per-lane direct-address loads through TMU.
```

Required cases:

```text
coalesced loads
strided loads
boundary-masked loads
multiple QPUs issuing loads concurrently
loads followed by VPM shared-memory writes
loads surrounding __syncthreads() where applicable
```

#### 4. Shared-memory access-pattern tests

Purpose:

```text
Decide how much of CUDA-like shared memory can be supported directly.
```

Required cases:

```text
row-contiguous stores/loads
affine row/column accesses
vertical VPM access modes
small transpose tile
block reduction through VPM
irregular scatter/gather slow path or rejection behavior
```

### 17.3 Recommended regression expansions

The barrier test has passed the production-critical smoke cases. Add these as regression expansions before aggressively using high occupancy across many block shapes:

```text
1 block × 1..12 warps
2 blocks × 1..6 warps/block
3 blocks × 1..4 warps/block
4 blocks × 1..3 warps/block
6 blocks × 1..2 warps/block
12 blocks × 1 warp/block
```

For each case:

```text
repeat many barrier generations
write distinct VPM rows before each barrier
read/check block-local rows after each barrier
verify no cross-block contamination
verify observed QPU mask is plausible
verify no timeouts
verify no relevant ERRSTAT changes
```

---

## 18. `__syncthreads()` implementation contract

This section is the compiler/runtime contract for the locked implementation.

### 18.1 Compiler obligations

The compiler must:

```text
1. compute warps_per_block = ceil(block_threads / 16),
2. ensure warps_per_block <= 12,
3. lower each logical warp to one QPU program instance,
4. lower lane id to ELEMENT_NUMBER,
5. lower logical warp id to a uniform,
6. lower workgroup memory to VPM row/column addresses,
7. lower gpu.barrier / __syncthreads() to the four-semaphore protocol,
8. ensure the barrier is reached by every logical warp in the block,
9. mask inactive lanes in partial final warps,
10. avoid combining semaphore ops with other closely-coupled peripheral accesses.
```

### 18.2 Runtime obligations

The runtime must:

```text
1. allocate a distinct VPM row range per resident block,
2. allocate four distinct semaphores per resident block that uses barriers,
3. initialize or require semaphore counts to be drained before use,
4. prepare one uniform stream per logical warp,
5. enqueue all warps of all resident barrier blocks as one resident wave,
6. not enqueue more barrier blocks than resource constraints allow,
7. wait for all wave QPU requests to complete before reusing VPM/semaphore resources,
8. treat physical QPU assignment as nondeterministic,
9. avoid using QPU_NUMBER for normal program indexing.
```

### 18.3 Kernel author / frontend obligations

If exposing a CUDA-like frontend, document these constraints:

```text
warpSize is 16, not 32.
max block size is 192 logical threads.
shared memory is at most 4 KiB per SM, and may be less per block under occupancy.
__syncthreads() must be reached uniformly by the block.
arbitrary shared-memory scatter/gather may be unsupported or slow.
arbitrary global scatter stores may be unsupported or slow.
native atomics are not available in the first implementation.
```

---

## 19. Final mental model

Use this model when designing the MLIR backend:

```text
VC4 is one CUDA-like SM.

The SM has:
    12 physical QPU warp slots,
    16 lanes per warp,
    192 max logical threads resident,
    4 KiB shared memory total,
    16 counting semaphores,
    one global QPU mutex,
    TMU-based global loads,
    VPM/VDW-based global stores.

A CUDA-like block is:
    1..12 QPU warp programs,
    scheduled as a full resident group if it uses barriers/shared memory,
    assigned a VPM row range,
    assigned semaphore IDs,
    indexed by uniforms plus ELEMENT_NUMBER.

__syncthreads() is:
    a reusable four-semaphore barrier among all resident logical warps in the block,
    hardware-validated for same-slice, cross-slice, full-block, repeated-generation,
    and two-resident-block partition cases.
```

This is the closest CUDA-like abstraction that remains faithful to measured VC4 behavior.

---

## 20. One-page implementation checklist

### Device constants

```c
VC4_CUDA_SM_COUNT              = 1;
VC4_CUDA_WARP_SIZE             = 16;
VC4_CUDA_MAX_WARPS_PER_SM      = 12;
VC4_CUDA_MAX_THREADS_PER_SM    = 192;
VC4_CUDA_MAX_THREADS_PER_BLOCK = 192;
VC4_CUDA_SHARED_BYTES_PER_SM   = 4096;
VC4_CUDA_VPM_ROWS              = 64;
VC4_CUDA_VPM_ROW_BYTES         = 64;
VC4_CUDA_SEMAPHORES            = 16;
VC4_CUDA_BARRIER_SEMS_PER_BLOCK= 4;
```

### Kernel admission checks

```c
warps_per_block = ceil(block_threads / 16);
if (warps_per_block > 12) reject;

rows_per_block = user_shared_rows + compiler_scratch_rows;
if (rows_per_block > 64) reject;

if (needs_barrier && barrier_is_not_uniform) reject;
if (needs_barrier && warps_per_block > 12) reject;
```

### Occupancy

```c
resident_blocks_by_qpus = 12 / warps_per_block;
resident_blocks_by_vpm  = 64 / rows_per_block;
resident_blocks_by_sems = needs_barrier ? 16 / 4 : UINT_MAX;

resident_blocks = min(resident_blocks_by_qpus,
                      resident_blocks_by_vpm,
                      resident_blocks_by_sems);

if (resident_blocks == 0) reject;
```

### Per-wave scheduling

```text
for each resident wave:
    allocate VPM rows per resident block
    allocate semaphore IDs per barrier block
    build uniforms per logical warp
    enqueue all logical warps in the wave
    wait for all completions
    reclaim VPM rows and semaphore IDs
```

### Per-QPU warp program

```text
read uniforms
lane = ELEMENT_NUMBER
flat_thread = logical_warp_id * 16 + lane
active = flat_thread < block_threads
execute masked computation
use VPM rows for workgroup memory
use four-semaphore protocol for __syncthreads()
terminate cleanly with required delay slots
```

