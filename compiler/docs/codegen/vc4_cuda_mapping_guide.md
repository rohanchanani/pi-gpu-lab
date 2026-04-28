# VC4-as-CUDA Mapping Guide for an MLIR Backend

**Status:** Working and hardware-validated design for a CUDA-like abstraction over Raspberry Pi VideoCore IV / VC4 general-purpose QPU execution.

**Intended reader:** Compiler/runtime implementer building an MLIR backend that wants VC4 kernels to *look* as much like CUDA kernels as possible while preserving VC4 correctness.

**Core conclusion:** Model the Raspberry Pi VC4 GPU as **one tiny CUDA-like SM** with **12 physical warp slots**, **16 SIMD lanes per warp**, and **one global 4 KiB user-visible shared-memory window** backed by VPM.

**Runtime-image conclusion:** A VC4 “CUDA” program should have **one persistent device-visible program allocation**. That allocation owns the compiled kernel code blobs, kernel descriptors, uniform streams, uniform-pointer arrays, VPM/semaphore scheduling metadata, and any fixed test/runtime payloads. Each compiled kernel is copied into that allocation **once** during runtime/program setup. Later kernel launches reuse the same device code address and only rewrite the relevant uniform streams and scheduling metadata.

**Current validation status:**

```text
Targeted qpu_num test:                   PASS
VPM slice visibility test:               PASS
QPU semaphore barrier / __syncthreads(): PASS
```

The `__syncthreads()` implementation described here is considered **locked for first compiler implementation**, subject to the invariants in Sections 8, 10, 11, 16, and 17.

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

The `qpu_barrier_syncthreads` hardware test established that a four-semaphore reusable barrier can act as the first backend implementation of CUDA-like `__syncthreads()` when the runtime guarantees full residency of all participating block warps. The test passed:

```text
same-slice 2-warp smoke test
cross-slice 0↔1 2-warp test
cross-slice 0↔2 2-warp test
full 12-QPU / 12-warp block stress test, 64 iterations
two-resident-block partition test, 2 blocks × 4 warps/block, 32 iterations
```

### 1.3 Backend policy facts introduced by this specification

This compiler/runtime design additionally adopts these ABI/runtime policies:

```text
1. One persistent device-visible program allocation per compiled VC4 CUDA-like program.
2. One runtime/program struct owns all generated QPU kernel code fields.
3. Each kernel code blob is copied to device-visible memory exactly once during setup.
4. Later launches enqueue the already-resident code address using SRQPC.
5. Uniform streams are persistent fields inside the same allocation and are overwritten per launch.
6. Uniform-pointer arrays gain a kernel dimension: kernel_id × request_id.
7. Uniform streams are non-rectangular across kernels because different kernels may have different uniform counts.
```

This policy is conceptually similar to CUDA’s “first launch is slow” behavior, where driver setup, JIT compilation, module loading, and code upload may occur lazily. For this VC4 backend, that work should be done **eagerly** during runtime/program setup instead of being hidden inside the first kernel launch.

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
persistent program allocations:   1 per compiled program/runtime instance
kernel code residency:            copied once during setup, reused thereafter
```

This is the central abstraction:

```text
CUDA-ish device       -> one VC4 V3D/QPU complex
CUDA SM               -> the whole VC4 general-QPU execution domain
CUDA warp             -> one QPU user program, 16 SIMD lanes
CUDA lane             -> one QPU SIMD element
CUDA thread           -> one SIMD lane in one logical QPU warp
CUDA thread block     -> a cooperative group of 1..12 logical QPU warps
CUDA shared memory    -> a software allocation in the single global 4 KiB VPM window
CUDA module/program   -> one persistent VC4 program image allocation
CUDA kernel function  -> one code field / descriptor inside the persistent program image
CUDA kernel launch    -> rewrite uniforms, enqueue existing code PC, wait or stream according to mode
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
| `__syncthreads()` | QPU semaphore barrier | Use the validated reusable four-semaphore protocol; all participating warps must already be resident. |
| global load | TMU direct memory lookup | Natural path for 32-bit reads. |
| global store | QPU register → VPM → VDW DMA store | Efficient mainly for coalesced/affine vector stores. |
| private registers | QPU vector registers and accumulators | Respect QPU register hazards and delay restrictions. |
| atomics | not native in this model | Reject, emulate slowly with global mutex, or lower only special cases. |
| divergent SIMT control flow | SIMD masks/predication | No native CUDA-style per-lane PCs/reconvergence. |
| CUDA module / cubin | persistent VC4 program image | One device-visible allocation containing all kernel code blobs and descriptors. |
| kernel function pointer | `kernel_desc[k].code_gpu_addr` | Stable GPU bus address of copied QPU code. |
| kernel launch arguments | per-kernel uniform streams | Rewritten in persistent host-visible memory before enqueueing. |
| launch queue entry | `SRQUA` + `SRQPC` write | `SRQUA = uniform stream bus address`; `SRQPC = resident code bus address`. |

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

---

## 5. SM model and slices

### 5.1 Correctness model

For correctness, model the device as:

```text
one SM
12 QPU warp slots
one global VPM user window
one persistent runtime/program allocation
```

The runtime’s block scheduler should act like a CUDA occupancy scheduler for a single SM.

### 5.2 Performance model

VC4 slices still matter for performance:

```text
slice 0: QPU 0, 1, 2, 3
slice 1: QPU 4, 5, 6, 7
slice 2: QPU 8, 9, 10, 11
```

Each slice shares I-cache, SFU, TMU, and VRI-like resources. This can affect instruction-cache locality, TMU pressure, and SFU-heavy code. However, slices do not define separate shared-memory scopes.

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
    uint32_t barrier_go_sem;
    uint32_t barrier_depart_sem;
    uint32_t barrier_reset_sem;

    // kernel argument pointers/scalars follow
};
```

The exact layout can be backend-specific, but it must be stable and documented because every QPU request consumes its uniforms sequentially.

### 6.2 Kernel-specific uniform layouts

Different kernels may have different uniform counts and layouts. The compiler should generate one uniform-layout definition per kernel:

```c
#define KERNEL0_NUM_UNIFS  17
#define KERNEL1_NUM_UNIFS  29
#define KERNEL2_NUM_UNIFS   8
```

Uniform storage therefore has a kernel dimension and is usually non-rectangular:

```text
kernel 0: NUM_QPUS × KERNEL0_NUM_UNIFS
kernel 1: NUM_QPUS × KERNEL1_NUM_UNIFS
kernel 2: NUM_QPUS × KERNEL2_NUM_UNIFS
```

The runtime should never assume that all kernels have the same uniform stride.

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

The validated reusable barrier uses four semaphores per block:

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
3. Overwrite the persistent uniform streams for the selected kernel and requests.
4. Queue all QPU warp requests for those blocks using the already-resident kernel code address.
5. Wait for all requests in the wave to complete.
6. Reuse VPM/semaphore/uniform-request slots for the next wave.
```

For initial correctness, a one-block-at-a-time cooperative-block mode is acceptable:

```text
resident_blocks = 1
```

The barrier test has also validated a two-resident-block partition case. Multi-resident blocks are therefore allowed when the runtime correctly partitions QPU slots, VPM rows, and semaphore IDs.

---

## 10. Persistent program image and one-allocation runtime policy

### 10.1 Core rule

A VC4 CUDA-like program should perform **one device-visible allocation** for the compiled program/runtime image:

```text
one allocation -> one struct vc4_gpu / struct qpu / program image
```

That allocation owns all compiler-generated QPU launch state:

```text
all kernel code blobs
all kernel descriptors
all per-kernel uniform streams
all per-kernel uniform-pointer arrays
all fixed runtime scheduling metadata
optional fixed payload/test buffers
```

Each compiled kernel is copied into the allocation **exactly once** during runtime/program setup. After that, every launch dispatches to the same resident code location by writing that code address to the QPU scheduler.

### 10.2 Scope of the “one allocation” rule

The hard rule for this backend is:

```text
No per-kernel-launch mem_alloc/mem_free for QPU code, uniform streams,
uniform-pointer arrays, kernel descriptors, or launch-control state.
```

For early tests and compiler bring-up, fixed input/output payloads should also be fields inside the same allocation when practical. If a future CUDA-like memory API supports dynamic user buffers, those buffers may become a separate memory-management layer. That should not change the rule that **kernel code and launch-control state are persistent and not reallocated per launch**.

### 10.3 Naming note: `struct qpu` versus `struct gpu`

If the project currently calls the device-visible launch struct `struct QPU` or `struct qpu`, keep that name for API continuity. Conceptually, though, this struct represents the whole CUDA-like VC4 GPU program image, not one physical QPU.

Recommended conceptual name:

```c
struct vc4_gpu_program;
```

Acceptable project-local names:

```c
struct qpu;
struct gpu;
struct vc4_cuda_state;
struct vc4_program_image;
```

The important invariant is not the name. The important invariant is:

```text
one persistent device-visible allocation owns all code + uniform launch state
```

### 10.4 Fixed-field layout for a known set of kernels

For a compiled program with a statically known kernel set, a fixed-field struct is straightforward and friendly to C:

```c
#define VC4_MAX_QPUS 12

#define KERNEL0_CODE_WORDS  128
#define KERNEL0_NUM_UNIFS    17

#define KERNEL1_CODE_WORDS  224
#define KERNEL1_NUM_UNIFS    29

struct vc4_kernel_desc {
    uint32_t code_gpu_addr;          // bus address to write to SRQPC
    uint32_t code_word_count;
    uint32_t unif_words_per_request;
    uint32_t max_requests_per_wave;  // usually <= 12
    uint32_t unif_gpu_addr;          // optional base address for diagnostics
    uint32_t unif_ptr_gpu_addr;      // optional base address for diagnostics
    uint32_t flags;
};

struct vc4_gpu_program {
    uint32_t magic;
    uint32_t total_size_bytes;
    uint32_t num_kernels;
    uint32_t active_qpus;
    uint32_t warp_size;
    uint32_t vpm_rows;

    struct vc4_kernel_desc kernel[2];

    // Kernel 0 code and launch state.
    uint32_t kernel0_code[KERNEL0_CODE_WORDS] __attribute__((aligned(8)));
    uint32_t kernel0_unif[VC4_MAX_QPUS][KERNEL0_NUM_UNIFS];
    uint32_t kernel0_unif_ptr[VC4_MAX_QPUS];

    // Kernel 1 code and launch state.
    uint32_t kernel1_code[KERNEL1_CODE_WORDS] __attribute__((aligned(8)));
    uint32_t kernel1_unif[VC4_MAX_QPUS][KERNEL1_NUM_UNIFS];
    uint32_t kernel1_unif_ptr[VC4_MAX_QPUS];

    // Optional fixed payload / scratch / result fields.
    uint32_t runtime_scratch[/* ... */];
};
```

This layout expresses the new uniform dimension directly:

```text
kernel_id -> request_id/QPU-warp slot -> uniform word
```

The arrays are non-rectangular across kernels because `KERNEL0_NUM_UNIFS` and `KERNEL1_NUM_UNIFS` may differ.

### 10.5 Descriptor/offset layout for arbitrary kernel counts

If the compiler wants one generic runtime struct for arbitrary programs, use descriptors plus offsets into a flexible storage area:

```c
struct vc4_kernel_desc {
    uint32_t code_word_offset;
    uint32_t code_word_count;

    uint32_t unif_word_offset;
    uint32_t unif_words_per_request;
    uint32_t max_requests_per_wave;

    uint32_t unif_ptr_word_offset;

    uint32_t code_gpu_addr;
    uint32_t flags;
};

struct vc4_gpu_program_header {
    uint32_t magic;
    uint32_t total_size_bytes;
    uint32_t num_kernels;
    uint32_t active_qpus;
    uint32_t warp_size;
    uint32_t vpm_rows;
    struct vc4_kernel_desc kernels[];

    // followed by aligned code/unif/unif_ptr storage
};
```

Then compute:

```c
uint32_t *code     = storage_words + desc->code_word_offset;
uint32_t *unif     = storage_words + desc->unif_word_offset;
uint32_t *unif_ptr = storage_words + desc->unif_ptr_word_offset;
```

The storage required for a kernel’s uniforms is:

```c
kernel_unif_words = max_requests_per_wave * unif_words_per_request;
```

The storage required for a kernel’s uniform pointers is:

```c
kernel_unif_ptr_words = max_requests_per_wave;
```

### 10.6 Runtime/program setup: eager code upload

Runtime/program setup should do all allocation and code copying eagerly:

```c
int vc4_cuda_program_setup(struct vc4_cuda_runtime *rt,
                           struct vc4_gpu_program **out)
{
    // 1. Initialize VC4 runtime and decode IDENT1.
    // 2. Compute total size for one program image allocation.
    // 3. mem_alloc exactly once for the program image.
    // 4. mem_lock and map it once.
    // 5. memset the program image.
    // 6. Copy each assembled kernel code blob into its assigned field/offset.
    // 7. Fill kernel descriptors, including code_gpu_addr.
    // 8. Precompute all unif_ptr[kernel][request] addresses.
    // 9. Set V3D_VPMBASE = 16 while V3D is idle.
    // 10. Clear instruction/uniform caches after initial code/uniform image setup.
    // 11. Return the persistent mapped program image.
}
```

After setup, the program image stays allocated and locked for the lifetime of the compiled program/runtime instance.

Do not do this in normal kernel launches:

```text
mem_alloc
mem_lock
copy qasm shader bytes
recompute code placement
mem_unlock
mem_free
```

### 10.7 Kernel launch: reuse resident code, overwrite uniforms

A kernel launch should be a scheduling operation, not a code-upload operation.

For each launch wave:

```c
int vc4_cuda_launch_kernel(struct vc4_gpu_program *gpu,
                           uint32_t kernel_id,
                           const struct launch_shape *shape,
                           const struct kernel_args *args)
{
    struct vc4_kernel_desc *desc = &gpu->kernel[kernel_id];

    // 1. Compute grid/block wave and occupancy.
    // 2. Assign VPM rows and semaphores for resident blocks.
    // 3. Overwrite desc's persistent uniform streams for each request in this wave.
    // 4. Clear/invalidate the uniforms cache for active slices.
    // 5. Queue requests:
    //        PUT32(V3D_SRQUA, unif_ptr[request]);
    //        PUT32(V3D_SRQPC, desc->code_gpu_addr);
    // 6. Wait for completions or stream next wave according to scheduling mode.
}
```

The code address written to `SRQPC` is stable:

```c
PUT32(V3D_SRQPC, gpu->kernel[kernel_id].code_gpu_addr);
```

The uniform address written to `SRQUA` changes by request:

```c
PUT32(V3D_SRQUA, gpu->kernel[kernel_id].unif_ptr[request]);
```

The uniform contents may change on every launch:

```c
gpu->kernel0_unif[request][0] = block_id_x;
gpu->kernel0_unif[request][1] = logical_warp_id;
gpu->kernel0_unif[request][2] = vpm_base_row;
gpu->kernel0_unif[request][3] = barrier_arrive_sem;
// ... kernel args ...
```

### 10.8 Uniform storage: kernel dimension, request dimension, word dimension

The old mental model for one test kernel was:

```text
unif[NUM_QPUS][NUM_UNIFS]
unif_ptr[NUM_QPUS]
```

The program-level model is:

```text
unif[kernel_id][request_id][uniform_word]
unif_ptr[kernel_id][request_id]
```

Because kernels can have different uniform counts, this is not naturally rectangular in C. Use one of these two implementations:

```text
Implementation A:
    Separate fields per kernel:
        kernel0_unif[NUM_QPUS][KERNEL0_NUM_UNIFS]
        kernel1_unif[NUM_QPUS][KERNEL1_NUM_UNIFS]
        ...

Implementation B:
    Flat storage plus descriptors:
        desc[k].unif_word_offset
        desc[k].unif_words_per_request
        desc[k].unif_ptr_word_offset
```

Both are correct. The fixed-field form is easier for generated C tests. The descriptor/offset form is better for a general MLIR runtime.

### 10.9 Multiple kernels in one program image

For a program with multiple kernels:

```text
kernel 0: elementwise_add
kernel 1: tiled_matmul
kernel 2: block_reduce
```

one allocation should contain all three code blobs:

```c
struct vc4_gpu_program {
    ...
    uint32_t elementwise_add_code[ELEMENTWISE_ADD_CODE_WORDS];
    uint32_t tiled_matmul_code[TILED_MATMUL_CODE_WORDS];
    uint32_t block_reduce_code[BLOCK_REDUCE_CODE_WORDS];

    uint32_t elementwise_add_unif[VC4_MAX_QPUS][ELEMENTWISE_ADD_NUM_UNIFS];
    uint32_t tiled_matmul_unif[VC4_MAX_QPUS][TILED_MATMUL_NUM_UNIFS];
    uint32_t block_reduce_unif[VC4_MAX_QPUS][BLOCK_REDUCE_NUM_UNIFS];

    uint32_t elementwise_add_unif_ptr[VC4_MAX_QPUS];
    uint32_t tiled_matmul_unif_ptr[VC4_MAX_QPUS];
    uint32_t block_reduce_unif_ptr[VC4_MAX_QPUS];
};
```

Launching different kernels only changes which code address is enqueued:

```c
launch kernel 0 -> SRQPC = gpu_addr(program->elementwise_add_code)
launch kernel 1 -> SRQPC = gpu_addr(program->tiled_matmul_code)
launch kernel 2 -> SRQPC = gpu_addr(program->block_reduce_code)
```

No code blob is recopied for repeated calls.

### 10.10 Cache and coherency requirements

Because code and uniforms are reused at stable addresses, cache handling is part of the ABI.

#### Kernel code

After copying kernel code during setup:

```text
clear instruction caches for active slices
optionally clear L2 according to the memory/coherency mode
```

If kernel code is never modified again, the runtime should not need to clear instruction caches merely because the kernel is launched again.

If a kernel is hot-patched or reassembled into the same code field, then the runtime must treat that like a new code upload:

```text
write code
flush/clean CPU side if needed
clear VC4 instruction cache for active slices
clear relevant L2 state if needed
```

#### Uniform streams

Uniform streams are intentionally overwritten per launch. Since VC4 has a uniforms cache per slice, the conservative rule is:

```text
After the host overwrites uniform streams at addresses that may have been used before,
clear the uniforms cache for active slices before queueing QPU requests.
```

A conservative launch can clear all slice caches as existing tests often do:

```c
PUT32(V3D_SLCACTL, 0xffffffffu);
```

A production runtime can narrow this to the relevant Uniforms Cache Clear bits for active slices once that is tested.

If the runtime uses a ring of never-reused uniform addresses, uniform-cache clearing may be relaxed later. The first implementation should not rely on that optimization.

#### Global data buffers

If the CPU writes global input buffers that TMU reads, the runtime must ensure those writes are visible to VC4 before launch. If the QPUs write output buffers through VDW, the runtime must ensure those writes are visible to the CPU before host-side validation/consumption.

The exact CPU/GPU cache protocol depends on the memory flags and address aliases used by the project. For this backend’s correctness contract, record the chosen policy explicitly in the runtime.

### 10.11 CUDA first-launch analogy

CUDA often has a slow first kernel launch because the driver may initialize context state, JIT compile code, load modules, allocate internal resources, and copy code to the device lazily.

The VC4 backend should not hide that work inside the first launch. It should do it eagerly:

```text
runtime/program setup:
    assemble or receive already-assembled QASM
    allocate one program image
    copy all kernel code once
    set up descriptors and uniform arrays
    configure VPM reservation
    clear caches after setup

kernel launches:
    update uniforms
    assign VPM/semaphore resources
    enqueue resident code address
```

The conceptual analogy is still useful:

```text
CUDA first-launch overhead      ~ VC4 runtime/program setup
CUDA later kernel launches      ~ VC4 SRQUA/SRQPC scheduling of resident code
```

### 10.12 Prohibited production launch behavior

Production kernel launch code should not:

```text
allocate a new struct qpu/gpu per launch
copy the same kernel code blob per launch
free the QPU launch allocation after each launch
compute code placement per launch
assume all kernels have the same uniform count
use physical QPU number as logical warp id
use launch order as physical QPU assignment
```

### 10.13 Recommended persistent launch-state invariants

The runtime should maintain these invariants:

```text
program_image != NULL for the lifetime of the VC4 CUDA-like program
program_image_handle is locked while kernels may launch
kernel_desc[k].code_gpu_addr is stable after setup
kernel_desc[k].unif_words_per_request is stable after setup
kernel_desc[k].unif_ptr[request] is stable after setup
unif contents may change every launch
VPM row allocations are assigned per resident wave
semaphore allocations are assigned per resident wave
```

---

## 11. Barrier mapping: `__syncthreads()`

### 11.1 Hardware primitive

VC4 provides sixteen system-wide 4-bit counting semaphores. A semaphore increment stalls if the count is 15; a decrement stalls if the count is 0.

### 11.2 Validation status

The first compiler implementation of `__syncthreads()` is locked to the four-semaphore reusable barrier protocol below.

The `qpu_barrier_syncthreads` hardware test passed these cases:

```text
same_slice_smoke:
    1 block × 2 warps, repeated 4 times
    observed mask matched expected mask
    mismatches = 0

cross_slice_0_1:
    1 block × 2 warps across slice 0 and slice 1
    mismatches = 0

cross_slice_0_2:
    1 block × 2 warps across slice 0 and slice 2
    mismatches = 0

full_block_stress:
    1 block × 12 warps
    all 12 QPUs participated
    64 barrier iterations
    every warp saw all 12 rows after every barrier
    mismatches = 0

two_block_partition:
    2 resident blocks × 4 warps/block
    32 barrier iterations
    block-local VPM/semaphore partitioning worked
    mismatches = 0
```

Therefore the compiler/runtime can use this barrier for:

```text
block-wide ordering among resident logical QPU warps
VPM shared-memory visibility before/after __syncthreads()
repeated barriers and loop barriers
multiple resident blocks when semaphores and VPM rows are partitioned
```

### 11.3 Recommended reusable barrier

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

### 11.4 Compiler lowering

Lower:

```mlir
gpu.barrier
```

or CUDA-like:

```c
__syncthreads();
```

to the above semaphore protocol.

### 11.5 Required scheduling invariant

The barrier only works if all participating logical QPU warps are resident. The runtime must guarantee:

```text
warps_per_block <= available QPU slots assigned to resident blocks
```

For a block with 12 warps, this means no other barrier-participating block can be resident at the same time.

### 11.6 Semaphore ownership invariant

Each resident block using barriers owns a distinct four-semaphore set:

```text
block 0: semaphores  0..3
block 1: semaphores  4..7
block 2: semaphores  8..11
block 3: semaphores 12..15
```

or an equivalent allocator result.

Do not reuse a block’s semaphores until all QPU requests for that block have completed.

### 11.7 Barrier participation invariant

CUDA-like `__syncthreads()` is only valid when every logical warp of the block reaches the same barrier generation.

Allowed:

```text
barrier in uniform control flow
barrier reached by all logical warps
partial final warp participates even if some lanes are inactive
```

Rejected or transformed:

```text
barrier in divergent control flow where some logical threads/warps may skip it
early return before a later barrier unless the compiler proves all block warps return uniformly
```

---

## 12. Global memory lowering

### 12.1 Loads

The natural path for global memory loads is TMU direct memory lookup. For direct memory lookup:

```text
write address to TMU s register
signal TMU read
consume result from r4
```

Use this for per-lane 32-bit loads. Optimize coalescing and uniform address patterns later.

### 12.2 Stores

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

### 12.3 VDW setup serialization

VDW/VPM setup registers are shared enough that unprotected concurrent setup/store sequences are unsafe unless proven otherwise by a setup-clobber test.

Initial backend rule:

```text
Protect VPM setup + VPM access + VDW setup + VDW address + VDW wait sequences with the global QPU mutex.
```

This is conservative and may reduce performance, but it avoids false correctness failures while the compiler is being built.

Later, after a dedicated setup-clobber test, relax this rule if safe.

---

## 13. Divergence and control flow

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

---

## 14. Register/private memory model

### 14.1 Private values

Per-thread private scalar values become vector registers:

```text
one logical scalar per CUDA thread -> one 16-lane QPU vector value
```

### 14.2 Register hazards

The guide documents QPU instruction restrictions, including no immediate read from a physical regfile location written by the previous instruction. Accumulators avoid some of these hazards.

The code generator must include a VC4 hazard scheduler that handles:

- physical regfile read-after-write restrictions,
- branch delay slots,
- thread-end delay slots,
- SFU result latency and `r4` restrictions,
- TMU result latency and `r4` use,
- VPM read setup latency,
- final instructions not accessing uniforms/VPM/VDW/VDR.

### 14.3 Spills

Do not initially promise large private memory. Spills are expensive and likely require either:

- global memory via TMU/VDW sequences,
- VPM scratch rows,
- or recomputation.

For an early backend, reject kernels whose register pressure requires spilling.

---

## 15. MLIR lowering recommendations

### 15.1 Target properties

Expose or internally assume:

```text
subgroup_size = 16
max_workgroup_size = 192
max_workgroup_memory = 4096 bytes minus compiler reserve
num_multiprocessors = 1
num_kernel_code_uploads_per_program_setup = num_kernels
num_kernel_code_uploads_per_launch = 0
```

### 15.2 Map MLIR GPU constructs

| MLIR construct | VC4 lowering |
|---|---|
| `gpu.thread_id x/y/z` | derived from `logical_warp_id`, `ELEMENT_NUMBER`, and block dimensions |
| `gpu.block_id x/y/z` | uniform |
| `gpu.block_dim x/y/z` | uniform or compile-time constant |
| `gpu.grid_dim x/y/z` | uniform |
| `gpu.barrier` | validated four-semaphore reusable QPU barrier |
| workgroup memory attribution | VPM row allocation |
| private memory attribution | QPU registers, reject/spill if too large |
| subgroup operations | QPU horizontal vector operations/rotates where possible |
| global loads | TMU direct memory lookup |
| global stores | VPM + VDW DMA store |
| kernel symbol | index into persistent `kernel_desc[]` |
| kernel launch operands | writes to that kernel’s persistent uniform streams |

### 15.3 Preferred kernel shapes

Good first targets:

```text
elementwise tensor ops
coalesced loads/stores
affine linalg tiles
small reductions
row-wise reductions
stencils with simple shared-memory halos
matrix/vector kernels with explicit VPM tiling
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
```

---

## 16. Runtime resource allocation and launch lifecycle

### 16.1 Runtime/program setup lifecycle

Program setup is where allocation and code copying happen:

```text
1. Decode hardware topology from V3D_IDENT1.
2. Verify expected QPU/VPM/semaphore resources.
3. Compute total persistent program-image size.
4. Allocate exactly one device-visible program image.
5. Lock/map that allocation for host access.
6. Copy every assembled kernel code blob into its assigned code field/offset.
7. Fill kernel descriptors and stable code_gpu_addr values.
8. Precompute per-kernel uniform-pointer arrays.
9. Set or verify V3D_VPMBASE = 16 while idle.
10. Clear relevant VC4 caches after setup.
11. Keep this allocation alive until program/runtime shutdown.
```

After this lifecycle completes, kernel code is resident.

### 16.2 Cooperative kernel launch lifecycle

For each cooperative kernel launch:

```text
1. Look up kernel descriptor by kernel_id.
2. Compute warps_per_block.
3. Compute rows_per_block.
4. Compute semaphores_per_block.
5. Compute resident_blocks.
6. For each resident block:
       assign VPM row range
       assign semaphore IDs
       assign logical block IDs
7. For each logical QPU warp request in the resident wave:
       overwrite the already-allocated uniform stream for this kernel/request
8. Clear/invalidate the uniforms cache for active slices.
9. Queue all QPU requests in the resident wave:
       SRQUA = persistent unif_ptr[kernel_id][request]
       SRQPC = persistent kernel_desc[kernel_id].code_gpu_addr
10. Wait for completions.
11. Reclaim logical VPM row/semaphore allocations for the next wave.
12. Launch next resident wave.
```

### 16.3 Independent-vector launch lifecycle

For kernels without barriers/shared memory, the runtime can be more relaxed and stream QPU requests through the hardware scheduler:

```text
1. Look up resident kernel code address.
2. Fill persistent uniform streams for a batch of independent vector requests.
3. Clear uniforms cache for active slices.
4. Enqueue requests using the resident code address.
5. Poll completions and continue streaming as appropriate.
```

Still do not allocate or copy code per launch.

### 16.4 Shutdown lifecycle

At runtime/program shutdown:

```text
1. Ensure all queued QPU programs have completed.
2. Clear QPU reservations if the runtime changed them.
3. Release or restore any persistent V3D state owned by the runtime.
4. Unlock and free the one persistent program-image allocation.
```

---

## 17. Conservative implementation rules

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
13. The runtime/program image uses one persistent device-visible allocation.
14. Every kernel code blob is copied into that allocation exactly once during setup.
15. Kernel launches reuse resident code addresses; they do not copy code.
16. Uniform streams are persistent and may be overwritten per launch.
17. Uniform cache invalidation/clearing is required after overwriting reused uniform streams.
18. Uniform storage has a kernel dimension and a request/QPU-warp dimension.
19. Different kernels may have different uniform counts; do not force a rectangular uniform layout unless using a max-stride padding policy intentionally.
20. The launch path must not call mem_alloc/mem_free for code or launch-control state.
```

---

## 18. Tests that should gate compiler features

### 18.1 Already established

1. **Targeted `qpu_num` test**
   - Confirms individual physical QPUs are runnable.
   - Confirms `QPU_NUMBER` is read correctly when QPU reservations target one QPU.
   - Confirms normal kernels should use logical warp IDs from uniforms rather than physical `QPU_NUMBER`.

2. **VPM slice visibility test**
   - Confirms user-visible VPM storage is global across slices.
   - Confirms cross-slice same-row collisions behave as global last-writer-wins.
   - Confirms slices should not be modeled as independent CUDA SMs.

3. **Semaphore barrier / `__syncthreads()` test**
   - Confirms the four-semaphore reusable barrier works same-slice and cross-slice.
   - Confirms a full 12-warp block can repeatedly synchronize and observe VPM writes.
   - Confirms two resident blocks can synchronize independently when VPM rows and semaphore IDs are partitioned.
   - Locks the first compiler implementation of `gpu.barrier` / `__syncthreads()`.

### 18.2 Required next tests

1. **Persistent program image / resident code reuse test**
   - Allocate exactly one program image.
   - Copy at least two kernels into separate code fields during setup.
   - Launch kernel A, then kernel B, then kernel A again.
   - Verify no per-launch code copy or allocation occurs.
   - Verify all launches dispatch using the same stable code addresses.
   - Verify different arguments are passed by overwriting the already-allocated uniform streams.
   - Verify cache clearing policy is sufficient when uniforms are overwritten at the same addresses.

2. **Uniform-cache reuse test**
   - Use one kernel and one persistent uniform stream address.
   - Launch with argument value X.
   - Overwrite the same uniform stream with argument value Y.
   - Clear uniforms cache according to the runtime policy.
   - Relaunch and verify the QPU observes Y, not X.
   - Optionally run a negative/diagnostic variant without uniform-cache clearing to determine whether stale uniform data can be observed.

3. **VPM setup-clobber test**
   - Deliberately interleave VPM setup from one QPU with VPM access from another.
   - Determine whether setup state is per-QPU, per-slice, or global.
   - Until this passes, keep mutex serialization around VPM setup/access.

4. **Global-store correctness test**
   - Coalesced vector stores through VPM+VDW.
   - Multiple QPUs store to disjoint output regions.
   - Verify no VDW setup race under chosen serialization protocol.

5. **TMU global-load test**
   - Per-lane direct-address loads.
   - Coalesced and strided patterns.
   - Boundary-mask behavior.

6. **Persistent multi-kernel ABI test**
   - Generate a program with kernels that have different uniform counts.
   - Store them in one persistent program allocation.
   - Verify `kernel0_unif[request][K0_NUM_UNIFS]` and `kernel1_unif[request][K1_NUM_UNIFS]` or descriptor/offset equivalents are addressed correctly.
   - Verify launches do not assume a rectangular `num_kernels × NUM_QPUS × max_num_unifs` layout unless the runtime intentionally pads to max stride.

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

A VC4 CUDA-like program image is:
    one persistent device-visible allocation,
    containing every kernel's QPU code blob,
    containing every kernel's uniform streams and uniform-pointer arrays,
    initialized once during runtime/program setup,
    reused for every kernel launch.

A kernel launch is:
    choose a resident kernel descriptor,
    overwrite persistent uniforms for this launch/wave,
    clear uniforms cache as required,
    enqueue SRQUA/SRQPC pairs using the stable resident code address,
    wait or stream according to independent/cooperative scheduling mode.
```

This is the closest CUDA-like abstraction that remains faithful to measured VC4 behavior while giving the compiler a clean implementation target.
