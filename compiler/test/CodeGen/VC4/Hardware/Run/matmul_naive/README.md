# matmul_naive

`matmul_naive` is the first CUDA-shaped matrix-multiply hardware-run golden for the VC4 backend.

It computes row-major single-precision matrix multiplication:

```text
C[M,N] = A[M,K] * B[K,N]
```

## Why this test is the next incremental step

The existing SAXPY tests cover independent elementwise work distribution.  This test keeps the same independent-vector launch style but moves to a nested-loop tensor kernel where each logical QPU warp computes a row/column tile of output elements.

The mapping follows the current VC4-as-CUDA model:

```text
one QPU user-program request = one CUDA-like warp
ELEMENT_NUMBER               = lane id, 0..15
qpu_id uniform               = logical warp id for grid-stride row distribution
num_qpus uniform             = active logical warp count
```

For each row assigned to a logical QPU, the 16 SIMD lanes compute one 16-column tile of `C`.  The kernel loops over `K` naively, using TMU direct memory loads for `A[row,k]` and `B[k,col+lane]`.

## Hardware path

The trusted reference qasm exercises:

- direct TMU0 memory loads for global `A` and `B`,
- QPU floating-point multiply/add accumulation,
- VPM staging for global stores,
- VDW DMA store to global `C`,
- dynamic VDW `DEPTH` for non-multiple-of-16 column tails,
- row grid-striding across all active QPUs,
- thread-end plus two delay slots.

VPM/VDW setup and store sequences are protected by the global QPU mutex.  This follows the conservative CUDA-mapping rule until the separate VPM setup-clobber test proves whether setup/access serialization can be narrowed.

## Tail and shape policy

`tail_policy = tail_safe`.

The harness includes zero-size dimensions, `K=0`, `N<16`, `N=16`, `N>16`, `M>active_qpus`, and a larger tail case.  The public launcher API has no shape knobs beyond semantic `m`, `n`, and `k`.

The launcher privately pads only the GPU-side `B` scratch allocation with a small guard region.  This keeps inactive tail-lane TMU reads within allocated GPU memory.  The logical matrix shape is still compact row-major `K*N`, and only logical `M*N` output elements are copied back.

## What this test proves

This test proves that a generated-style qasm/launcher pair can implement a naive CUDA-like matmul shape using:

```text
row = qpu_id + t * num_qpus
col = column_tile_base + ELEMENT_NUMBER
```

It also proves dynamic output tail stores for matrix columns and row work distribution across more rows than active QPUs.

## What it does not prove yet

This is not a tiled/shared-memory GEMM.  It does not use `gpu.barrier`, VPM tiling of `A`/`B`, or Pete Warden-style optimized GEMM blocking.  It also does not measure performance.

Candidate/codegen side remains disabled until generated bundles exist.
