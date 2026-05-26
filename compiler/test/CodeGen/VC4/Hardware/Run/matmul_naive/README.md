# matmul_naive

`matmul_naive` is a generated VC4 hardware-run test for a canonical row-major
`f32` matrix multiply:

```text
C[M,N] = A[M,K] * B[K,N]
```

The generated kernel takes its loop bounds from runtime shape arguments:

```text
M, N, K are runtime u32 kernel arguments
```

There are no baked-in kernel shape limits.  The harness sizes ARM host buffers
with `kmalloc` and VC4 device buffers with the generated-runtime `vc4_m2_malloc`
path from the actual sweep cases; larger cases require enough host and device
memory but do not require shape-specific MLIR edits.

## Device Mapping

The fixture uses the current VC4-as-CUDA independent-vector launch model:

```text
one QPU user-program request = a strided set of output rows
ELEMENT_NUMBER               = column lane within a 16-wide tile
logical_request uniform      = initial row id
total_requests uniform       = row stride
```

Each request walks rows by `total_requests`, tiles columns in 16-wide VPM/VDW
stores, and loops over runtime `K`.  The kernel uses TMU direct memory reads for
`A[row,kk]` and `B[kk,col]`, performs the canonical `fmul`/`fadd` accumulation
on device, and uses a dynamic final-tile store depth for `N % 16`.

The harness launches 12 row requests for every case.  Initial requests with
`row >= m` exit before any TMU or VDW access; valid requests then walk rows by
the runtime row stride.

## Verification

The candidate harness sweeps eight runtime `(m,n,k)` cases, including hundreds
scale `M`, `N`, and `K`, partial row strides, partial final column tiles, `K=0`,
and single-column output.  For each case it:

- fills compact row-major `A` and `B` buffers deterministically,
- computes an unweakened ARM CPU oracle with the same canonical loop nest,
- allocates device buffers with `vc4_m2_malloc`, not the ARM host heap,
- compares every live output by exact `f32` bits,
- checks the compact output tail and final guard region are untouched,
- requires generated-runtime allocation, launch, code-upload, and failure
  counters to match the sweep.

Inactive launched rows do not write.  The output tail and final guard region
are still checked for exact guard-word preservation.

## Current Hardware Result

```text
VC4_TEST_RESULT name=matmul_naive status=PASS cases=8 checked_elements=127421 total_mismatches=0 guard_mismatches=0 launch_failures=0 recorded_launch_failures=0 active_qpus=12 lanes=16 sweep_max_m=192 sweep_max_n=224 sweep_max_k=192 backing_a_floats=36864 backing_b_floats=41065 backing_c_floats=32112 b_guard_words=16 scratch_words=0 checksum_accum=-8128 expected_checksum_accum=-8128 max_abs_diff=0.0 runtime_allocations=1 runtime_launches=8 runtime_capacity=505840 code_uploads=1 ...
```

This is still a naive matmul.  It does not use shared-memory tiling, barriers,
or optimized GEMM blocking, and it is not a performance benchmark.
