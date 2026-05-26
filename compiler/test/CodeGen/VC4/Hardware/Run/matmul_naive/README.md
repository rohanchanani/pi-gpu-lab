# matmul_naive

`matmul_naive` is a generated VC4 hardware-run test for a small canonical
row-major `f32` matrix multiply:

```text
C[M,N] = A[M,K] * B[K,N]
```

The generated kernel supports bounded runtime shape arguments:

```text
1 <= M <= 12
1 <= N <= 16
0 <= K <= 4
```

## Device Mapping

The fixture uses the current VC4-as-CUDA independent-vector launch model:

```text
one QPU user-program request = one output row
ELEMENT_NUMBER               = column lane, 0..15
logical_request uniform      = row id
```

Each request computes one row.  The kernel uses TMU direct memory reads for
`A[row,kk]` and `B[kk,col]`, unrolls the four possible `K` terms, predicates
each multiply/add on the runtime `k`, and stores the row through VPM/VDW with a
runtime `n` store depth.

The harness launches the maximum 12 row requests for every case.  Rows with
`row >= m` route their VDW stores to a scratch region after the live C capacity,
so the host can verify that the logical output tail remains untouched.

## Verification

The candidate harness sweeps eight runtime `(m,n,k)` cases, including partial
rows, partial columns, `K=0`, and single-element output.  For each case it:

- fills compact row-major `A` and `B` buffers deterministically,
- computes an unweakened ARM CPU oracle with the same canonical loop nest,
- compares every live output by exact `f32` bits,
- checks the compact output tail and final guard region are untouched,
- requires generated-runtime allocation, launch, code-upload, and failure
  counters to match the sweep.

Scratch rows are intentionally not checked for equality; they are only a legal
sink for inactive launched rows.

## Current Hardware Result

```text
VC4_TEST_RESULT name=matmul_naive status=PASS cases=8 checked_elements=504 total_mismatches=0 guard_mismatches=0 launch_failures=0 recorded_launch_failures=0 active_qpus=12 lanes=16 max_m=12 max_n=16 max_k=4 scratch_words=192 checksum_accum=6464 expected_checksum_accum=6464 max_abs_diff=0.0 runtime_allocations=1 runtime_launches=8 runtime_capacity=65536 code_uploads=1 ...
```

This is still a naive matmul.  It does not use shared-memory tiling, barriers,
or optimized GEMM blocking, and it is not a performance benchmark.
