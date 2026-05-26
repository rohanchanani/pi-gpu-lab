# VC4 hardware golden: saxpy_tmu

This test is the TMU-backed counterpart to `saxpy_basic`.

The generated candidate kernel reads one 16-lane `f32` vector of `x` and one
16-lane `f32` vector of `y` per active QPU using TMU0 direct memory lookups,
computes:

```text
y[i] = alpha * x[i] + y[i]
```

and stores the result back to `y` using the same VPM/VDW output mechanism as
`tmu_read_nop_write` and `saxpy_basic`.

## What this test verifies

- The generated `input.mlir` bundle can run a QPU user program on real
  hardware.
- The public launcher API remains semantic: runtime handle, `x`, `y`, `alpha`,
  and `n`; no public `qpu_id`, `num_qpus`, raw uniforms, bus addresses, VPM
  rows, or scheduler internals.
- The launcher packs the physical uniform stream in this order per QPU:

  ```text
  [0] x base address
  [1] y base address/result address
  [2] alpha as one f32/u32 word
  [3] n element count
  [4] qpu_id
  [5] num_qpus
  ```

- The input side uses TMU0 direct memory lookups instead of the VDR DMA-load
  path used by `saxpy_basic`.
- The arithmetic remains the canonical SAXPY sequence: `fmul` followed by
  `fadd`.
- The writeback path remains VPM write plus VDW DMA store.

## What this test deliberately does not prove yet

- It does not prove looped SAXPY over arbitrary-length buffers.
- It does not prove tail-safe non-multiple-of-16 handling.
- It does not prove multiple outstanding TMU requests or TMU FIFO depth.
- It does not prove TMU1 use, TMU swap/no-swap policy, or TMU load balancing.
- It does not prove overlapped/multiple-outstanding TMU requests; that is
  covered by `saxpy_tmu_overlap`.

## Work distribution

The test runs one vector per active QPU:

```text
base_element = qpu_id * 16
n = active_qpus * 16
```

The launcher rejects any `n` other than `active_qpus * lane_width`. This is a
local test policy for this small golden and must not be generalized to all future
VC4 codegen.

## Expected result

The candidate harness initializes deterministic dyadic `f32` inputs for four
alpha cases, runs the generated GPU bundle, computes the same result on the ARM
CPU, and verifies every live lane with exact `f32` bit comparisons.  It also
checks a guard tail after the `y` buffer and generated-runtime counters.

The final successful log line must look like:

```text
VC4_TEST_RESULT name=saxpy_tmu status=PASS cases=4 checked_elements=768 total_mismatches=0 guard_mismatches=0 active_qpus=12 n=192 checksum_accum=7648000 max_abs_diff=0.0 ...
```

Only stable semantic fields are checked by `expected.json`.
