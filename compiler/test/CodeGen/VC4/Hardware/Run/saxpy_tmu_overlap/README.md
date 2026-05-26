# saxpy_tmu_overlap

`saxpy_tmu_overlap` is the latency-hiding generated TMU SAXPY hardware test.

It computes the same semantic result as `saxpy_tmu`:

```text
for i in 0..n:
  y[i] = alpha * x[i] + y[i]
```

The difference is the final-stage schedule.  `saxpy_tmu` serializes each direct
TMU request/read pair.  This test issues the independent `x` and `y` direct
TMU0 requests before synchronizing on either result, then receives them in FIFO
order.  That makes the overlap visible in the generated qasm emitted from
`input.mlir`.

## What this verifies

- The input side uses TMU0 direct memory lookup, not VDR DMA.
- Two independent direct-address TMU requests can be outstanding before the
  first `ldtmu0`.
- The requested values are consumed in TMU receive FIFO order: first `x`, then
  `y`.
- The arithmetic is canonical SAXPY: `fmul` for `alpha * x`, then `fadd` with
  `y`.
- The output side remains the established VPM/VDW write path.
- The launcher ABI remains semantic: public args are `x`, `y`, `alpha`, and
  `n`; `qpu_id`, `num_qpus`, uniform packing, and scheduler details stay inside
  the launcher/runtime boundary.

## What this does not verify yet

- It is not a benchmark and does not require a timing threshold.
- It does not cover more than two outstanding direct TMU requests.
- It does not exercise TMU1, TMU no-swap, texture-mode descriptors, or
  non-direct texture sampling.
- It does not implement tail handling; this small golden intentionally uses one
  full 16-lane vector per active QPU.

## Layout

```text
input.mlir             final-stage VC4 compiler input
expected.json          stable semantic oracle
reference/             trusted qasm/C/H reference bundle
candidate/             generated-candidate hardware harness
```

Run the generated candidate side with:

```bash
compiler/test/CodeGen/VC4/Support/run_candidate_codegen_test.sh \
  saxpy_tmu_overlap run
```

The candidate harness initializes deterministic dyadic `f32` inputs for four
alpha cases, computes the same result on the ARM CPU, checks all 768 live
outputs by exact `f32` bits, verifies a guard tail after the `y` buffer, and
requires generated-runtime counters.

The final successful log line must look like:

```text
VC4_TEST_RESULT name=saxpy_tmu_overlap status=PASS cases=4 checked_elements=768 total_mismatches=0 guard_mismatches=0 active_qpus=12 n=192 checksum_accum=7648000 max_abs_diff=0.0 ...
```
