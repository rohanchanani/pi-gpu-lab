# saxpy_tmu_overlap

`saxpy_tmu_overlap` is the latency-hiding TMU SAXPY hardware golden.

It computes the same semantic result as `saxpy_tmu`:

```text
for i in 0..n:
  y[i] = alpha * x[i] + y[i]
```

The difference is the final-stage schedule.  `saxpy_tmu` serializes each direct
TMU request/read pair.  This test issues the independent `x` and `y` direct
TMU0 requests before synchronizing on either result, then receives them in FIFO
order.  That makes the overlap visible in both the qasm reference and the
`input.mlir` scheduled sink.

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
input.mlir             future compiler input
expected.json          stable semantic oracle
reference/             trusted qasm/C/H reference bundle
candidate/README.md    placeholder until codegen exists
```

Run the reference side with:

```bash
compiler/test/CodeGen/VC4/Support/run_hardware_test.sh \
  compiler/test/CodeGen/VC4/Hardware/Run/saxpy_tmu_overlap \
  reference
```
