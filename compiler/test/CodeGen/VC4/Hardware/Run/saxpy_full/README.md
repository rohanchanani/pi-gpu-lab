# saxpy_full

`SAXPY_FULL` is the tail-safe successor to `saxpy_16`.

It computes:

```text
y[i] = alpha * x[i] + y[i], for i in [0, n)
```

The important difference from `saxpy_16` is that `n` is no longer restricted to an exact multiple of `active_qpus * 16` or even to a single 16-lane QPU vector.  The harness exercises `n = 0`, sub-vector tails, vector-boundary values, active-QPU-round boundaries, and values larger than the old four-chunks-per-QPU shape.

## Hardware path

The trusted reference qasm uses the same basic path as `saxpy_16`:

- direct TMU0 memory loads for `x` and `y`,
- QPU floating-point multiply/add,
- VPM staging,
- VDW memory store,
- thread-end plus two delay slots.

The per-QPU work distribution remains:

```text
base_element   = qpu_id * 16
stride_element = num_qpus * 16
```

Each QPU processes vector chunks while `base_element < n`.

## Tail policy

`tail_policy = tail_safe`.

For a final partial vector, the qasm computes:

```text
store_count = min(16, n - base_element)
```

and programs the VDW store `DEPTH` dynamically so that only `store_count` 32-bit words are written for that final chunk.

The generated-style launcher pads only its private GPU scratch buffers up to a 16-element boundary.  This keeps inactive tail-lane TMU reads within allocated GPU memory.  The public semantic `n` remains unchanged, and only the first `n` output elements are copied back to the caller.

## What this test proves

This test proves that the SAXPY launcher/kernel pair computes the correct result for arbitrary tested `n` values and that the public host-side `y[n..]` guard region is not modified.

It also proves that the qasm work distribution covers values spanning multiple active-QPU rounds.

## What it does not prove yet

This test does not prove a zero-copy user-buffer tail policy.  The current reference launcher follows the existing SAXPY reference style and uses private GPU scratch buffers.

This test also does not measure performance.

## Expected result

The reference hardware run prints one `SAXPY_FULL_CASE` line per tested `n` and a final line:

```text
VC4_TEST_RESULT name=saxpy_full status=PASS cases=19 total_mismatches=0 sentinel_mismatches=0 launch_failures=0 active_qpus=12 lanes=16 ...
```

Candidate/codegen side remains disabled until generated bundles exist.
