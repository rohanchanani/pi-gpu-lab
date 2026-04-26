# saxpy_16 hardware golden

`saxpy_16` is the first SAXPY golden that is not limited to exactly one
16-lane vector per active QPU.

The reference kernel computes:

```text
y[i] = alpha * x[i] + y[i]
```

for `N = active_qpus * 16 * 4` elements in the checked hardware harness.  The
qasm loops across per-QPU chunks rather than assuming `N == 192`.  The
generated-style launcher intentionally does **not** guard on `N`; this
exact-multiple precondition belongs to the handwritten/upstream-style harness
and to the kernel semantics, not to the generated launcher.

## What this proves

This test proves one incremental codegen concept beyond `saxpy_tmu_overlap`:

- a QPU can process more than one 16-lane chunk by using
  `base = qpu_id * 16` and `stride = num_qpus * 16`;
- the checked harness uses exact-multiple `N`, so there is no tail masking yet;
- scalar and buffer ABI packing remains the same as earlier SAXPY tests;
- loads use direct TMU0 memory lookups, following the current project default
  for ordinary global/shared-memory-like loads;
- stores still use the VPM/VDW output path.

It does **not** prove non-multiple-of-16 tail behavior, reuse-oriented VPM DMA
loads, thread switching, semaphore coordination, or generated-code candidate
execution.

## Files

```text
input.mlir
expected.json
candidate/README.md
reference/3-test-saxpy-16.c
reference/saxpy_16.qasm
reference/saxpy_16_launch.c
reference/saxpy_16_launch.h
```

Codex should mechanically add `reference/Makefile`, `reference/run.sh`,
`reference/mailbox.c`, `reference/mailbox.h`, and `share/`.

## Stable result contract

The successful reference run prints a final line like:

```text
VC4_TEST_RESULT name=saxpy_16 status=PASS mismatches=0 active_qpus=12 n=768 chunks_per_qpu=4 checksum=11592704 max_abs_diff=0.0 ...
```

`elapsed_usec` is informational only and is not checked by `expected.json`.
