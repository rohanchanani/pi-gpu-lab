# minimal_thrend

This is the first runnable VC4 hardware contract test.

It pairs a final-stage `vc4` MLIR input with a trusted reference bundle.  Future
codegen should generate a candidate qasm/launcher bundle from `input.mlir`; the
candidate will pass only if it produces the same semantic hardware result as the
reference bundle.

## What this test verifies

`minimal_thrend` verifies only the base QPU user-program launch/completion path:

- QPU enable through the bare-metal mailbox/runtime support.
- QPU user-program queue submission.
- Per-QPU uniform stream allocation and pointer setup.
- Correct program termination through `thrend` plus delay slots.
- Scheduler completion for all active QPUs.
- Machine-readable `VC4_TEST_RESULT` reporting.

It intentionally does **not** verify memory output, VPM/VDW, TMU, SFU,
semaphores, mutexes, thread switching, or structured lowering.

## Contract files

```text
input.mlir                 final-stage VC4 program for future codegen
expected.json              semantic result oracle
reference/minimal_thrend.qasm
reference/minimal_thrend_launch.c
reference/minimal_thrend_launch.h
reference/3-test-minimal-thrend.c
```

The mechanically copied/adapted files are:

```text
reference/Makefile
reference/run.sh
reference/mailbox.c
reference/mailbox.h
share/                      copied from old_compiler/reference/share for vc4asm templates/includes
```

## Public and physical ABI

The public launcher surface is:

```c
int minimal_thrend_launch(struct vc4_runtime *rt);
```

The physical per-QPU uniform stream is still two words:

```text
[0] qpu_id
[1] num_qpus
```

The minimal kernel does not read these uniforms.  They are present to keep the
first hardware test aligned with the project-wide public/physical launch ABI
contract.

## Expected result

The harness prints a final line like:

```text
VC4_TEST_RESULT name=minimal_thrend status=PASS completed_qpus=12 active_qpus=12 elapsed_usec=<time>
```

`expected.json` requires only stable semantic fields.  Timing is informational.
