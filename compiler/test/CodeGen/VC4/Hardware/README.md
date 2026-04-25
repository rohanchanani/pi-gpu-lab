# VC4 Hardware Golden Tests

Hardware tests in this directory are first-class ground truth for VC4 codegen.
They are intentionally small, explicit, and self-contained.  Each test should
be understandable and runnable without relying on future compiler passes.

## Directory shape

A runnable hardware test should look like this:

```text
compiler/test/CodeGen/VC4/Hardware/Run/<test-name>/
  README.md
  Makefile
  run.sh
  3-test-<test-name>.c
  mailbox.c
  mailbox.h
  <kernel>.qasm
  <kernel>_launch.c
  <kernel>_launch.h
  expected.json
```

Assembler-only hardware qualification tests may use a smaller shape, but they
must still have a clear `run.sh`, an `expected.json`, and a machine-readable
`VC4_TEST_RESULT` line.

## Required hardware test behavior

`run.sh` should perform the test-local work only: assemble qasm, build/link the
bare-metal program, and run it using the configured Raspberry Pi install path.
It should not power-cycle the Pi itself.  The outer support runner handles
power cycling so every hardware test gets the same setup:

```bash
compiler/test/CodeGen/VC4/Support/run_hardware_test.sh <test-dir>
```

The support runner power-cycles with `uhubctl -l 0-1 -a cycle` by default, then
runs `bash run.sh` inside the test directory, captures `run.log`, and compares
the final `VC4_TEST_RESULT` line against `expected.json`.

## Result line

The harness must print one final successful result line:

```text
VC4_TEST_RESULT name=<test-name> status=PASS mismatches=0 max_abs_diff=0.0
```

The checker uses the last `VC4_TEST_RESULT` line in `run.log`.  Avoid putting
unstable values such as execution time or speedup in the required oracle unless
they are informational only.

## Expected JSON shape

Example:

```json
{
  "name": "saxpy_reference",
  "status": "PASS",
  "required": {
    "mismatches": 0
  },
  "float_max": {
    "max_abs_diff": 0.0001
  }
}
```
