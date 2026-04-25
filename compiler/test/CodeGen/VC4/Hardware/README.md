# VC4 Hardware Golden Tests

Hardware tests in this directory are first-class ground truth for VC4 codegen.
They are intentionally small, explicit, and self-contained.

The current development phase builds the **reference side** of each test. The
candidate/generated side is enabled only after codegen exists.

## Normal hardware-run directory shape

A normal runnable hardware test should look like this:

```text
compiler/test/CodeGen/VC4/Hardware/Run/<test-name>/
  README.md
  input.mlir
  expected.json

  reference/
    Makefile
    run.sh
    3-test-<test-name>.c
    mailbox.c
    mailbox.h
    <kernel>.qasm
    <kernel>_launch.c
    <kernel>_launch.h

  candidate/
    # Created later by the code generator.
    # Not required during ground-truth authoring.
```

`input.mlir` is the final-stage `vc4` dialect program that the future code
generator must consume. The hand-authored `reference/` bundle is the semantic
ground truth for that input program.

The future generated `candidate/` bundle may choose a different instruction
schedule or C implementation. It passes if it produces the same semantic result
under the same `expected.json` oracle.

## `reference/run.sh` contract

`reference/run.sh` should perform only test-local work:

1. assemble the reference qasm,
2. build/link the bare-metal test binary,
3. install/run it using the configured Raspberry Pi flow,
4. stream serial output to stdout.

It should **not** power-cycle the Pi. The outer runner handles power cycling so
all tests get the same setup:

```bash
compiler/test/CodeGen/VC4/Support/run_hardware_test.sh <test-root> reference
```

The support runner:

1. validates `<test-root>/input.mlir`,
2. validates `<test-root>/expected.json`,
3. validates `<test-root>/reference/run.sh`,
4. power-cycles the Pi unless `VC4_SKIP_POWER_CYCLE=1`,
5. runs `bash run.sh` inside `reference/`,
6. writes `reference/run.log`,
7. compares the final `VC4_TEST_RESULT` line against `expected.json`.

## Result line

The harness must print one final successful result line:

```text
VC4_TEST_RESULT name=<test-name> status=PASS mismatches=0 max_abs_diff=0.0
```

The checker uses the **last** `VC4_TEST_RESULT` line in the log. Avoid putting
unstable values such as execution time or speedup in required oracle fields.

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

Top-level `name` and `status` are exact required fields. Fields in `required`
are exact typed comparisons. Fields in `float_max` require
`abs(actual) <= limit`.

## Environment knobs

The hardware runner uses:

```bash
VC4_PI_POWER_CYCLE_CMD='uhubctl -l 0-1 -a cycle'
VC4_PI_POWER_CYCLE_SLEEP_SEC=4
```

Set:

```bash
VC4_SKIP_POWER_CYCLE=1
```

only for local runner self-tests or explicit manual debugging.

## First hardware tests

Build tests in this order:

1. `minimal_thrend`
   - verifies QPU launch/completion and thread-end epilogue with minimal moving
     parts.

2. `vpm_constant_store`
   - verifies deterministic memory output through VPM/VDW.

3. `read_nop_write`
   - verifies reading an input buffer and writing an output buffer with minimal
     arithmetic.

4. `saxpy_reference`
   - imports the known-good SAXPY reference as the first rich end-to-end
     compute example.
