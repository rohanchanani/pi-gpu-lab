# VC4 CodeGen Test Corpus

This directory is the home for VC4 code-generation tests.  The corpus is
built incrementally: a test is added here only after its inputs, ground-truth
artifacts, and run procedure exist and have been checked.  Planned tests live
in `compiler/docs/codegen/test-backlog.md` until they are materialized.

## Ground rules

1. **Hardware execution is the gold standard** for runnable kernel examples.
   A hardware test must run through the same bare-metal Raspberry Pi flow used
   by the reference SAXPY program: assemble qasm, build/link the harness, boot
   the Pi, run the program, and inspect the serial output.

2. **The catalog is factual, not aspirational.**
   `catalog.json` contains only implemented tests.  Do not add future tests to
   the catalog as placeholders.

3. **A hardware test is self-contained.**
   Each hardware test directory must contain its qasm, launcher C/H, runtime
   support files if needed, harness C, Makefile, `run.sh`, `expected.json`, and
   a README explaining what the test proves.

4. **Every hardware test prints a machine-readable result line.**
   The final successful harness output must include a line of the form:

   ```text
   VC4_TEST_RESULT name=<test-name> status=PASS key=value ...
   ```

5. **Do not freeze prototype-only behavior.**
   The existing SAXPY reference is a proven hardware harness shape, not the
   universal future codegen contract.  In particular, future generated launchers
   should not be forced to copy shader code on every call or reject all
   non-multiple-of-16 problem sizes unless the kernel ABI explicitly requires
   that policy.

## Current stage

Stage 0 only installs support infrastructure.  The implemented catalog starts
empty.  The first real implemented test should be a normalized copy of the
known-good SAXPY reference hardware run.

## Useful commands

Self-test the hardware result checker:

```bash
python3 compiler/test/CodeGen/VC4/Support/check_vc4_test_result.py --self-test
```

Run a materialized hardware test once it exists:

```bash
compiler/test/CodeGen/VC4/Support/run_hardware_test.sh \
  compiler/test/CodeGen/VC4/Hardware/Run/<test-name>
```
