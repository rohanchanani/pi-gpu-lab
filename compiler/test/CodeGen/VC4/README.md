# VC4 CodeGen Test Corpus

This directory is the home for VC4 code-generation tests.

The corpus is built incrementally. A test is added here only after its
final-stage `vc4` MLIR input, reference bundle, semantic oracle, and run
procedure exist and have been checked. Planned tests live in
`compiler/docs/codegen/test-backlog.md` until they are materialized.

## Locked test model

A normal hardware-backed codegen test has one **test root**:

```text
compiler/test/CodeGen/VC4/Hardware/Run/<test-name>/
  README.md
  input.mlir
  expected.json
  reference/
    run.sh
    ... reference qasm/C/H/harness files ...
  candidate/
    ... generated later by codegen; absent during ground-truth authoring ...
```

The current phase is the **ground-truth/reference phase**. During this phase,
the `reference/` bundle is hand-authored, imported, or otherwise trusted. It is
run on real hardware and compared against `expected.json`.

The future phase is the **candidate/generated-code phase**. In that phase, the
compiler will consume `input.mlir`, emit a candidate `qasm + launcher.c +
launcher.h` bundle into `candidate/`, run it through the same hardware path, and
compare the candidate result against the same semantic oracle.

The reference bundle does **not** need to be character-for-character identical
to the future generated bundle. It must compute the same semantic result for
the same `input.mlir`.

## Ground rules

1. **Hardware execution is the gold standard** for runnable kernel examples.
   A hardware test must run through the same bare-metal Raspberry Pi flow used
   by the reference programs: assemble qasm, build/link the harness, boot the
   Pi, run the program, and inspect the serial output.

2. **The catalog is factual, not aspirational.**
   `catalog.json` contains only implemented tests. Do not add future tests to
   the catalog as placeholders.

3. **Every normal hardware test has an MLIR input.**
   `input.mlir` is final-stage `vc4` dialect IR immediately before codegen.
   For generated-code testing later, this file is the compiler input.

4. **The reference side is self-contained.**
   `reference/` contains its own `run.sh`, qasm, launcher C/H, runtime support
   files if needed, harness C, and build files.

5. **Every hardware run prints a machine-readable result line.**
   The final successful harness output must include a line of the form:

   ```text
   VC4_TEST_RESULT name=<test-name> status=PASS key=value ...
   ```

6. **`expected.json` is semantic.**
   It checks stable result fields such as `status=PASS`, `mismatches=0`, or
   `max_abs_diff <= epsilon`. It should not require unstable values such as
   timing, speedup, serial-port noise, or incidental assembler diagnostics.

7. **Do not freeze prototype-only behavior.**
   Existing references such as SAXPY are proven hardware harnesses, not
   universal future codegen policy. Future generated launchers should not be
   forced to copy shader code on every call or reject all non-multiple-of-16
   problem sizes unless the kernel ABI explicitly requires that policy.

## Current stage

Stage 0 installs support infrastructure only. The implemented catalog starts
empty:

```json
{
  "implemented_tests": []
}
```

The first materialized tests should be small and hardware-facing:

1. `minimal_thrend`
2. `vpm_constant_store`
3. `read_nop_write`
4. `saxpy_reference`

## Useful commands

Self-test the hardware result checker:

```bash
python3 compiler/test/CodeGen/VC4/Support/check_vc4_test_result.py --self-test
```

Self-test the hardware runner without using hardware:

```bash
compiler/test/CodeGen/VC4/Support/run_hardware_test.sh --self-test
```

Run a materialized reference-side hardware test once it exists:

```bash
compiler/test/CodeGen/VC4/Support/run_hardware_test.sh \
  compiler/test/CodeGen/VC4/Hardware/Run/<test-name> \
  reference
```

Run a future candidate-side hardware test once codegen exists:

```bash
compiler/test/CodeGen/VC4/Support/run_hardware_test.sh \
  compiler/test/CodeGen/VC4/Hardware/Run/<test-name> \
  candidate
```
