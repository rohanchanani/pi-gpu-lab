# VC4 CodeGen Test Backlog

This document tracks planned tests.  It is intentionally separate from
`compiler/test/CodeGen/VC4/catalog.json`: the catalog lists only tests that are
implemented, have ground-truth artifacts, and have been run.

## Stage 0: harness support

- Add `check_vc4_test_result.py`.
- Add `run_hardware_test.sh`.
- Keep `catalog.json` empty until the first real test is implemented.

## Stage 1: known-good hardware reference

- `Hardware/Run/saxpy_reference`
  - Normalize the existing working SAXPY reference.
  - Add a final `VC4_TEST_RESULT` line.
  - Add `expected.json` requiring `status=PASS`, `mismatches=0`, and
    `max_abs_diff <= 0.0001`.
  - This proves the test harness and Pi execution path before any new qasm is
    authored.

## Stage 2: minimal launch/completion

- `Hardware/Run/minimal_thrend`
  - Tiny hand-authored qasm kernel that immediately terminates correctly.
  - Verifies QPU enable, user-program queue submission, scheduler completion,
    and thread-end epilogue handling with minimal moving parts.

## Stage 3: minimal output-producing kernel

- `Hardware/Run/vpm_constant_store`
  - Hand-authored qasm writes a known pattern to memory using VPM/VDW.
  - Verifies memory output without TMU or structured lowering complexity.

## Stage 4: TMU assembler qualification

- `Hardware/AssemblerQual/tmu_direct_single`
- `Hardware/AssemblerQual/tmu_direct_x4`
  - Verify that the chosen vc4asm version accepts the TMU direct-address subset
    that codegen will eventually emit.
  - Compare semantic PASS/FAIL summaries, not raw vc4asm diagnostic text.

## Stage 5: runnable TMU direct hardware test

- `Hardware/Run/tmu_direct_single`
  - Direct TMU memory lookup, result written back through VPM/VDW.
  - Start with one request before adding pipelined outstanding-request tests.

## Later compiler-facing tests

Add these only after the hardware harness is healthy:

1. ABI metadata and launch-plan tests.
2. Scheduled sink qasm emission tests.
3. Launcher header/C surface tests.
4. Mock-runtime trace tests.
5. Structured-to-scheduled lowering tests.

Do not add broad future tests as `XFAIL: *` placeholders.  If a tool or pass
does not exist yet, keep the test in this backlog until the implementation
milestone reaches it.
