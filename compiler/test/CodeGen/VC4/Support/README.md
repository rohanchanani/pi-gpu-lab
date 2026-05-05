# VC4 codegen hardware support helpers

These helpers are deterministic support code for Milestone 1 automation.

- `run_candidate_codegen_test.sh` generates, assembles, builds, and runs candidate bundles from `input.mlir` without modifying checked-in reference bundles.
- `check_vc4_test_result.py` checks the last `VC4_TEST_RESULT` line in one or more logs against a hardware-run `expected.json` file.

Reference bundles under `compiler/test/CodeGen/VC4/Hardware/Run/*/reference/` are read-only ground truth during codegen work. Candidate artifacts are generated under `.vc4_auto/codegen_m1/`.
