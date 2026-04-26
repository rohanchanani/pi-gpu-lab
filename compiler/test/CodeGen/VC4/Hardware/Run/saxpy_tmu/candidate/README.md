# Candidate side disabled

The candidate side for `saxpy_tmu` is intentionally disabled until VC4 codegen
can emit a runnable qasm/C/H bundle from `../input.mlir`.

When candidate-side execution is enabled, it should generate a bundle from the
MLIR input, build it with the same semantic harness contract, run it on real
hardware, and compare its `VC4_TEST_RESULT` against `../expected.json`.
