# Candidate side: `read_nop_write`

This side is intentionally disabled until VC4 codegen exists.

Future workflow:

1. Run the VC4 code generator on `../input.mlir`.
2. Emit the candidate bundle:
   - `read_nop_write.qasm`
   - `read_nop_write_launch.c`
   - `read_nop_write_launch.h`
3. Build and run the candidate bundle with the same semantic harness shape as
   the reference side.
4. Compare the final `VC4_TEST_RESULT` line against `../expected.json`.

Do not add candidate generated files to git until the codegen milestone defines
the candidate-side artifact layout.
