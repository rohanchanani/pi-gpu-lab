# candidate side disabled

Candidate-side generation is intentionally disabled for `saxpy_full` until VC4 codegen emits launchable qasm/C/H bundles from `input.mlir`.

The trusted source of truth for now is the hand-authored `reference/` bundle and the shared `expected.json` semantic oracle.
