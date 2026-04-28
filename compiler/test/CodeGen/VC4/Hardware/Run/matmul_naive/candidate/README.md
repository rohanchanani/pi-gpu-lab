# candidate side disabled

Candidate-side generation for `matmul_naive` is intentionally disabled until VC4 codegen can emit the qasm, launcher C, and launcher H bundle from `input.mlir`.

The trusted reference side is the hardware source of truth for now.
