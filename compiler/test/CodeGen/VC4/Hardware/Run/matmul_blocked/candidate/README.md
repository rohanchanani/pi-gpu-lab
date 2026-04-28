# matmul_blocked candidate side

Candidate/codegen execution is intentionally disabled until the VC4 backend can
emit the qasm plus launcher C/H bundle from `input.mlir`.

The trusted source of truth for this hardware-run test is the hand-authored
`reference/` bundle.
