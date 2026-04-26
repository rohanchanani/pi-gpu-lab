# saxpy_16 candidate side

Candidate-side generation is intentionally disabled until VC4 codegen exists.

When codegen is implemented, this side should be generated from `../input.mlir`
and should produce a qasm/C/H bundle that computes the same semantic result as
the trusted reference side.  The candidate bundle does not need to be textually
identical to the reference bundle, but it must preserve the public launcher ABI,
uniform stream layout, exact-multiple-of-`active_qpus * 16` policy, TMU load
semantics, and output result.
