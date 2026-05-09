# Minimal VC4/QPU user-program reference kernel.
#
# This is the hardware ground-truth qasm for input.mlir in the parent test
# directory.  It verifies only clean scheduler launch/completion and the
# required program-end epilogue shape.
#
# No uniforms are consumed by this kernel.  The launcher still provides the
# two-word physical builtin suffix [qpu_id, num_qpus] for ABI consistency.

thrend
nop
nop
