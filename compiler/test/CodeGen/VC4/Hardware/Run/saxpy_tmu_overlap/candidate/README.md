# Candidate side

The candidate side generates a runnable qasm/C/H bundle from `../input.mlir`
and runs it on real hardware.

The harness sweeps four deterministic `alpha` cases, checks all 768 live
outputs with exact `f32` bit comparisons against the ARM CPU oracle, verifies a
guard tail after the `y` buffer, and requires generated-runtime launch and code
upload counters to match the sweep.
