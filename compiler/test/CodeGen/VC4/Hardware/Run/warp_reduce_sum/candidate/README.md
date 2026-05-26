
warp_reduce_sum candidate side

This candidate harness builds and launches the generated VC4 kernel from
`input.mlir` on hardware.

The host oracle computes the canonical per-vector f32 sum over active lanes and
requires exact output equality for every active element. It also checks guard
sentinels, matching checksums, real runtime launch/allocation/code-upload
counters, and recorded launch failures.
