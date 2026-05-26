# Candidate side

The candidate side generates a runnable qasm/C/H bundle from `../input.mlir`
and runs it on real VC4 hardware.

The generated kernel computes bounded dynamic naive `f32` matrix
multiplication:

```text
C[M,N] = A[M,K] * B[K,N]
1 <= M <= 12, 1 <= N <= 16, 0 <= K <= 4
```

The harness sweeps eight deterministic runtime `(m,n,k)` cases.  It checks all
504 live output values with exact `f32` bit comparisons against an ARM CPU
oracle, verifies the compact output tail and final guard region, and requires
the generated-runtime launch/code-upload counters to match the sweep.  Inactive
launched rows write to the candidate-owned scratch region, not the logical
output.
