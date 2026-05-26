# Candidate side

The candidate side generates a runnable qasm/C/H bundle from `../input.mlir`
and runs it on real VC4 hardware.

The generated kernel computes dynamic naive `f32` matrix
multiplication:

```text
C[M,N] = A[M,K] * B[K,N]
M, N, K are runtime u32 kernel arguments
```

The candidate harness sizes ARM host buffers with `kmalloc` and VC4 device
buffers with the generated-runtime `vc4_m2_malloc` path from the sweep cases,
not by semantic `M/N/K` maxima.  Larger cases only require enough host and
device memory; the MLIR kernel does not need shape-specific edits.

The harness sweeps eight deterministic runtime `(m,n,k)` cases.  It checks all
127421 live output values with exact `f32` bit comparisons against an ARM CPU
oracle, verifies the compact output tail and final guard region, and requires
the generated-runtime launch/code-upload counters to match the sweep.  Inactive
launched rows exit before any device memory access.
