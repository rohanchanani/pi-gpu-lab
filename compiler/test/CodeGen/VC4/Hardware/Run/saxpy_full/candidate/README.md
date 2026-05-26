# candidate side

The candidate harness builds the `input.mlir` VC4 scheduled kernel into a
launchable generated bundle, sweeps the same 19 tail cases as the reference, and
checks every live output against an exact host oracle with guard-tail
verification.
