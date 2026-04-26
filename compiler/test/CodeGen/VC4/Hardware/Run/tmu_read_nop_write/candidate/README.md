# Candidate side placeholder

The candidate/generated-code side of `tmu_read_nop_write` is intentionally
disabled for now because VC4 code generation is not implemented yet.

When codegen exists, this side should be populated by running the emitter on:

```text
../input.mlir
```

and producing a generated bundle equivalent to the trusted reference bundle:

```text
tmu_read_nop_write.qasm
tmu_read_nop_write_launch.c
tmu_read_nop_write_launch.h
```

The candidate run must use the same semantic oracle in `../expected.json`.
