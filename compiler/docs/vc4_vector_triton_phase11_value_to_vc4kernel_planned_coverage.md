PHASE11_VALUE_TO_VC4KERNEL_PLANNED_COVERAGE=YES
PHASE11_VALUE_STRIDED_RANKED_MEMORY_CONTRACT=LOCKED
READY_FOR_PHASE11_4_VALUE_RANKED_STRIDED_STATIC=YES
READY_FOR_TRITON=NO

# Phase 11 Value-To-VC4Kernel Planned Coverage

Phase 11.3 is a value-surface contract phase. It does not implement executable
ranked/strided memory lowering. The following planned static conversion
coverage is locked for the implementation phase:

- rank-1 flattened stride address lowers;
- rank-2 identity row-slice transfer lowers;
- rank-2 strided outer row-slice transfer lowers;
- `memref.dim` maps to explicit shape args;
- nonunit inner stride rejects;
- lane-varying stride/gather rejects;
- column-slice rejects;
- hidden descriptor extraction rejects.

The required lower path remains:

```text
value -> vc4kernel -> ssavc4 -> scheduled vc4
```

No direct value-to-VC4 path, hidden memref descriptor ABI, gather/scatter,
block-pointer, reduction, dot, or Triton-readiness claim is introduced here.
