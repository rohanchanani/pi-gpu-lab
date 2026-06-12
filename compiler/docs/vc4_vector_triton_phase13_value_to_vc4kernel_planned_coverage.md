# VC4 Phase 13 Value-to-VC4Kernel GEMV Planned Coverage

PHASE13_VALUE_GEMV_ROWWISE_DOT_CONTRACT=LOCKED
VALUE_TO_VC4KERNEL_PLANNED_COVERAGE=YES
VALUE_GEMV_ROWWISE_DOT_F32_SURFACE=ACCEPTED
VALUE_GEMV_ROWWISE_DOT_I32_SURFACE=ACCEPTED
F32_DOT_FINITE_TREE_POLICY=YES
TL_DOT_TT_DOT_STAGED=YES
VECTOR_CONTRACT_STAGED=YES
MULTIBLOCK_K_ACCUMULATION_STAGED=YES
READY_FOR_PHASE13_4_VALUE_GEMV_STATIC=YES
READY_FOR_TRITON=NO

This document records Phase 13.3 planned static conversion coverage. It is not
an executable lowering implementation and does not claim hardware proof.

Phase 13.4 should add static conversion tests for:

- f32 `vector<16xf32>` elementwise `arith.mulf` feeding finite-tree
  `vector.reduction <add>` and scalar `memref.store`;
- i32 `vector<16xi32>` elementwise `arith.muli` feeding
  `vector.reduction <add>` and scalar `memref.store` when
  `vc4value.i32_mul_policy = "mul24_safe"` is present;
- tail row dots whose inactive lanes are zeroed by Phase 10 transfer reads;
- row-strided GEMV-v0 value patterns using Phase 11 rank-2 row-slice memory;
- partial K-block dots that store one scalar partial per `(row, kblock)`;
- deterministic staging/reject diagnostics for `vector.contract`;
- deterministic staging/reject diagnostics for exact/default f32 dot without
  finite-tree policy;
- deterministic staging/reject diagnostics for multi-block K accumulation into
  final `y[row]`.

The required lowering path remains:

```text
standard value surface
  -> VC4Kernel
  -> SSAVC4
  -> scheduled VC4
```

No direct TTIR-to-VC4Kernel or value-to-scheduled-VC4 path is allowed.
