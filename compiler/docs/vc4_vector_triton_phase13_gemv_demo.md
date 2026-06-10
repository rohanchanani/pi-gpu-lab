PHASE13_DEMO_SOURCE_SNAPSHOT=LOCKED
FEATURE=TTIR_GEMV_ROWWISE_DOT_DEMO
REAL_TRITON_GEMV_DEMO_SOURCE=YES
REAL_TTIR_GEMV_DEMO_SNAPSHOT=YES
GEMV_DEMO_KERNEL_SHAPE=tl_load_row_strided_A_plus_tl_load_contiguous_X_plus_tl_sum_product
GEMV_DEMO_K_RUNTIME_SCALAR=YES
GEMV_DEMO_LDA_RUNTIME_SCALAR=YES
FULL_PHASE13_LOCK=NO
READY_FOR_PHASE13_DEMO_1_IMPORTER_STATIC_PIPELINE=YES
READY_FOR_TRITON=NO

# Phase 13 Accelerated GEMV Demo Source Snapshot

This document locks only the accelerated Phase 13 demo input snapshot. It does
not claim full Phase 13 support.

The demo source is:
`examples/triton/phase13_gemv_demo/sources/gemv_row_dot_f32_b16.py`

The real TTIR snapshot is:
`examples/triton/phase13_gemv_demo/generated/gemv_row_dot_f32_b16.ttir.mlir`

The source shape is a single-row f32 GEMV-v0 dot product:

- one program computes one row with `tl.program_id(axis=0)`;
- `A` is loaded as `A + row * LDA + offs`;
- `X` is loaded as `X + offs`;
- `K` is a runtime scalar `i32` that forms the tail mask `offs < K`;
- `LDA` is a runtime scalar `i32` row stride;
- the product uses elementwise f32 multiply;
- the scalar result is produced by `tl.sum(a * x, axis=0)`;
- the result stores to `Y + row`.

The demo intentionally excludes `tl.dot`, `tt.dot`, block pointers, atomics,
f16/subword forms, casts, SFU/math, multi-block K accumulation, and i32
variants. Later demo phases may add importer support, static pipeline proof,
and hardware proof for this exact source-controlled input.
