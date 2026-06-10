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

## Accelerated Demo 1 Importer Static Pipeline

PHASE13_DEMO_IMPORTER_STATIC=LOCKED
TTIR_GEMV_DEMO_IMPORTER_STATIC=PASS
TTIR_GEMV_DEMO_STATIC_PIPELINE=PASS
TTIR_GEMV_DEMO_PRODUCT_REDUCTION_LOWERING=YES
TTIR_GEMV_DEMO_INDEPENDENT_POINTER_PLANNING=YES
TTIR_GEMV_DEMO_F32_FINITE_REDUCTION_POLICY=YES
TTIR_GEMV_DEMO_TT_DOT_STAGED=YES
FRONTEND_ROBUSTNESS_AUDIT=PASS
READY_FOR_PHASE13_DEMO_2_HARDWARE_LOCK=YES
FULL_PHASE13_LOCK=NO
READY_FOR_TRITON=NO

The exact source-controlled GEMV demo TTIR snapshot now lowers statically
through:

```text
TTIR -> value -> value verifier -> scf-to-cf -> vc4kernel -> ssavc4 -> scheduled vc4
```

The importer repair is limited to structural pointer planning for `tt.splat`
of scalar pointer expressions, allowing the row-strided A pointer
`A + row * LDA` to be splatted before adding the contiguous lane range. This
also preserves independent pointer plans for A, X, and Y. The product is
lowered as value-layer `arith.mulf`, and the existing Phase 12 reduction path
lowers `tl.sum(a * x)` to `vector.reduction <add>` with finite f32 reduction
policy attrs.

No `tl.dot`, `tt.dot`, `vector.contract`, multi-block K accumulation, atomics,
f16/casts, SFU/math, or full Phase 13 support is claimed.
