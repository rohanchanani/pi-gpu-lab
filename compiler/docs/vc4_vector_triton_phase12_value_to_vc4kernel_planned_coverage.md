# VC4 Phase 12 Value-to-VC4Kernel Planned Reduction Coverage

PHASE12_VALUE_REDUCTION_CONTRACT=LOCKED
VALUE_VECTOR_REDUCTION_ADD_I32_SURFACE=ACCEPTED
VALUE_VECTOR_REDUCTION_ADD_F32_FINITE_SURFACE=ACCEPTED
VALUE_SCALAR_MEMREF_STORE_FOR_REDUCTION_SURFACE=ACCEPTED
F32_REDUCTION_FINITE_TREE_POLICY=YES
NON_ADD_REDUCTIONS_STAGED=YES
DOT_GEMV_STAGED_FOR_PHASE13=YES
VALUE_TO_VC4KERNEL_PLANNED_COVERAGE=YES
READY_FOR_PHASE12_4_VALUE_REDUCTION_STATIC=YES
READY_FOR_TRITON=NO

Phase 12.3 locks the source-level value reduction contract but does not
implement executable value-to-VC4Kernel reduction lowering.

Phase 12.4 static conversion coverage must include:

- `vector.reduction <add>` over `vector<16xi32>` lowers to accepted
  VC4Kernel i32 reduction forms.
- `vector.reduction <add>` over `vector<16xf32>` lowers only when finite-input
  and finite-tree policy are explicit.
- f32 reduction without finite-tree policy rejects before VC4Kernel IR is
  constructed.
- scalar `memref.store` of `i32`/`f32` reduction results to rank-1
  `#vc4value.global` output memrefs lowers.
- tail reductions lower by relying on Phase 10 inactive-zero transfer reads,
  not by introducing a masked reduction op.
- row-strided sum value patterns lower through the Phase 11 central address
  planner.
- max, min, product, and custom reductions reject.
- rank greater than 1 reductions reject.
- `vector.multi_reduction` rejects.

No Phase 12.3 test or document claims hardware proof. Dot, GEMV, GEMM,
atomics, scans, f16/subword reductions, exact/default f32 reductions, and
generalized math remain staged.
