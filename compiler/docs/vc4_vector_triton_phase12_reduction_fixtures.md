PHASE12_CONTROLLED_TRITON_REDUCTION_FIXTURES=YES
REAL_TRITON_REDUCTION_SOURCES=YES
REAL_TTIR_REDUCTION_SNAPSHOTS=YES
ACCEPTED_FIXTURES_EXCLUDE_UNRELATED_STAGED_FEATURES=YES
F32_REDUCTION_FINITE_TREE_POLICY=YES
NON_ADD_REDUCTION_FIXTURE_STAGED=YES
RANK_GT_1_REDUCTION_FIXTURE_STAGED=YES
DOT_GEMV_STAGED_FOR_PHASE13=YES
READY_FOR_PHASE12_3_VALUE_SURFACE_CONTRACT=YES
READY_FOR_TRITON=NO

# VC4 Vector/Triton Phase 12 Reduction Fixtures

Phase 12.2 adds controlled real Triton sources and real emitted TTIR snapshots
for the reduction feature band. These fixtures define the Phase 12 TTIR
acceptance contract; the earlier Phase 12.1 probes remain inventory only.

Source-controlled fixture directory:

```text
examples/triton/phase12_reductions/
```

Accepted snapshots:

- `ttir_reduce_sum_f32_b16`: f32 vector<16> tail-masked load with `other=0.0`,
  `tl.sum(..., axis=0)`, and scalar reduction-result store. f32 semantics are
  finite-tree target semantics, not exact IEEE ordering.
- `ttir_reduce_sum_i32_b16`: i32 vector<16> tail-masked load with `other=0`,
  `tl.sum(..., axis=0)`, and scalar reduction-result store.
- `ttir_reduce_sum_tail_scalar_store_b16`: f32 add reduction with a scalar
  `pid < num_blocks` store mask, used to define the scalar masked-store TTIR
  form for reduction outputs.
- `ttir_reduce_row_strided_sum_f32_b16`: Phase 11 scalar row-strided base plus
  contiguous `tl.arange(0, 16)` row-slice load, f32 add reduction, and scalar
  row output store.
- `mixed_ttir_reduction_axes_mask_cf_strided_b16`: combines Phase 9 axis0/axis1
  launch identity and `tl.num_programs(0)`, Phase 10 tail masks, Phase 11
  row-strided memory, Phase 8.5 scalar control flow, and Phase 12 f32 add
  reduction.

Staged snapshots:

- `ttir_reduce_max_reject_b16`: real `tl.max` TTIR emits `tt.reduce`, but the
  combiner is `arith.maxnumf`; non-add reductions remain staged.
- `ttir_reduce_rank2_axis_reject_b16`: real rank-2 axis reduction emits
  shape-changing TTIR and a vector reduction result; rank>1 reductions remain
  staged.

The accepted snapshots intentionally exclude unrelated staged features:
`sitofp`, `fptosi`, dot, GEMV/GEMM, block pointers, explicit gather/scatter,
subword storage, SFU/math expansion, TTGIR, TritonGPU, NVGPU, NVVM, and GPU
backend dialects are absent. Triton frontend integer overflow proof scaffolding
such as `arith.extsi`, `arith.cmpi`, and `arith.andi` remains normal emitted
TTIR provenance from prior phases and is not a new source feature.

The controlled TTIR shape for accepted add reductions is:

```text
tl.sum over tensor<16xT>
  -> private tt.func wrapper
  -> "tt.reduce" axis = 0
  -> scalar T result
  -> scalar tt.store to rank-1 output pointer
```

The f32 accepted fixtures require finite CPU oracle inputs in future hardware
phases and must lower with finite-tree target policy. Phase 12 does not claim
exact/default IEEE f32 reduction ordering.

Dot, GEMV, GEMM, product reductions, min/max reductions, custom reducers,
rank>1 reductions, scans, atomics, f16/subword reductions, numeric casts, and
SFU/generalized math remain staged.

`READY_FOR_TRITON=NO` remains locked.
