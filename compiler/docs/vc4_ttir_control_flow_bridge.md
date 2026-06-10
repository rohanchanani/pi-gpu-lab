PHASE85_TTIR_CONTROL_FLOW_CONTRACT=ACTIVE
REAL_TTIR_SNAPSHOTS_SOURCE_CONTROLLED=YES
ANY_REAL_RUNTIME_TTIR_CF_LOWERABLE_NOW=YES
TTIR_TL_RANGE_LOOP_SKELETON_POLICY=SCF_CF
TTIR_PERSISTENT_LOOP_SKELETON_POLICY=SCF_CF
TTIR_STATIC_RANGE_POLICY=STATIC_SPECIALIZED_NO_RUNTIME_CF
TTIR_SCALAR_IF_POLICY=SCF_CF
TTIR_WHILE_POLICY=SCF_CF
TTIR_VECTOR_BRANCH_CONDITION_POLICY=DETERMINISTIC_REJECT_AS_CFG_USE_MASKS
TTIR_UNSUPPORTED_LOOP_BODY_POLICY=STAGE_BY_BODY_FEATURE_NOT_CF
TTIR_BACKEND_DIALECT_CF_POLICY=DETERMINISTIC_REJECT_SOURCE_BOUNDARY
READY_FOR_PHASE85C_IMPORTER_REGION_FRAMEWORK=YES
READY_FOR_TRITON=NO

# VC4 TTIR Control-Flow Bridge Contract

## 1. Purpose and stack boundary

This document locks the Phase 8.5 TTIR control-flow bridge contract. It is a
docs, taxonomy, and test contract only. It does not implement importer support,
does not run hardware, and does not claim full Triton readiness.

The accepted stack remains:

```text
real Triton source
  -> real emitted TTIR / tt dialect
  -> standard VC4 value surface
       func + tiny vc4value + vector + memref + arith + math + scf/cf
  -> vc4kernel
  -> ssavc4
  -> scheduled vc4
  -> artifacts/runtime
  -> real VC4 hardware
```

TTIR control flow must enter through the value `scf`/`cf` surface. It must not
lower directly to `vc4kernel`, `ssavc4`, scheduled `vc4`, TTGIR, vendor GPU IR,
LLVM IR, PTX, cubin, or hsaco.

## 2. Snapshot generation and provenance

The contract is derived from Phase85a1 and Phase85a2:

- Phase85a1 generated real Triton 3.7.0 TTIR from real Triton Python probes
  using `tools/vc4_emit_ttir.py`.
- The generator used Triton frontend `ASTSource.make_ir`; it did not use the
  retired Python semantic importer.
- No LLVM source build was used.
- Phase85a2 promoted the source files and generated TTIR snapshots into:

```text
compiler/test/CodeGen/Triton/Snapshots/ControlFlow/
```

The promoted snapshots parse and roundtrip with:

```text
compiler/build-triton-llvm/bin/vc4-triton-opt
```

## 3. Real TTIR inventory summary

The source-controlled Phase85a2 snapshots classify the exact observed forms:

- `tl_range_loop_skeleton`: `tl.range` emitted `scf.for` with loop-carried
  `tensor<16xf32>`, `scf.yield`, and `tt.loop_unroll_factor = 1`.
- `persistent_loop_skeleton`: persistent-style `tl.range(start_pid, num_tiles,
  NUM_SMS, flatten=True)` emitted `scf.for` with `tt.flatten` and
  `tt.loop_unroll_factor = 1`.
- `static_range_probe`: `tl.static_range(0, 3, 1)` emitted no runtime
  `scf`/`cf`; it specialized into straight-line TTIR.
- `scalar_if_probe`: runtime scalar `if flag > 0` emitted `scf.if` returning
  `tensor<16xf32>` with `scf.yield` in both regions.
- `while_probe`: runtime while emitted `scf.while`, `scf.condition`,
  `scf.yield`, and loop-carried `tensor<16xf32>, i32`.

No promoted Phase85a2 snapshot contains `tt.dot`, `tt.reduce`, `tt.advance`,
`tt.make_block_ptr`, `ttg.`, `triton_gpu.`, `nvgpu.`, or `nvvm.`.

## 4. Supported Phase 8.5 forms

The following observed real TTIR forms are lowerable Phase 8.5 implementation
targets if the body remains inside the supported value subset:

- `scf.for` emitted by `tl.range`;
- persistent-loop `scf.for` emitted by real Triton source;
- scalar `scf.if`;
- `scf.while`;
- loop-carried scalar/vector values used by those snapshots;
- rank-1 contiguous pointer expressions of the form
  `base + lane + loop_iv * constant_stride` for vector<16> bodies.

This is not a claim that the importer already lowers those forms. Phase85c is
the earliest region-framework implementation phase.

## 5. Static-specialized or frontend-rejected forms

`tl.static_range` is classified as `STATIC_SPECIALIZED_NO_RUNTIME_CF` for the
promoted Phase85a2 snapshot. It is useful classification evidence, but it is not
runtime control-flow support.

No Phase85a2 control-flow probe was frontend-rejected. Future frontend-rejected
forms must be documented with real source/diagnostics, not fake TTIR.

## 6. Staged-by-body-feature forms

Unsupported loop-body features are staged by exact body op, not by generic
control-flow failure. The Phase85a2 promoted set has:

```text
UNSUPPORTED_LOOP_BODY_FEATURES=none
```

Future control-flow snapshots containing `tt.dot`, `tt.reduce`,
`tt.make_block_ptr`, block pointers, tensor descriptors, atomics, or backend
dialect operations must stage or reject those exact body features while keeping
the control-flow skeleton classification separate.

## 7. Deterministic rejects

Vector or per-lane branch conditions are rejected as branch CFG:

```text
TTIR_VECTOR_BRANCH_CONDITION_POLICY=DETERMINISTIC_REJECT_AS_CFG_USE_MASKS
```

VC4 QPUs execute one coherent program counter across SIMD-16 lanes. Lane-varying
choices must be expressed as mask/select dataflow when supported, not as
per-lane branch control flow.

Backend dialect control flow rejects at the source boundary:

```text
TTIR_BACKEND_DIALECT_CF_POLICY=DETERMINISTIC_REJECT_SOURCE_BOUNDARY
```

Accepted Phase 8.5 input is real TTIR / `tt` plus standard MLIR control-flow
forms. TTGIR, `triton_gpu`, `nvgpu`, and `nvvm` are not accepted source input
for the VC4 path.

## 8. Loop attribute policy

Phase85a2 observed surviving loop attrs:

```text
LOOP_ATTRS_SURVIVE=YES: tt.loop_unroll_factor, tt.flatten
```

`tt.loop_unroll_factor` and `tt.flatten` must be parsed structurally and
preserved or deliberately classified. Phase85 implementation may ignore them
only if the implementation contract proves they are optimization-only for the
accepted source form. If an attr changes runtime semantics, it must be staged or
rejected with an exact diagnostic.

`warp_specialize` was not emitted by Phase85a2:

```text
WARP_SPECIALIZE_ATTR_POLICY_CANDIDATE=NOT_EMITTED
```

Do not claim a warp-specialization policy until real parseable TTIR evidence is
promoted.

## 9. Pointer/index expression policy for loop skeletons

The Phase85 bridge may target scalar affine loop-index contributions into
rank-1 contiguous vector pointer expressions:

```text
base + lane + loop_iv * constant_stride
```

Initial lowerability is restricted to vector<16> bodies, scalar loop IVs,
supported `i32`/index arithmetic, and global pointer tensors whose memory
effects remain canonical tail-masked load/store forms. Non-affine, rank-2,
block-pointer, gather/scatter, sparse-store, or body-feature-staged memory forms
must be rejected or staged by exact reason.

## 10. Hardware plan

Because real runtime TTIR control flow exists, hardware proof is required after
implementation:

- `triton_tl_range_accumulate_tail_vc4triton`;
- `triton_persistent_loop_skeleton_vc4triton`;
- `triton_scalar_if_control_vc4triton`;
- `triton_while_control_vc4triton`;
- `triton_static_range_specialized_classification`;
- `mixed_ttir_cf_range_persistent_tail_vc4triton`.

Hardware fixtures must lower through:

```text
TTIR -> value -> vc4kernel -> ssavc4 -> scheduled vc4 -> artifacts/runtime
```

No hardware proof is claimed by this document phase.

## 11. Future phases

Phase85c should implement the importer region framework for the observed real
runtime TTIR control-flow forms. Later phases may add body-feature support or
new control-flow forms only after source-controlled real TTIR evidence is
generated, parsed, structurally classified, and documented.

Phase 9 must not become active before the Phase 8.5 TTIR control-flow bridge is
locked or deliberately finalized through a no-runtime-CF lock. Since Phase85a2
found real runtime TTIR control flow:

```text
READY_FOR_PHASE85C_IMPORTER_REGION_FRAMEWORK=YES
READY_FOR_PHASE85N_NO_RUNTIME_CF_FINAL_LOCK=NO
READY_FOR_TRITON=NO
```
