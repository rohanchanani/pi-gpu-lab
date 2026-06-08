# VC4 Vector/Triton Phase 5 - Value Elementwise Hardware Lock

## 1. Result

VC4_VALUE_TO_VC4KERNEL_PHASE5_ELEMENTWISE_LOCKED=YES
VC4_VALUE_TO_VC4KERNEL_PASS_ACCEPTED=YES
VC4_VALUE_ELEMENTWISE_VECTOR16_COMPUTE_ACCEPTED_HARDWARE=YES
VC4_VALUE_TRANSFER_READ_TMU_ACCEPTED_HARDWARE=YES
VC4_VALUE_TRANSFER_WRITE_VDW_PRESERVE_ACCEPTED_HARDWARE=YES
VC4_VALUE_ELEMENTWISE_ISOLATION_FIXTURES_ACCEPTED=YES
VC4_VALUE_ELEMENTWISE_MIXED_ACCEPTANCE_ACCEPTED=YES
VC4_VALUE_MIXED_FIXTURE_CLAIMS_AUDITED=YES
VC4_VALUE_PHASE5_VECTOR16_IS_FIRST_LOWERABLE_SUBSET=YES
READY_FOR_PHASE6_REAL_TTIR_INVENTORY_AND_IMPORTER_SKELETON=YES
READY_FOR_PHASE7_TTIR_ELEMENTWISE_HARDWARE=NO
READY_FOR_TRITON=NO

Phase 5 locks the first executable handwritten standard value-layer vertical
slice. The accepted path starts from `func` plus tiny `vc4value`, `vector`,
`memref`, and `arith` operations, lowers through VC4Kernel, SSAVC4, scheduled
VC4, generated artifacts/runtime, and passes real VC4 hardware checks.

## 2. Locked Stack Boundary

The only accepted executable path remains:

```text
standard value layer -> vc4kernel -> ssavc4 -> scheduled vc4 -> artifacts/runtime/hardware
```

Phase 5 does not add a direct VC4Kernel-to-VC4 route, does not revive VC4Tile,
does not allow producer dialect operations inside verified VC4Kernel, and does
not start Triton or TTIR ingestion.

## 3. Accepted Phase 5 V1 Lowerable Subset

The accepted kernel wrapper subset is a `func.func` with `vc4value.kernel` and
`vc4value.grid_rank = 1`. Public arguments follow the Phase 4 ABI: rank-1
contiguous i32/f32 `#vc4value.global` memrefs, scalar i32/index/f32 values, and
no hidden memref descriptor ABI.

The accepted launch and lane subset is `vc4value.program_id` and
`vc4value.num_programs` on axis 0, plus `vector.step` for the logical SIMD-16
lane range. Program id remains logical launch identity, not physical QPU
identity.

The accepted compute subset is:

- scalar and vector constants for supported i32/f32/index forms;
- `vector.splat` and `vector.broadcast` from supported scalars;
- `vector<16xi32>`, `vector<16xindex>` lowered to i32 where needed, and
  `vector<16xf32>`;
- i32 add/sub/shift where accepted by the Phase 5 conversion policy;
- f32 add/sub/mul;
- i32 comparisons;
- finite-policy ordered/equality f32 comparisons under
  `vc4value.fp_domain = "finite"`;
- `arith.select` with `vector<16xi1>` conditions and supported vector values.

The accepted memory subset is:

- `vector.transfer_read` from rank-1 contiguous i32/f32 global memrefs to
  `vector<16xi32>` or `vector<16xf32>`, lowered to
  `vc4kernel.tmu_load_fragment` with `tmu_global_read`, `readonly_tmu`,
  `inactive_load<zero>`, and explicit safe offset;
- `vector.transfer_write` from `vector<16xi32>` or `vector<16xf32>` to rank-1
  contiguous i32/f32 global memrefs, lowered to
  `vc4kernel.vdw_store_fragment` with `vdw_global_store`, `dma_ordered`, and
  `inactive_store<preserve>`;
- no-mask full predicates and `vector.create_mask` tail predicates.

Verified VC4Kernel output must contain no `func`, `vc4value`, `vector` dialect
ops, `memref`, `scf`, `tensor`, `linalg`, `gpu`, `tt`, or lower producer ops.

## 4. Staged Rejects

The following remain staged value-surface forms, not Phase 5 lowerable:

- program-id and num-programs axes 1 and 2;
- non-16 fixed rank-1 vectors and rank-2 vectors;
- rank-2 memrefs and non-identity or strided memref layouts;
- i8/i16/f16 memory lowering and native f16 arithmetic;
- reductions, contracts, gather/scatter, sparse stores, arbitrary shuffles, and
  vector masks not classified as full or tail for stores;
- SFU/math dialect lowering and unordered or NaN-sensitive f32 comparisons;
- `scf`/general control-flow lowering;
- Triton and TTIR ingestion.

These staged forms remain future value-layer work. Phase 5 does not collapse
the entire value surface to `vector<16>`.

## 5. Hardware Fixtures And Coverage

Phase 5 isolation fixtures accepted on real hardware:

- `value_copy_f32_tail_vc4value`: first value-to-hardware copy smoke,
  narrowed to `active_qpus=1` after the original `active_qpus=12` variant
  exposed a real overlaunch/tail-mask gap;
- `value_saxpy_f32_tail_vc4value`: f32 scalar splat, transfer reads, f32
  multiply/add, and tail VDW preserve;
- `value_i32_add_select_tail_vc4value`: i32 transfer reads, i32 arithmetic,
  signed compare, select, and i32 tail store;
- `value_f32_cmp_select_tail_vc4value`: finite f32 compare/select and f32 tail
  store;
- `value_store_preserve_sentinel_vc4value`: adversarial inactive-lane
  sentinel preservation;
- Phase 5h/5j mixed fixtures restore multi-request, `active_qpus=12` value
  coverage with checked hashes, sentinels, and strict CPU oracles.

Phase 5 mixed acceptance fixtures accepted on real hardware:

- `mixed_value_saxpy_select_tail_vc4value`: f32 TMU reads, f32 scalar
  splat/mul/add, finite f32 compare/select, tail VDW preserve, and
  multi-request `active_qpus=12` launch geometry;
- `mixed_value_i32_f32_dual_path_tail_vc4value`: i32 TMU read and compare
  feeding f32 arithmetic/select, f32 tail VDW preserve, and multi-request
  `active_qpus=12` launch geometry.

The mixed manifest records `value_i32_transfer_write` as an isolation-backed
staged mixed addition through `value_i32_add_select_tail_vc4value`.

## 6. Final Hardware Results

Value mixed final acceptance ran in:

```text
.vc4_auto/codegen_phase5k_value_mixed_final
```

Both value mixed fixtures reported `VC4_TEST_RESULT status=PASS` with
`total_mismatches=0`, `sentinel_mismatches=0`, `launch_failures=0`,
`active_qpus=12`, and nonzero output hashes.

The locked VC4Kernel mixed regression suite ran in:

```text
.vc4_auto/codegen_phase5k_vc4kernel_mixed_regression
```

All 25 VC4Kernel mixed fixtures reported `VC4_TEST_RESULT status=PASS`, proving
that the lower-stack gate used by value lowering remained intact.

## 7. Intermediate Layer Validation Policy

The value hardware runner stores stable intermediates under the configured
state root and rejects stale generated candidates unless explicit reuse is
requested. It verifies each boundary:

- value input contains `vc4value.kernel` and no source VC4Kernel, SSAVC4, or
  scheduled VC4;
- converted VC4Kernel contains `vc4kernel.kernel` and no producer dialect ops;
- SSAVC4 contains no producer dialects or VC4Kernel ops;
- scheduled VC4 contains scheduled VC4 modules/QPU ops and no SSAVC4 or
  VC4Kernel ops;
- generated artifacts are assembled, built, power-cycled, run on real hardware,
  and checked against fixture `expected.json`.

TMU load fragments must use explicit safe offsets and `inactive_load<zero>`.
VDW store fragments must use `inactive_store<preserve>`.

## 8. Debugging Doctrine

If a future value mixed fixture fails, run the relevant isolation fixtures,
reduce only after preserving the failing case, inspect value IR, VC4Kernel,
SSAVC4, scheduled VC4, emitted artifacts, and runtime packing, and instrument
only temporarily. Do not weaken CPU oracles, sentinels, dimensions, diagnostics,
or hardware checks. If the evidence points to a real lower-half bug, report it
as a blocker rather than reshaping the value kernel to hide it.

## 9. Static Acceptance

Phase 5 final static acceptance included value dialect, value surface,
VC4Value-to-VC4Kernel, VC4Kernel-to-SSAVC4, SSAVC4-to-VC4, VC4Value codegen,
VC4Kernel codegen, value matrix/audits, value mixed coverage and claim audits,
P13 VC4Kernel final audits, `check-vc4`, and `git diff --check`.

## 10. Phase 6 Handoff

The handwritten value path is now hardware-proven for Phase 5 elementwise V1.
Phase 6 may inventory real emitted TTIR and design/import the elementwise
subset into the standard value layer.

Phase 6 and Phase 7 must still lower TTIR into the value layer first. There is
no direct TTIR-to-VC4Kernel path. Phase 7 TTIR hardware smoke may use the Phase
5 path only after Phase 6 real TTIR inventory/importer skeleton succeeds.

Triton remains not ready:

```text
READY_FOR_TRITON=NO
```
