# VC4 Vector/Triton Phase 1 Lock

## 1. Result

Phase 1 is complete as a docs/spec package. It added the value/Triton taxonomy,
the standard value-surface specification, the TTIR target profile, the
permanent-reject proof policy, and the value-to-VC4Kernel planning guide.

Phase 1 added no executable value/Triton semantics. It did not implement
`vc4value`, value-to-VC4Kernel lowering, TTIR import, Triton support, tests that
require Triton, runtime changes, or hardware fixtures.

Phase 1 preserves the locked path:

```text
standard value layer -> vc4kernel -> ssavc4 -> scheduled vc4
  -> artifacts/runtime/hardware
```

## 2. Source-controlled deliverables

Phase 1 deliverables:

- `compiler/docs/vc4_value_ttir_taxonomy_schema.md`
- `compiler/docs/vc4_value_ttir_feature_taxonomy.json`
- `compiler/docs/vc4_value_surface_spec.md`
- `compiler/docs/vc4_ttir_target_profile.md`
- `compiler/docs/vc4_ttir_reject_proof_policy.md`
- `compiler/docs/vc4_value_to_vc4kernel_planning.md`
- `compiler/docs/vc4_vector_triton_phase1_lock.md`

The taxonomy JSON is the machine-readable feature matrix. The remaining files
are human-readable contracts for later implementation phases.

## 3. Locked Phase 1 decisions

Locked decisions:

- standard value layer first;
- tiny `vc4value` only;
- `func.func` wrapper;
- `memref` logical memory;
- `vector.step` lane model;
- value surface does not expose TMU, VDR, VDW, or VPM;
- planner chooses memory path;
- TTIR ingestion later;
- FULL means supported/lowered or permanent reject with proof.

The tiny `vc4value` scope is limited to launch/policy plumbing:

```text
vc4value.program_id
vc4value.num_programs
kernel/grid/policy metadata
```

The value layer expresses standard semantics. VC4Kernel remains the target
planning dialect, and verified VC4Kernel may contain only accepted VC4Kernel
operations and carrier types.

## 4. Taxonomy status categories

The taxonomy status categories are:

- `supported_native`
- `supported_composite`
- `supported_emulated`
- `supported_with_policy_caveat`
- `supportable_later`
- `temporary_reject_not_implemented`
- `deterministic_reject_value_surface_policy`
- `deterministic_reject_vc4kernel_contract`
- `permanent_reject_hardware_or_semantic_impossible`
- `out_of_scope_non_compute_hardware`
- `internal_only`

No feature may be classified as a permanent reject merely because it is hard or
not yet implemented. Permanent reject requires proof that VC4 cannot preserve
the required semantics even through slow emulation, or that the feature is
outside the compute target.

## 5. Initial value surface summary

The standard value surface is:

```text
builtin, func, vc4value, vector, memref, arith, math, scf, cf
```

Initial value surface decisions:

- `vc4value.program_id` and `vc4value.num_programs` provide logical launch-grid
  identity;
- `func.func` with `vc4value.kernel` and grid metadata is the wrapper model;
- `memref` carries logical rank, type, layout, memory space, and access attrs;
- `vector.step` is the lane/arange form;
- `vector.transfer_read` and `vector.transfer_write` are the initial memory
  transfer forms;
- mask classes are full, empty, tail, rect, sparse, and unknown;
- f16 storage conversion plus f32 compute is the accepted f16 value model;
- native f16 arithmetic and native bf16/fp8 arithmetic/conversion remain locked
  target rejects.

The value layer does not expose `vc4value` memory, tile, fragment, lane-id, or
barrier ops.

## 6. TTIR target-profile summary

The TTIR profile is conceptual until Phase 6 inventories real emitted TTIR from
a pinned Triton version. Formal support is based on emitted TTIR / `tt` dialect
families, while demo credibility comes from real Triton Python kernels that emit
those families.

The future importer shape is:

```text
vc4-triton-import input.ttir.mlir -o output.vc4value.mlir
```

The first smoke profile is elementwise and excludes dot, reductions, and block
pointers. TTIR ingestion starts only after the handwritten value path is proven.

## 7. Value-to-VC4Kernel planning summary

The planner maps value patterns into accepted VC4Kernel target operations only.
It chooses target memory paths:

- contiguous/tail low-reuse loads may plan to TMU safe-offset inactive-zero;
- structured tile/reuse loads may plan to VDR -> VPM -> QPU VPM read;
- contiguous/tail stores may plan to VDW inactive preserve;
- rectangular tile stores may plan to VPM/VDW;
- sparse stores remain direct VC4Kernel rejects unless a future hardware-proven
  phase changes the target contract.

P12 coordinate and selector rules remain locked: row, word-X, and dynamic
subword selector are separate fields; dynamic selector is byte/halfword selector
inside a static packed mode; setup field masks are not modulo semantics; VDR
and VDW selector semantics are separately proven.

Lane broadcast is a composite:

```text
lane_range -> compare lane == selected_lane -> select value or zero
  -> fragment_reduce add
```

The first contract shape is row-fragment GEMM:

```text
BLOCK_M=1
BLOCK_N=16
BLOCK_K=4 first
```

## 8. Explicit non-goals

No Phase 1 Triton support exists.

Phase 1 did not:

- implement `vc4value`;
- implement value-to-VC4Kernel lowering;
- implement TTIR/Triton import;
- add tests that require Triton;
- run hardware;
- add direct VC4KernelToVC4;
- revive VC4Tile;
- permit producer dialect ops inside verified VC4Kernel;
- expose TMU, VDR, VDW, or VPM as value-surface ops.

Documentation may reference forbidden forms as non-goals or deterministic
rejects only. Such references are not support claims.

## 9. Consistency audit results

Phase 1 consistency audit passed:

- all seven Phase 1 deliverables exist;
- taxonomy JSON is valid;
- all mandatory taxonomy feature IDs are present;
- no taxonomy feature classification is `UNKNOWN`;
- hard-phrase hits are contextual non-goals, hard exclusions, or reject-policy
  references, not positive support claims;
- required positive phrases appear across the Phase 1 docs and this lock;
- taxonomy JSON, value surface spec, TTIR profile, and planning guide agree on
  tiny `vc4value`, `func.func`, `memref`, `vector.step`, mask classes, sparse
  store reject, f16 storage versus native f16 arithmetic, approximate SFU only,
  lane broadcast composite, first contract shape, and TTIR only after the
  handwritten value path;
- Phase 1 docs do not contradict the VC4Kernel final lock matrix or readiness
  lines.

The VC4Kernel final matrix remains 44 features: 30
`accepted_hardware_proven`, 12 `deterministic_reject`, 2 `internal_only`, 0
nonfinal rows, generated by `P13c_final_surface_lock`.

## 10. Phase 2 handoff

Phase 2 should implement only the tiny `vc4value` launch/policy layer:

```text
vc4value.program_id
vc4value.num_programs
vc4value.kernel / grid attrs as scoped by the docs
```

Phase 2 must not implement memory, tile, fragment, lane-id, barrier, TMU, VDR,
VDW, or VPM operations. Phase 2 must not implement value-to-VC4Kernel lowering
or TTIR ingestion.

## 11. Readiness lines

```text
VC4_VECTOR_TRITON_PHASE1_TAXONOMY_SPEC_LOCKED=YES
READY_FOR_PHASE2_TINY_VC4VALUE=YES
READY_FOR_VALUE_TO_VC4KERNEL_LOWERING=NO
READY_FOR_TRITON=NO
```
