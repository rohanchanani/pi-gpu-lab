# VC4 TTIR Target Profile

## 1. Purpose

This document defines the conceptual TTIR-facing target profile for VC4. It is
a Phase 1 specification artifact only. It does not inspect a pinned Triton
version, does not implement TTIR import, does not add tests that require
Triton, and does not claim executable Triton support.

The profile describes the importer expectations for future work:

```text
Triton / emitted TTIR
  -> standard VC4 value surface
       func + tiny vc4value + vector + memref + arith + math + scf/cf
  -> vc4kernel
  -> ssavc4
  -> scheduled vc4
  -> artifacts/runtime/hardware
```

The value-surface contract is specified in
`compiler/docs/vc4_value_surface_spec.md`. Feature classification is tracked in
`compiler/docs/vc4_value_ttir_feature_taxonomy.json`.
Phase 3.5 vector abstraction policy is specified in
`compiler/docs/vc4_value_surface_abstraction_policy.md`.

## 2. Definition of FULL for this project

Definition of FULL:

Every relevant Triton/TTIR construct eventually either lowers through the value
surface to hardware, lowers with an explicit caveat/emulation, or
deterministically rejects with a compelling hardware/semantic proof.

FULL is a long-term target profile, not a Phase 1 implementation claim.
Temporary rejects are expected while the handwritten value path, importer, and
target-profile tests are staged. A temporary reject means no safe lowering has
been implemented or specified yet.

A permanent reject is different. It requires proof that VC4 cannot preserve the
required source semantics even through slow emulation, or that the construct is
outside the VC4 compute-kernel target. The proof policy is specified in
`compiler/docs/vc4_ttir_reject_proof_policy.md`.

## 3. Source of truth: TTIR forms versus Triton Python demos

The formal compiler support matrix is based on emitted TTIR / `tt` dialect
forms, not on Python source snippets alone.

Triton Python demos still matter: a demo is credible when it is a real Triton
kernel that emits the TTIR families classified by this profile. Phase 6 will
inventory real emitted TTIR from a pinned Triton version before importer work
claims exact operation spelling support.

Phase 1 should therefore use stable conceptual families such as program-id
queries, arange-like lane construction, masked loads/stores, elementwise ops,
reductions, dot, block pointers, and metadata. It should not overfit to exact
current TTIR spelling before Phase 6.

## 4. Import architecture

The accepted semantic importer shape after Phase 7.5 is the optional C++ MLIR
tool:

```text
compiler/build-triton-llvm/bin/vc4-triton-opt input.ttir.mlir \
  --convert-triton-to-vc4-value \
  -o output.vc4value.mlir
```

This path is gated by `VC4_ENABLE_TRITON_CPP_FRONTEND=ON`; the default
`compiler/build` lane remains Triton-free and has no `vc4-triton-opt` target.
`VC4TritonFrontendDeps` is the only CMake target that may carry Triton
source/build/include/object/library closure details.

The importer parses MLIR structurally through Triton/MLIR dialect machinery.
It must not use a regex parser or dispatch by kernel name.

The first path is real emitted TTIR into the standard VC4 value surface. It is
not direct Triton-to-VC4Kernel, not TTIR-to-VC4Kernel, and not TTIR-to-SSAVC4.
It must not accept TTGIR/NVIDIA-specific IR as the first VC4 path.

Python remains allowed for Triton Python source to `.ttir.mlir` snapshot
generation and inventory. The old Phase 7 Python semantic lowering mode,
`vc4-triton-import --mode lower-elementwise-v1`, is retired from accepted
lowering tests and hardware proof and is hard-disabled with a diagnostic that
points users to the C++ importer.

Normal `check-vc4` must not require Triton regeneration. TTIR fixtures, once
they exist, should be checked in or generated through a controlled non-default
workflow so ordinary compiler tests remain deterministic without requiring a
Triton installation.

## 5. Initial TTIR smoke profile

The initial smoke profile is a narrow conceptual subset:

- `tt.func`-like kernel wrapper family into `func.func` plus `vc4value`
  metadata;
- `tt.get_program_id` family into `vc4value.program_id`;
- `tt.arange` family into `vector.step`, splat, and add patterns;
- masked load with `other=0` into `vector.transfer_read` for contiguous/tail
  cases;
- masked store into `vector.transfer_write` for dense/tail cases;
- simple arithmetic, comparison, and select;
- `BLOCK_SIZE=16` first for the Phase 5 V1 lowerable fragment-normal form.

The first smoke profile explicitly excludes dot, reductions, and block pointers.
Those families are staged later so the importer does not conflate a small
elementwise smoke test with broader Triton support.

### Phase 7c implemented static subset

Phase 7c implements static TTIR-to-value lowering for exactly the three real
Triton 3.7.0 Phase 6 elementwise snapshots:

```text
examples/triton/phase6/generated/vector_add_b16.ttir.mlir
examples/triton/phase6/generated/saxpy_select_b16.ttir.mlir
examples/triton/phase6/generated/i32_add_select_b16.ttir.mlir
```

The accepted forms are the Phase 7 V1 elementwise profile only:

- one public axis-0 `tt.func` kernel with `tt.return`;
- `tt.get_program_id x` and `tt.make_range` exactly representing `0..16`;
- contiguous pointer expressions equivalent to `base + (pid * 16 + lane)`;
- masked `tt.load` with `other=0` or `other=0.0`;
- masked `tt.store` with the same canonical tail mask;
- scalar `i32`/`f32` splats, `i32`/`f32` elementwise arithmetic, accepted
  comparisons, and vector select forms used by the three snapshots.

The importer emits standard value IR only. It does not emit `vc4kernel`,
`ssavc4`, scheduled `vc4`, TTGIR, vendor GPU IR, LLVM IR, PTX, cubin, or hsaco.

Phase 7c remains static only. It is not hardware proof and does not make
Triton support complete:

```text
READY_FOR_TRITON=NO
```

### Phase 9.7 multi-axis launch identity static subset

Phase 9.7 extends the optional C++ importer for the controlled Phase 9 real
Triton 3.7.0 TTIR snapshots:

```text
examples/triton/phase9_multi_axis_launch/generated/
```

The accepted static TTIR forms are:

- `tt.get_program_id` axes x/y/z to `vc4value.program_id` axes 0/1/2;
- `tt.get_num_programs` axes x/y/z to `vc4value.num_programs` axes 0/1/2;
- `vc4value.grid_rank` inferred as maximum used accepted axis plus one;
- canonical 2D/3D linearized launch identity feeding flattened rank-1
  transfers;
- canonical tail mask `idx < n` lowered to `vector.create_mask(n - base)`.

```text
TTIR_MULTI_AXIS_IMPORTER_STATIC=PASS
TTIR_PROGRAM_ID_AXES_0_1_2_TO_VALUE=YES
TTIR_NUM_PROGRAMS_AXES_0_1_2_TO_VALUE=YES
TTIR_MULTI_AXIS_LINEARIZED_TAIL_TO_VALUE=YES
NO_STRINGLY_AXIS_CLASSIFICATION=YES
FRONTEND_ROBUSTNESS_AUDIT=PASS
READY_FOR_PHASE9_8_TTIR_HARDWARE_ISOLATION=YES
READY_FOR_TRITON=NO
```

The importer classifies axes through typed Triton op attributes and SSA
use-def structure. It does not parse raw TTIR text and does not use fixture
names, source names, public names, file paths, or generated paths as semantic
signals.

This is not Phase 10 mask or memory legality. Noncanonical masks, sparse
masks, rank-2 memory, block pointers, gather/scatter, dot, reduce,
`arith.sitofp`, and `arith.fptosi` remain staged or rejected. Global
`READY_FOR_TRITON` remains `NO`.

### Phase 7h hardware-proven elementwise V1 subset

Phase 7 originally hardware-proved the first real TTIR executable path for the
Phase 6 real elementwise corpus through the Python Phase 7 importer. Phase 7.5
re-locked the accepted path through the optional C++ importer:

```text
real emitted TTIR snapshot
  -> vc4-triton-opt --convert-triton-to-vc4-value
  -> standard VC4 value IR
  -> vc4kernel
  -> ssavc4
  -> scheduled vc4
  -> artifacts/runtime
  -> real VC4 hardware
```

The proof source is the checked-in real Triton 3.7.0 TTIR snapshots, not fake
or hand-written TTIR:

```text
examples/triton/phase6/generated/vector_add_b16.ttir.mlir
examples/triton/phase6/generated/saxpy_select_b16.ttir.mlir
examples/triton/phase6/generated/i32_add_select_b16.ttir.mlir
```

The accepted/hardware-proven operation forms remain the narrow elementwise V1
profile:

- one public `tt.func` kernel and `tt.return`;
- axis-0 `tt.get_program_id` only;
- `tt.make_range` / arange pattern exactly representing `0..16`;
- scalar splats and broadcasts needed by the three snapshots;
- contiguous pointer expressions equivalent to `base + (pid * 16 + lane)`;
- masked `tt.load` with zero `other` operands;
- canonical tail-masked `tt.store`;
- tensor `i32` and `f32` elementwise add/sub/mul where present;
- finite `f32` and exact `i32` compare/select forms accepted by the Phase 5
  value lowering contract;
- active_qpus=12, lanes=16, adversarial tail/overlaunch sizes, strict CPU
  oracles, sentinels, and nonzero output hashes.

Phase 7h does not accept reductions, dot/contract, gather/scatter, block
pointers, atomics, rank-2 tensors, subword or f16/bf16/int8 memory lowering,
SFU/math, `scf`/control flow, `BLOCK_SIZE != 16`, or program-id axes 1/2.
Those are staged future work unless a later proof classifies a specific form
as a surface-policy or hardware-forbidden deterministic reject.

The locked Phase 7 support matrix is
`compiler/docs/vc4_ttir_elementwise_v1_support_matrix.json`. The mixed
hardware gate is under
`compiler/test/CodeGen/Triton/Hardware/MixedAcceptance/`.

The Phase 7.5 C++ frontend lock is recorded in
`compiler/docs/vc4_vector_triton_phase7_5_cpp_frontend_lock.md`.

Even after Phase 7h:

```text
READY_FOR_TRITON=NO
```

### Phase 8.5R3 controlled control-flow importer subset

```text
PHASE85_STATIC_TTIR_CONTROL_FLOW_LOCK=YES
TTIR_CONTROL_FLOW_FEATURE_DRIVEN_FIXTURES=YES
REAL_TRITON_CF_SOURCES=YES
REAL_TTIR_CF_SNAPSHOTS=YES
PHASE85_TTIR_REGION_LOWERING_FRAMEWORK=YES
TTIR_REGION_LOWERING_FRAMEWORK=YES
PHASE85_CONTROLLED_TTIR_CF_STATIC_PIPELINE=PASS
TTIR_CF_STATIC_PIPELINE=PASS
VECTOR_INDEX_CAST_IMPORTER_BOILERPLATE_REMOVED=YES
ARITH_SITOFP_STAGED_BY_BODY_FEATURE=YES
FRONTEND_ROBUSTNESS_AUDIT=PASS
NO_TTIR_TOOLCHAIN_REBUILD=YES
READY_FOR_PHASE85R4_STATIC_AUDITS_CONTRACT_LOCK=YES
READY_FOR_PHASE85R5_HARDWARE_ISOLATION=YES
READY_FOR_TRITON=NO
```

Phase 8.5R3 adds structural TTIR region lowering for the source-controlled
Phase 8.5 control-flow snapshots:

```text
examples/triton/phase8_5_control_flow/generated/
```

The accepted forms are scalar/coherent TTIR control flow: `scf.for`, `scf.if`,
and `scf.while` inside real emitted `tt.func` bodies. The importer lowers these
regions into the standard value surface using scoped SSA mappings for region
block arguments and loop-carried values.

`tt.get_num_programs x` is accepted for the controlled persistent-loop skeleton
and lowers through `vc4value.num_programs`. Program-id axes 1/2 and
num-programs axes 1/2 remain staged until the value grid-rank contract expands.

`tt.make_range 0..16` is consumed structurally for canonical contiguous
pointer/tail-mask expressions instead of emitting a dead common-prefix
`vector<16xindex>` to `vector<16xi32>` cast. Data uses of `tt.make_range` are
handled separately from the transfer path.

This subset intentionally does not add numeric-cast support. `arith.sitofp`
is staged by body feature and is not accepted merely because it appears inside
control flow. Dot, reduce, gather/scatter, block pointers, tensor descriptors,
atomics, subword forms, SFU/math expansion, TTGIR, TritonGPU, NVGPU, NVVM, GPU,
and lower-half dialects remain outside this control-flow feature band.

Phase 8.5R4 locks the static contract before hardware. The controlled sources
and generated TTIR snapshots are source-controlled provenance artifacts, later
implementation and hardware phases consume those snapshots without TTIR
regeneration, and the accepted control-flow snapshots statically reach scheduled
VC4 through the normal value, VC4Kernel, SSAVC4, and scheduled VC4 path.

## 6. Launch and program IDs

TTIR launch identity maps through value-layer launch plumbing:

```text
program_id family -> vc4value.program_id -> vc4kernel.program_id
num_programs family -> vc4value.num_programs -> vc4kernel num programs
```

Program IDs are logical launch-grid identities. They are not physical QPU IDs
and must not expose physical scheduling to the value layer.

Axes 0, 1, and 2 are supportable. The initial smoke profile should start with a
1D grid. 2D and 3D grids are target-profile extensions once the handwritten
value path and ABI model are stable.

## 7. Arange, offsets, and block sizes

Arange-like TTIR forms lower conceptually through the value lane model. In the
Phase 8.5R3 accepted contiguous transfer path, `tt.make_range 0..16` is
recognized structurally as the lane component of pointer and mask expressions,
and the importer emits the scalar transfer base plus value transfer operations
without dead vector index-cast boilerplate. Future true data uses may materialize
a dense lane vector or be staged by exact diagnostic, depending on the feature
contract.

The earlier conceptual lane model remains:

```text
vector.step -> vector<16xindex> lane range
```

`BLOCK_SIZE=16` is first because it matches the Phase 5 V1 lowerable
`vector<16>` fragment-normal form. This is not the global value-layer vector
limit. Fixed rank-1 `vector<NxT>` with `N != 16` is surface-admissible and
staged until splitting, tail masks, and loop structure are specified. Fixed
rank-2 vectors are surface-admissible and staged for tile, contract, and VPM
planning. Scalable vectors remain initial deterministic rejects.

Offsets should remain ordinary value expressions over `index` or proven i32
values until the ABI profile proves target conversion is safe.

## 8. Loads, stores, masks, and pointer patterns

Contiguous/tail tensor-of-pointer patterns map to value transfer operations:

```text
contiguous/tail load -> vector.transfer_read
contiguous/tail store -> vector.transfer_write
```

The value layer does not name TMU, VDR, VDW, or VPM. The planner chooses those
mechanisms later.

Affine strided loads are later strided transfer or gather candidates. Gather
loads are later vector-gather/TMU candidates and require address, mask, and
inactive-lane safety analysis.

Dense, tail, and rectangular stores are candidates for
`vector.transfer_write` and later VDW inactive-preserve planning. Arbitrary
sparse stores reject unless future hardware-proven support exists or a separate
value-level emulation strategy is specified.

Unknown store masks must be diagnosed before target lowering. They must not
silently become sparse VDW stores.

Phase 7c diagnostics reject V1-adjacent forms including nonzero load `other`,
unmasked loads/stores that the importer cannot prove full and canonical,
non-affine or scattered pointer expressions, axis-1/axis-2 program ids,
`BLOCK_SIZE != 16`, rank-2/block-pointer forms, unsupported element types, dot,
and reductions.

## 9. Elementwise arithmetic, comparisons, and select

The first TTIR arithmetic profile covers simple integer and floating-point
elementwise arithmetic, comparisons, constants, splats, and select-like forms.

i32 arithmetic and comparisons map naturally into the value surface. f32
comparison support requires finite-policy classification before it lowers to
the locked VC4Kernel finite f32 comparison contract.

Select-like forms map to value `select` over vector masks. The masks must still
be classified before they drive memory effects.

## 9.5 TTIR control-flow bridge status

Phase 8 locks value-layer `scf`/`cf` control flow through the standard
`value -> vc4kernel -> ssavc4 -> scheduled vc4` path. It does not claim TTIR
control-flow import. Phase 8.5 owns TTIR control-flow inventory,
support/reject classification, and any executable bridge work.

Phase85a2 promoted real Triton 3.7.0 control-flow TTIR snapshots under
`compiler/test/CodeGen/Triton/Snapshots/ControlFlow/`. The Phase85b canonical
contract is `compiler/docs/vc4_ttir_control_flow_bridge.md`.

PHASE8R_CF_COMPLETENESS_SCOPE=ACTIVE
SCF_WHILE_VALUE_TARGET=SUPPORT_NOW
NESTED_STRUCTURED_CF_VALUE_TARGET=SUPPORT_NOW
TL_RANGE_STYLE_LOOP_SKELETON_VALUE_TARGET=SUPPORT_NOW
PERSISTENT_LOOP_SKELETON_VALUE_TARGET=SUPPORT_NOW
VECTOR_BRANCH_CONDITION_POLICY=DETERMINISTIC_REJECT_AS_CFG_USE_MASKS
PHASE8R_CF_SWITCH_INDEX_SWITCH_POLICY=LOCKED
PHASE8R_MULTI_EXIT_REDUCIBLE_LOOP_POLICY=LOCKED
PHASE8R_IRREDUCIBLE_CFG_POLICY=PROBED_NOT_REQUIRED_FOR_TRITON
IRREDUCIBLE_CFG_POLICY=PROBE_NOT_REQUIRED_FOR_SANE_TRITON
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

QPU control flow is scalar/coherent across SIMD lanes. Per-lane control
divergence is not accepted as branch CFG; masks/selects carry lane-varying
dataflow. Value-level `scf.while`, nested structured control flow,
tl.range-style loop skeletons, and persistent-loop skeletons are the relevant
support targets before TTIR import.

Phase 8Rd additionally locks scalar `cf.switch` by lowering it to explicit
scalar branch chains, supports `scf.index_switch` when upstream SCF-to-CF emits
that `cf.switch`, and proves multi-exit reducible loops statically through
scheduled VC4. Irreducible CFG is probed and currently limited by lower-half
natural-loop block-argument lowering; it is not required for sane Triton Phase
8.5 control-flow import and is not classified as hardware-impossible.

Locked Phase 8.5 classifications:

- TTIR `tl.range` loop skeleton:
  `SCF_CF`; real Triton emitted `scf.for` with loop-carried
  `tensor<16xf32>`, `scf.yield`, and `tt.loop_unroll_factor = 1`.
- TTIR persistent loop skeleton:
  `SCF_CF`; real Triton emitted `scf.for` with `tt.flatten` and
  `tt.loop_unroll_factor = 1`.
- TTIR `tl.static_range`:
  `STATIC_SPECIALIZED_NO_RUNTIME_CF`; the promoted snapshot contains no runtime
  `scf`/`cf` and is classification evidence only. This is the Phase85 locked
  form of the older `frontend_specialization` status.
- TTIR scalar `scf.if`:
  `SCF_CF`; real Triton emitted a scalar branch returning `tensor<16xf32>`.
- TTIR `scf.while`:
  `SCF_CF`; real Triton emitted `scf.while`, `scf.condition`, `scf.yield`, and
  loop-carried `tensor<16xf32>, i32`.
- TTIR scalar switch/index-switch forms:
  `lowerable_if_canonicalized_to_scalar_cf_switch_or_branch_chain`; raw forms
  must enter through upstream canonicalization or the value `cf.switch`
  scalar-chain policy.
- TTIR vector or per-lane branch conditions:
  deterministic reject as control flow. VC4 has one program counter per QPU
  program; per-lane divergence must be represented as masks/selects, not
  per-lane PC.
- TTIR backend dialect control flow in `ttg`, `triton_gpu`, `nvgpu`, or `nvvm`:
  deterministic reject at the TTIR frontend boundary because those are not the
  accepted source dialects for the VC4 path.
- TTIR unsupported loop bodies:
  stage by exact body feature, such as `tt.dot`, `tt.reduce`, or block pointer
  forms; do not report the control-flow skeleton as the failed feature.
- TTIR loop attrs:
  `tt.loop_unroll_factor` and `tt.flatten` survive in the promoted snapshots.
  `warp_specialize` was not emitted by Phase85a2 and remains unclassified.

These are taxonomy and profile classifications only. They do not implement
TTIR control-flow lowering, and `READY_FOR_TRITON=NO` remains true.

Legacy taxonomy readers may still see this as
`lowerable_if_matches_value_cf_subset`: Phase85b narrows that placeholder to
the exact real TTIR `SCF_CF` forms listed above.

## 10. Reductions

TTIR reductions map to `vector.reduction` only when value reduction legality is
proven.

i32 reductions are the first natural reduction family. f32 reductions require
an explicit finite-tree or approximate policy. Strict IEEE f32 reductions must
not silently lower to the locked VC4 finite-tree reduction contract.

If strict IEEE reduction semantics are required, the target profile must either
provide a slow semantic-preserving emulation plan or diagnose a temporary reject.
No permanent reject is claimed in Phase 1.

## 11. Math and approximate policy

Triton fast math maps to explicit value-level approximate policy. Precise or
default math needs exact lowering, a semantic-preserving emulation path, or a
diagnostic.

The exploratory direction is lower more and reject less, but never silently
approximate precise semantics. Approximate SFU is an explicit target policy
only.

There is no direct SFU sqrt path in the locked VC4Kernel contract. Approximate
sqrt can be considered as an explicit composite under approximate policy.
Exact/default sqrt remains a temporary reject until exact or emulated lowering
is specified.

Exp/log base semantics must be explicit. Target approximate base-2 SFU forms
must not be confused with precise source default math.

## 12. Subword, f16 storage, and unsupported native float formats

i8 and i16 storage should first go through exact mode tables and the static
packed-mode rules from the value-surface and VC4Kernel contracts.

f16 storage conversion plus f32 compute is supportable through the locked
VC4Kernel f16 storage conversion model. It is not native f16 arithmetic.

Native f16 arithmetic is rejected by the locked VC4Kernel surface. Native bf16
and fp8 arithmetic/conversion are also rejected by the locked VC4Kernel surface
unless a future value-level emulation design proves source semantics can be
preserved without changing the target contract.

Dynamic subword selector remains byte/halfword selector inside an accepted
static packed mode. It is not dynamic width, dynamic subword mode, dynamic
orientation, or dynamic layout.

## 13. Dot, contract, and GEMM

`tt.dot` conceptually lowers through the value layer as `vector.contract` or a
canonical equivalent, then into the VC4 tile planner. It must not bypass the
value surface and must not lower directly to VC4Kernel as the first path.

The first planned dot shape is:

```text
BLOCK_M=1
BLOCK_N=16
BLOCK_K=4 initially
```

This is a row-fragment GEMM/GEMV-style shape with small K staged through later
memory/tile planning. The profile assumes no tensor-core behavior and no TF32
semantics. Input, accumulation, storage, mask, and math policy must be explicit.

## 14. Block pointers and tensor descriptors

Block pointers and tensor descriptors are later TTIR profile work. They should
preserve rank, shape, layout, pitch, stride, bounds, and mask semantics in the
value layer before the planner chooses TMU, VDR/VPM, VDW, or a diagnostic.

The first profile should treat block pointer loads and dense rectangular block
stores as supportable later. Sparse block stores remain rejected for direct
VC4Kernel sparse VDW lowering unless future hardware proof or a separate
emulation plan changes the value-level classification.

## 15. Cooperative metadata: num_warps and num_stages

`num_warps` and `num_stages` are planner metadata for VC4. They are not SIMT
semantics in the value layer and must not expose physical QPU identity.

The planner may eventually use these metadata to choose cooperative resource
layout, VPM rows, barriers, semaphores, or double buffering. The value program
still expresses logical work, memory, vectors, masks, and control.

## 16. Atomics, ordering, and synchronization

Atomics are `temporary_reject_not_implemented` pending an emulation or hardware
design. This profile does not permanently reject atomics.

A future atomic design must specify ordering, visibility, masks, element types,
addressing, and interaction with host/runtime support. It must consider whether
slow scalarization, host mediation, semaphore use, or another mechanism can
preserve semantics before any permanent reject is claimed.

Synchronization constructs must be separated into source semantics and planner
metadata. VC4 barriers and semaphores exist in the locked lower target, but the
value layer must not expose hardware barrier ops as initial source operations.

## 17. Cache and eviction hints

Cache and eviction hints are initially temporary profile work. They may be
ignored, diagnosed, or mapped only if the target profile proves that doing so
preserves source semantics.

Some hints may be semantically droppable. Others may encode ordering,
visibility, or performance assumptions that should be diagnosed until the
profile is precise.

## 18. Out-of-scope non-compute hardware

Fixed-function graphics, texture filtering, tile-buffer rendering, display
paths, cube maps, rasterization, and similar graphics features are outside the
VC4 compute-kernel target.

These are out of scope for the TTIR compute profile. They are not evidence for
or against normal compute-kernel value lowering.

## 19. Staged TTIR acceptance sequence

The staged TTIR sequence is:

- Phase 6 inventory/importer skeleton;
- Phase 7 elementwise smoke;
- Phase 8.5 control-flow bridge/support/reject lock;
- Phase 9.5 mask/memory-legality bridge/support/reject lock;
- Phase 10.5 gather/strided memory bridge;
- Phase 12 reductions smoke;
- Phase 14 math smoke;
- Phase 16 subword smoke;
- Phase 20 dot smoke;
- Phase 24 block pointers;
- Phase 25 broader memory;
- Phase 26 ML-relevant corpus;
- Phase 28 profile closure.

Each stage must enter through the standard value surface. `READY_FOR_TRITON`
remains `NO` until the relevant importer and handwritten value path are proven.

## 20. Demo corpus expectations

The demo corpus should consist of real Triton Python kernels that emit the
target-profile TTIR families being claimed. Demos should cover elementwise,
masks, contiguous memory, reductions, math policy, subword/f16 storage, dot,
block pointers, and broader ML-relevant kernels as the staged sequence reaches
them.

Demo credibility requires checked emitted TTIR and checked value-surface output.
It must not depend on decorative source names, fixture paths, public names, or
status strings. Once upper-stack hardware fixtures exist, they must inherit the
mixed fixture saw/no claim discipline from the locked VC4Kernel acceptance
contract.

## 21. Phase 6 real TTIR frontend lock notes

Phase 6 pins the frontend source of truth to real Triton 3.7.0 TTIR. The
checked Python source corpus under `examples/triton/phase6/kernels/` is source
provenance and demo material. The compiler support matrix is based on emitted
TTIR snapshots under `examples/triton/phase6/generated/`, plus parse inventory
from Triton's MLIR parser.

The Phase 7 elementwise candidate forms observed in the required corpus are:

- `tt.func` and `tt.return`;
- axis-0 `tt.get_program_id`;
- `tt.make_range` as the emitted arange/lane-range form;
- `tt.splat`;
- tensor `arith` integer and floating-point ops, comparisons, and select;
- masked `tt.load` tensor-of-pointer forms with `other=0` or `other=0.0`;
- masked `tt.store` tensor-of-pointer forms.

Future staged forms observed or reserved by Phase 6 are:

- `tl.sum` / `tt.reduce` and `tt.reduce.return`;
- `tl.exp` / `math.exp` and other math policy forms;
- `tl.dot` / `tt.dot` as a future `vector.contract` source;
- block pointers and tensor descriptors;
- atomics;
- cache and eviction modifiers;
- volatile memory behavior.

`READY_FOR_TRITON=NO` remains true after Phase 6. Phase 6 proves frontend setup,
real TTIR generation, parse inventory, corpus snapshots, and an importer
skeleton only. Full Triton support requires later semantic TTIR-to-value
lowering and hardware proof.

## 22. Phase 10.7 mask/memory importer profile

Phase 10.7 statically accepts the controlled TTIR mask/memory forms emitted by
real Triton 3.7.0:

- `tt.load` and `tt.store` with no mask as full transfers;
- canonical tail memory masks formed by `offsets < scalar_bound`;
- compute masks from vector comparisons when they feed `arith.select`;
- rank-1 identity pointer arithmetic for `i32`/`f32` `tensor<16x...>` forms.

The importer stages sparse or unknown `tt.load`/`tt.store` memory masks,
nonzero `tt.load` `other`, rank-2/shape-changing memory forms, and block
pointers. The classification is structural and does not use source names,
fixture paths, printed TTIR, or generated-output names.

TTIR_MASK_MEMORY_IMPORTER_STATIC=PASS
TTIR_CANONICAL_TAIL_MASK_LOWERING=YES
TTIR_FULL_NO_MASK_LOWERING=YES
TTIR_COMPUTE_MASK_SELECT_LOWERING=YES
TTIR_SPARSE_MEMORY_MASK_REJECTS=PASS
TTIR_NONZERO_LOAD_OTHER_REJECTS=PASS
FRONTEND_ROBUSTNESS_AUDIT=PASS
READY_FOR_PHASE10_8_TTIR_HARDWARE_ISOLATION=YES
READY_FOR_TRITON=NO

## 23. Phase 10 final mask/memory TTIR profile lock

The final Phase 10 TTIR profile accepts the controlled real Triton mask/memory
forms that lower structurally through the C++ importer into the locked value
surface: no-mask full transfers, canonical `offsets < scalar_bound` tail
masks, compute-mask select, and rank-1 flattened i32/f32 memory. These forms
are hardware-proven through both isolation and mixed TTIR fixtures.

Sparse/unknown TTIR memory masks, nonzero `tt.load` `other`, rank-2/strided/
gather/block-pointer memory, and unrelated dot/reduce/conversion/SFU features
remain staged. The profile does not claim broad Triton readiness.

PHASE10_RESULT=LOCKED
FEATURE=VALUE_AND_TTIR_MASK_CLASSIFIER_MEMORY_LEGALITY
VALUE_MASK_MEMORY_CONTRACT=LOCKED
VALUE_MASK_CLASSIFIER_IMPLEMENTED=YES
VALUE_MEMORY_LEGALITY_CLASSIFIER_IMPLEMENTED=YES
VALUE_MASK_MEMORY_STATIC=PASS
VALUE_MASK_MEMORY_HARDWARE_ISOLATION=PASS
VALUE_MASK_MEMORY_MIXED_ACCEPTANCE=PASS
REAL_TRITON_MASK_MEMORY_SOURCES=YES
REAL_TTIR_MASK_MEMORY_SNAPSHOTS=YES
TTIR_MASK_MEMORY_IMPORTER_STATIC=PASS
TTIR_MASK_MEMORY_HARDWARE_ISOLATION=PASS
TTIR_MASK_MEMORY_MIXED_ACCEPTANCE=PASS
SPARSE_TRANSFER_MASKS_STAGED=YES
NONZERO_LOAD_OTHER_STAGED=YES
RANK2_STRIDED_GATHER_BLOCK_POINTER_STAGED=YES
VALUE_MIXED_REGRESSION=PASS
TTIR_MIXED_REGRESSION=PASS
LAYERED_REGRESSION_POLICY_APPLIED=YES
NO_TEMPORARY_MASK_MEMORY_WORKAROUNDS=YES
READY_FOR_PHASE11_STRIDED_RANKED_MEMORY_SKELETONS=YES
READY_FOR_TRITON=NO

## 24. Phase 11.7 strided memory TTIR importer profile

TTIR_STRIDED_MEMORY_IMPORTER_STATIC=PASS
TTIR_ROW_STRIDED_POINTER_LOWERING=YES
TTIR_LANE_VARYING_STRIDE_GATHER_REJECTS=PASS
TTIR_COLUMN_SLICE_REJECTS=PASS
TTIR_RANK_INFERENCE_FROM_NAMES=NO
FRONTEND_ROBUSTNESS_AUDIT=PASS
READY_FOR_PHASE11_8_TTIR_HARDWARE_ISOLATION=YES
READY_FOR_TRITON=NO

Phase 11.7 accepts the controlled real Triton row-strided memory snapshots in
the C++ importer. TTIR pointer arguments remain flattened rank-1 value memrefs;
the importer lowers structurally recognized pointer offsets of the form scalar
row/column base terms plus one contiguous `tt.make_range(0, 16)` lane vector to
value-layer transfer indices. Canonical tail masks continue to use the
contiguous column/block offset, while row-base terms are part of the memory
address only.

The importer does not infer rank or layout from source names, kernel names,
fixture paths, public argument names, or status strings. Lane-varying
stride/gather, column/vertical slices, block pointers, rank-2 tensor forms,
nonzero load `other`, and sparse or unknown memory masks remain staged.

## 25. Phase 11 final strided memory TTIR profile lock

PHASE11_RESULT=LOCKED
FEATURE=VALUE_AND_TTIR_STRIDED_RANKED_MEMORY_SKELETONS
TTIR_STRIDED_MEMORY_IMPORTER_STATIC=PASS
TTIR_ROW_STRIDED_POINTER_LOWERING=YES
TTIR_STRIDED_MEMORY_HARDWARE_ISOLATION=PASS
TTIR_STRIDED_MEMORY_MIXED_ACCEPTANCE=PASS
TTIR_LANE_VARYING_STRIDE_GATHER_REJECTS=PASS
TTIR_COLUMN_SLICE_REJECTS=PASS
TTIR_RANK_INFERENCE_FROM_NAMES=NO
GATHER_LANE_STRIDE_STAGED=YES
COLUMN_SLICE_STAGED=YES
HIDDEN_MEMREF_DESCRIPTOR_REJECTED=YES
FRONTEND_ROBUSTNESS_AUDIT=PASS
VALUE_MIXED_REGRESSION=PASS
TTIR_MIXED_REGRESSION=PASS
LAYERED_REGRESSION_POLICY_APPLIED=YES
NO_TEMPORARY_STRIDED_MEMORY_WORKAROUNDS=YES
READY_FOR_PHASE12_REDUCTIONS=YES
READY_FOR_TRITON=NO

The final Phase 11 TTIR profile accepts controlled real Triton row-strided
memory snapshots with scalar `row * stride + col_block * 16` address bases plus
one contiguous `tt.make_range(0, 16)` lane vector. The final mixed fixture
combines row-strided memory with elementwise compute, scalar control flow,
multi-axis launch identity, and Phase 10 full/empty/tail masks on hardware.

This profile does not claim gather, column-slice, rank-2 tile, block-pointer,
reduction, dot, GEMV, or GEMM support. It also does not introduce hidden memref
descriptor ABI behavior or broad Triton readiness.

## 26. Phase 12.7 reduction importer profile

TTIR_REDUCTION_IMPORTER_STATIC=PASS
TTIR_TL_SUM_ADD_LOWERING=YES
TTIR_SCALAR_REDUCTION_STORE_LOWERING=YES
TTIR_F32_REDUCTION_FINITE_TREE_POLICY=YES
TTIR_NON_ADD_REDUCTIONS_REJECT=PASS
TTIR_DOT_GEMV_STAGED_FOR_PHASE13=YES
FRONTEND_ROBUSTNESS_AUDIT=PASS
READY_FOR_PHASE12_8_TTIR_HARDWARE_ISOLATION=YES
READY_FOR_TRITON=NO

The Phase 12.7 TTIR profile accepts the controlled real Triton add-reduction
forms emitted as public `tt.call` operations to private helper functions whose
bodies contain `tt.reduce` axis 0. The importer resolves those symbols and
classifies the reducer structurally from operand/result types, reducer block
arguments, `tt.reduce.return`, and the add combiner operation. Accepted f32
reductions attach `vc4value.fp_domain = "finite"` and
`vc4value.reduction_policy = "finite_tree"` to the value function.

Scalar `tt.store` of the reduction result to a scalar rank-1 output pointer is
lowered to value-layer `memref.store`, with scalar masks represented as value
control flow. Rank>1 reductions, non-add reductions, dot/GEMV/GEMM, scans,
atomics, and generalized math remain staged. `READY_FOR_TRITON=NO` remains
locked.

## 27. Phase 12 final reduction TTIR profile lock

PHASE12_RESULT=LOCKED
FEATURE=VALUE_AND_TTIR_REDUCTIONS
VALUE_REDUCTION_CONTRACT=LOCKED
VALUE_REDUCTION_TO_VC4KERNEL_STATIC=PASS
VALUE_VECTOR_REDUCTION_ADD_I32_STATIC=PASS
VALUE_VECTOR_REDUCTION_ADD_F32_FINITE_STATIC=PASS
VALUE_SCALAR_STORE_FOR_REDUCTION_STATIC=PASS
VALUE_REDUCTION_HARDWARE_ISOLATION=PASS
VALUE_REDUCTION_MIXED_ACCEPTANCE=PASS
REAL_TRITON_REDUCTION_SOURCES=YES
REAL_TTIR_REDUCTION_SNAPSHOTS=YES
TTIR_REDUCTION_IMPORTER_STATIC=PASS
TTIR_REDUCTION_HARDWARE_ISOLATION=PASS
TTIR_REDUCTION_MIXED_ACCEPTANCE=PASS
F32_REDUCTION_FINITE_TREE_POLICY=YES
EXACT_F32_REDUCTION_NOT_CLAIMED=YES
NON_ADD_REDUCTIONS_STAGED=YES
RANK_GT_1_REDUCTIONS_STAGED=YES
DOT_GEMV_STAGED_FOR_PHASE13=YES
VALUE_MIXED_REGRESSION=PASS
TTIR_MIXED_REGRESSION=PASS
LAYERED_REGRESSION_POLICY_APPLIED=YES
NO_TEMPORARY_REDUCTION_WORKAROUNDS=YES
READY_FOR_PHASE13_GEMV_ROWWISE_DOT=YES
READY_FOR_TRITON=NO

The final TTIR reduction profile accepts controlled real Triton `tl.sum`
snapshots that lower to `tt.reduce` add over the sole vector dimension and
store the scalar result through a rank-1 output pointer. The importer
classifies reducer bodies structurally from SSA values, region operations,
typed operands/results, and exact attributes. It does not parse printed TTIR or
special-case fixture names, paths, kernel names, source variable names, or
public argument names.

The final Phase 12 TTIR mixed fixture combines the reduction bridge with
elementwise operations, scalar control flow, multi-axis launch identity, tail
masks, row-strided memory, scalar stores, and strict hardware checks. Non-add
reducers, rank>1 reducers, dot/GEMV/GEMM, scans, atomics, and exact/default
f32 reductions remain staged.

## 28. Accelerated Phase 13 GEMV Demo Static Profile

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

The accelerated Phase 13 demo profile accepts only the exact source-controlled
GEMV-v0 snapshot shape:

```text
A + row * LDA + arange(0, 16)
X + arange(0, 16)
mask = arange(0, 16) < K
tl.sum(load(A) * load(X), axis=0)
store Y[row]
```

This static profile is implemented through structural composition of the
Phase 11 row-strided pointer planner, Phase 12 add-reduction importer, and a
local pointer-splat planning repair for scalar pointer expressions. It does
not infer roles from source names or paths, does not parse printed TTIR, and
does not emit lower-half IR from the importer.

`tl.dot`, `tt.dot`, `vector.contract`, multi-block K accumulation, atomics,
f16/casts, SFU/math, and full Phase 13 GEMV support remain staged.

## 29. Phase 13.7 GEMV row-wise dot importer profile

TTIR_GEMV_ROWWISE_DOT_IMPORTER_STATIC=PASS
TTIR_GEMV_TL_SUM_PRODUCT_LOWERING=YES
TTIR_GEMV_SCALAR_RESULT_STORE_LOWERING=YES
TTIR_GEMV_F32_FINITE_TREE_POLICY=YES
TTIR_TL_DOT_TT_DOT_REJECT=PASS
TTIR_MULTIBLOCK_K_ACCUMULATION_STAGED=YES
FRONTEND_ROBUSTNESS_AUDIT=PASS
READY_FOR_PHASE13_8_TTIR_HARDWARE_ISOLATION=YES
READY_FOR_TRITON=NO

The Phase 13.7 TTIR profile accepts controlled real Triton GEMV-v0 row-wise
dot snapshots emitted as `tl.sum(a * x, axis=0)`. The importer lowers the
product and add-reduction structurally to the standard value layer, preserves
Phase 10 inactive-zero tail behavior, Phase 11 row-strided memory, and Phase 12
scalar reduction stores, and attaches the finite f32 reduction policy required
by the value surface.

The importer stages real `tl.dot` / `tt.dot` forms and loop-carried
multi-block K accumulation. It does not add vector.contract, full GEMM,
atomics, casts, f16/subword, SFU/math, lower-half emission, or a broad Triton
readiness claim.
