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

Arange-like TTIR forms lower conceptually through the value lane model:

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
