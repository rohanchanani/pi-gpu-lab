# VC4 Value-to-VC4Kernel Planning Guide

PHASE10_VALUE_MASK_CLASSIFIER_CONTRACT=LOCKED
PHASE10_VALUE_MEMORY_LEGALITY_CONTRACT=LOCKED
SPARSE_MEMORY_MASKS_STAGED=YES
NONZERO_LOAD_OTHER_STAGED=YES
RANK2_STRIDED_MEMORY_STAGED=YES
READY_FOR_PHASE10_4_VALUE_MASK_MEMORY_CLASSIFIER_STATIC=YES
READY_FOR_TRITON=NO

## 1. Purpose and non-goals

This document specifies how later lowering phases should plan standard
value-surface patterns into the locked VC4Kernel target surface.

It is not an implementation. It does not add accepted value features, does not
add a `vc4value` dialect, does not add a verifier, does not add conversion
passes, does not add TTIR import, does not run hardware, and does not change the
locked VC4Kernel surface.

Phase 5 implementation status:

```text
VC4_VALUE_TO_VC4KERNEL_PASS_SKELETON_PRESENT=YES
PASS_FLAG=--convert-vc4-value-to-vc4kernel
VC4_VALUE_TO_VC4KERNEL_STATIC_ELEMENTWISE_LOWERING=YES
VC4_VALUE_TRANSFER_READ_TMU_STATIC=YES
VC4_VALUE_TRANSFER_WRITE_VDW_STATIC=YES
READY_FOR_HARDWARE_PHASE5G=YES
READY_FOR_TRITON=NO
```

Phase 5g hardware note:

```text
PHASE5G_VALUE_COPY_F32_TAIL_HARDWARE_SMOKE=YES
PHASE5G_ACTIVE_QPUS_1_ONLY=YES
PHASE5G_ORIGINAL_ACTIVE_QPUS_12_FIXTURE_EXPOSED_OVERLAUNCH_TAIL_MASK_GAP=YES
PHASE5G_NARROWED_TO_ONE_LOGICAL_VECTOR16_REQUEST_PER_BLOCK=YES
PHASE5G_NOT_SUFFICIENT_FOR_FINAL_PHASE5_TAIL_MEMORY_COVERAGE=YES
PHASE5H_OR_PHASE5J_REQUIRES_MULTI_REQUEST_12_ACTIVE_QPU_VALUE_HARDWARE=YES
```

The original Phase 5g `active_qpus=12` copy/tail fixture exposed a real
overlaunch/tail-mask gap: logical requests beyond `ceil(n / 16)` could observe
negative remaining element counts and write full inactive chunks. The committed
Phase 5g fixture was intentionally narrowed to one logical `vector<16>` request
per block only to prove the initial value-to-hardware path. It must not be
treated as complete Phase 5 tail/memory coverage.

Phase 5 final hardware lock status:

```text
VC4_VALUE_TO_VC4KERNEL_PHASE5_ELEMENTWISE_LOCKED=YES
VC4_VALUE_TO_VC4KERNEL_PASS_ACCEPTED=YES
VC4_VALUE_ELEMENTWISE_VECTOR16_COMPUTE_ACCEPTED_HARDWARE=YES
VC4_VALUE_TRANSFER_READ_TMU_ACCEPTED_HARDWARE=YES
VC4_VALUE_TRANSFER_WRITE_VDW_PRESERVE_ACCEPTED_HARDWARE=YES
VC4_VALUE_ELEMENTWISE_MIXED_ACCEPTANCE_ACCEPTED=YES
READY_FOR_PHASE6_REAL_TTIR_INVENTORY_AND_IMPORTER_SKELETON=YES
READY_FOR_TRITON=NO
```

Phase 5 locks only the handwritten V1 elementwise lowering subset:
rank-1 contiguous i32/f32 `#vc4value.global` transfer reads and writes,
`vector<16>` i32/f32 compute, finite-policy f32 compare/select, i32 compare,
tail masks, TMU inactive-zero safe-offset loads, and VDW inactive-preserve
stores. Non-16 vectors, rank-2 vectors and memrefs, subword/f16 memory paths,
reductions, contracts, gather/scatter, SFU/math lowering, control-flow
lowering, and Triton/TTIR ingestion remain staged and are not made value-surface
rejects by this lock.

The guide is a planning contract for future phases. It describes the decisions
those phases must make before creating VC4Kernel IR from:

```text
func + tiny vc4value + vector + memref + arith + math + scf/cf
```

Feature classification remains in
`compiler/docs/vc4_value_ttir_feature_taxonomy.json`.
The Phase 3.5 vector abstraction boundary is recorded in
`compiler/docs/vc4_value_surface_abstraction_policy.md`.
The Phase 4 public value ABI is recorded in
`compiler/docs/vc4_value_kernel_abi.md`.
Executable value-lowering hardware proof requirements are recorded in
`compiler/docs/vc4_value_hardware_verification_policy.md`.

## 2. Planning boundary

The value surface expresses standard semantics: logical functions, memrefs,
vectors, masks, scalar arithmetic, math, and control flow.

The planner chooses target mechanisms such as TMU, VDR, VPM, and VDW. Those are
not value-surface operations. They are VC4Kernel planning outputs when the value
pattern is legal.

Verified VC4Kernel output may contain only accepted VC4Kernel operations,
attributes, and carrier types from the locked Surface v2 contract. Producer
dialect operations must not appear inside verified VC4Kernel.

VC4Kernel lowers only through SSAVC4. There is no direct VC4KernelToVC4 path.
VC4Tile is retired and must not be revived as a value-layer or planner
intermediate.

## 3. Launch identity planning

Phase 9 multi-axis launch identity contract:

```text
PHASE9_FEATURE=VALUE_AND_TTIR_MULTI_AXIS_LAUNCH_IDENTITY
VALUE_MULTI_AXIS_SURFACE_CONTRACT=LOCKED
VC4VALUE_GRID_RANK_1_2_3_EXECUTABLE_FOR_LAUNCH_IDENTITY=YES
VC4VALUE_PROGRAM_ID_AXES_0_1_2_LOWERABLE_NOW=YES
VC4VALUE_NUM_PROGRAMS_AXES_0_1_2_LOWERABLE_NOW=YES
PHASE9_MEMORY_MODEL=FLATTENED_RANK1_ONLY
PHASE9_MASK_MODEL=CANONICAL_LINEARIZED_TAIL_ONLY
MASK_CLASSIFIER_SCOPE_CREEP=NO
RANK2_MEMORY_SCOPE_CREEP=NO
READY_FOR_TRITON=NO
```

`vc4value.program_id {axis = 0|1|2}` maps to the VC4Kernel program-id operation
for the same logical axis.

`vc4value.num_programs {axis = 0|1|2}` maps to VC4Kernel num-programs behavior
and associated resource/launch metadata.

Axes are logical launch-grid axes only. They are not physical QPU IDs and must
not expose physical scheduling to the value layer.

For Phase 9, `vc4value.grid_rank` values 1, 2, and 3 are executable for launch
identity. The logical mapping from a flattened launch request to grid
coordinates is:

```text
pid0 = logical_block_id % grid.x
pid1 = (logical_block_id / grid.x) % grid.y
pid2 = logical_block_id / (grid.x * grid.y)
```

`vc4value.num_programs(axis)` returns `grid.x`, `grid.y`, or `grid.z` for axes
0, 1, and 2 respectively.

Phase 9 multi-axis launch identity does not change the memory model. Memory
remains flattened rank-1 only. Multi-axis kernels may compute a linearized
rank-1 element index from logical grid coordinates, but rank-2 memory,
rank-2 transfer planning, gather/scatter, block pointers, tensor descriptors,
and VPM tile planning remain staged.

Phase 9 also does not add a general mask classifier. The only accepted dynamic
mask form for this feature is the canonical linearized tail:

```text
idx = flattened_block_id * 16 + lane
mask = idx < n
```

Index values may remain `index` in the value surface, but the planner must prove
the target i32 conversion policy before constructing VC4Kernel scalar i32
operands.

## 4. ABI and memref lowering plan

A value memref base pointer lowers later to a raw i32 uniform for VC4Kernel.
Dynamic sizes, strides, offsets, and other ABI values lower to scalar arguments
or metadata after the value-layer ABI plan proves their type and range.

Phase 4 locks the producer-facing public memref ABI as rank-1/rank-2
`#vc4value.global` memrefs with element types:

```text
i8, i16, i32, f16, f32
```

Dynamic dimensions are named by `vc4value.shape_args`; dynamic strides are
named by `vc4value.stride_args`. There is no hidden memref descriptor ABI.
`memref.dim` is metadata-only and must resolve to explicit extent arguments or
static dimensions before target lowering.

The first Phase 5 V1 lowerable memory subset is narrower: rank-1 contiguous
i32/f32 global memrefs for elementwise kernels. i8/i16/f16 memrefs, rank-2
memrefs, and dynamic-stride layouts remain staged until the planner implements
the relevant subword, f16-storage, or VPM/tile paths.

Read-only, write-only, and inout attributes are planning hints. They can guide
TMU, VDR, and VDW path selection, but they are not permission to weaken source
memory semantics or skip required inactive-lane handling.

## 5. Lane and fragment planning

`vector.step` maps to VC4Kernel lane identity / `lane_range` equivalent in the
Phase 5 V1 subset when the vector shape is `vector<16xindex>` or can be safely
converted to the target i32 lane carrier.

`vector<16xi32>` and `vector<16xf32>` are the first lowerable VC4Kernel
fragment carrier types for executable arithmetic. `vector<16xi1>` and
`vector<16xindex>` are support forms for masks and address/program-id
arithmetic. `vector<16xT>` is not the global value-layer type limit.

Fixed rank-1 `vector<NxT>` with `N != 16` is surface-admissible and staged for
later splitting into `vector<16>` fragments plus tails or loops. Fixed rank-2
`vector<MxNxT>` is surface-admissible and staged for later tile, contract, and
VPM planning. The value planner must not invent wider VC4Kernel fragment
carriers.

`vector.splat` and constants map to VC4Kernel `splat` and `fragment_const` where
the element type and encoding are legal. Constants that are not directly
encodable must be decomposed into accepted operations or diagnosed.

## 6. Mask classifier

The planner must classify masks before memory lowering:

- full: all lanes active;
- empty: no lanes active;
- tail: active prefix;
- rect: dense rectangular active region;
- sparse: arbitrary lane pattern;
- unknown: not proven to be one of the accepted forms.

Required decisions:

- full -> accepted for compute, load, and store;
- empty -> no-op/store skip or identity path;
- tail -> accepted for TMU safe load and VDW preserve store;
- rect -> accepted for tile/VDW rect where representable;
- sparse -> compute predicate may be legal, but store rejects unless future
  hardware-proven support exists;
- unknown -> reject for stores, and reject or require canonicalization for
  loads depending on the selected path.

Sparse and unknown stores must not silently lower to sparse VDW stores.

Phase 10 refines this into the locked value mask classifier contract:

- `FULL`: no transfer mask, or `vector.create_mask` active count at least 16
  after clamping.
- `EMPTY`: `vector.create_mask` active count at most 0 after clamping.
- `TAIL_0_TO_16`: `vector.create_mask %count : vector<16xi1>`, clamped to
  `[0, 16]`, with active lanes starting at lane 0.
- `COMPUTE_MASK`: `vector<16xi1>` compare/logical masks used only by compute
  operations such as `arith.select`. These are accepted as compute masks, not
  as memory transfer predicates.
- `SPARSE_OR_UNKNOWN_MEMORY_MASK`: any transfer mask not classified as full,
  empty, or tail. Stores deterministically stage/reject; loads remain staged
  unless a later phase implements exact support.
- `RECT`: staged until rank-2/tile memory planning.

The classifier must be structural: typed operation classes, exact operation
names, attributes, and SSA use-defs decide semantics. It must not parse printed
IR or special-case fixture names, paths, public names, or generated-output
locations.

## 6.1 Phase 10 memory legality

The accepted Phase 10 value memory legality subset is:

- rank-1 memrefs in `#vc4value.global`;
- identity layout;
- element type `i32` or `f32`;
- `vector<16xi32>` or `vector<16xf32>` transfers;
- rank-1 identity transfer permutation maps;
- scalar index base;
- load padding/`other` exactly zero;
- transfer-read inactive lanes through safe-offset inactive-zero;
- transfer-write inactive lanes through inactive preserve.

The staged or rejected Phase 10 memory forms are:

- nonzero load padding/`other`;
- non-identity transfer maps;
- rank greater than 1 memrefs;
- strided layouts;
- tensor/block pointer memory;
- gather/scatter;
- sparse or unknown store masks;
- subword or f16 memory;
- boundary-check/padding-option producer semantics;
- vector rank greater than 1.

Phase 10.3 is a contract lock only. Planned Phase 10.4 static coverage must
cover full/unmasked transfers, empty masks, clamped tails, compute-mask selects
that are not memory masks, sparse transfer read/write staging, nonzero `other`
staging, and non-identity map staging before any hardware claim.

## 7. Elementwise arithmetic/cmp/select planning

Value `arith` and `vector` elementwise operations map to VC4Kernel fragment ALU
where opcode, type, and policy are accepted.

i32 add, sub, mul, bitwise operations, shifts, min/max, comparisons, constants,
splats, and selects map through the accepted fragment ALU, compare, const,
splat, and select families.

f32 arithmetic maps only for the accepted basic target arithmetic set. f32
comparisons require explicit finite policy before lowering to VC4Kernel finite
f32 compare operations.

Unsupported element type, unsupported opcode, missing finite policy, or an
unclassified mask must produce a deterministic diagnostic before VC4Kernel IR is
constructed.

## 8. Transfer-read planning

Transfer-read planning chooses among several target paths:

- simple contiguous/tail low-reuse load -> TMU safe-offset inactive-zero;
- structured rank-2, tile, or reuse load -> VDR -> VPM -> QPU VPM read;
- gather load later -> TMU direct address;
- unsupported or unclassified path -> diagnostic.

The value layer does not name TMU, VDR, VPM, or VDW. These are planner choices.

Inactive value, pad, or `other` handling must preserve source semantics. For
the initial contiguous/tail profile, inactive lanes can use the locked
VC4Kernel TMU safe-offset inactive-zero contract only when the value operation
requires zero or when the planner can legally materialize the requested
inactive value through a subsequent value operation.

No old TMU signature or inferred safe address is allowed. The planner must
provide explicit safe-offset behavior for inactive lanes.

## 9. Transfer-write planning

Transfer-write planning chooses among:

- contiguous/tail store -> VDW register-fragment inactive preserve;
- rect/tile store -> VPM/VDW path;
- sparse store -> reject;
- unknown mask -> reject until classified or emulated by a specified future
  path.

Sparse stores cannot lower to sparse VDW stores. Sparse VDW stores remain a
locked deterministic reject unless a future hardware-proven phase changes the
VC4Kernel surface.

The planner must not use branchy kernel reshaping to avoid compiler bugs. It
may use correct control-flow or masking plans, but those plans must preserve
source semantics and lower through accepted VC4Kernel forms.

## 10. Gather/scatter planning

Gather loads are supportable through future TMU planning when address formation,
coalescing assumptions, inactive-lane safety, and mask legality are specified.
They are not a Phase 1 implementation claim.

Scatter stores remain a reject/emulation candidate. They cannot be lowered to
sparse VDW. A future value-level emulation plan must prove source semantics and
must not alter the locked direct sparse VDW reject.

## 11. VPM/VDR/VDW tile planning

Value IR remains `memref`, `vector`, and `scf/cf`. There is no tile DSL and no
source-visible VPM, VDR, or VDW operation.

The planner may recognize structured patterns and emit accepted VC4Kernel
operations for:

- VPM allocation;
- QPU VPM read/write;
- VDR global-to-VPM movement;
- VDW VPM/register-fragment stores;
- barriers and semaphores where the cooperative contract requires them;
- resource metadata.

Structured tile planning must preserve memref shape, stride, layout, mask, and
element policy until a legal VC4Kernel path is selected.

## 12. P12 coordinate and selector model

P12 row, word-X, and subword selector are separate fields.

Dynamic subword selector is byte/halfword selector only. It does not mean
dynamic width, dynamic subword mode, dynamic orientation, or dynamic layout.

Setup field masks are not modulo semantics. They isolate VC4 hardware fields
and do not legalize out-of-range runtime values.

VPM QPU read/write normalized readback differs from raw DMA carrier placement.
The planner must not conflate QPU-normalized subword readback with VDR or VDW
carrier placement.

VDR and VDW selector semantics are separately proven. Never infer VDW behavior
from VDR symmetry. VDW has its own setup composition and preserve-store
constraints.

Vector or lane-varying coordinate operands are rejected for VC4Kernel dynamic
coordinate paths.

## 13. Subword and f16 storage planning

i8 and i16 storage paths must use exact mode-table entries in future
implementation. Each accepted subword path must cite its width, packing mode,
orientation, selector range, extension/truncation policy, and memory path.

f16 storage conversion plus f32 compute is the accepted f16 model. The planner
may lower f16 storage through fragment pack/unpack and f32 compute carriers
when the value policy permits it.

Native f16 arithmetic is rejected. Native bf16/fp8 arithmetic and conversion
are rejected by the locked VC4Kernel surface. Future value-level emulation
would require a separate proof and taxonomy update.

Dynamic selector support does not extend to dynamic width, dynamic mode,
dynamic orientation, or dynamic layout.

## 14. Reduction planning

`vector.reduction` maps to `vc4kernel.fragment_reduce` only after type, kind,
mask, identity, and math policy are proven.

i32 reductions are exact for the accepted integer reduction kinds.

f32 reductions require finite-tree or explicit approximate policy only. Strict
IEEE f32 reductions must not silently lower to finite-tree VC4Kernel reduction.
They require an exact/emulated plan or a diagnostic.

Masked reductions must specify inactive lane identity behavior before lowering.

## 15. Math and SFU planning

`math.exp`, `math.log`, reciprocal, and rsqrt-like value patterns may map to
approximate SFU only under explicit approximate policy.

Exact/default math must either lower through an exact sequence or reject with a
diagnostic. It must not silently lower to approximate SFU.

There is no direct SFU sqrt. Approximate sqrt may be planned only as an explicit
approximate composite, such as rsqrt plus arithmetic, when source policy allows
it.

Exp/log base semantics must be explicit. Base-2 target SFU forms and source
default exp/log forms are not interchangeable without a specified conversion
and policy.

## 16. Shuffle, rotate, and lane-broadcast planning

Rotate patterns map to VC4Kernel dynamic rotate only when the amount is scalar
and the modulo-16 lane semantics match the locked target rule.

Lane broadcast uses the canonical composite:

```text
lane_range -> compare lane == selected_lane -> select value or zero -> fragment_reduce add
```

For f32 bit-preserving broadcast, bitcast f32->i32, use the i32 composite, then
bitcast i32->f32.

Arbitrary direct shuffle/permutation is rejected at VC4Kernel. Value-level
emulation is future supportable, but it must be a specified composite or
emulation path, not a direct VC4Kernel arbitrary permutation.

## 17. Control-flow planning

Phase 8 is a value-layer control-flow phase, not TTIR control-flow import. The
executable value-to-VC4Kernel control boundary consumes `cf`, not raw `scf`.

PHASE8R_CF_COMPLETENESS_SCOPE=ACTIVE
SCF_WHILE_VALUE_TARGET=SUPPORT_NOW
NESTED_STRUCTURED_CF_VALUE_TARGET=SUPPORT_NOW
TL_RANGE_STYLE_LOOP_SKELETON_VALUE_TARGET=SUPPORT_NOW
PERSISTENT_LOOP_SKELETON_VALUE_TARGET=SUPPORT_NOW
VECTOR_BRANCH_CONDITION_POLICY=DETERMINISTIC_REJECT_AS_CFG_USE_MASKS
IRREDUCIBLE_CFG_POLICY=PROBE_NOT_REQUIRED_FOR_SANE_TRITON
READY_FOR_TRITON=NO

`scf` may appear in value input as a source convenience. Raw `scf` must not
survive into verified VC4Kernel. Before executable value-to-VC4Kernel lowering,
`scf.if`, `scf.for`, and `scf.while` must canonicalize through the upstream MLIR
`--convert-scf-to-cf` pass into `cf` branches and block arguments. The planner
must not grow a custom SCF lowering unless the upstream pass is proven unusable
and that blocker is recorded for user review.

QPU control flow is scalar/coherent across SIMD lanes. Per-lane control
divergence is not accepted as branch CFG; it is represented by masks/selects.
Nested structured control flow, tl.range-style loop skeletons, and
persistent-loop skeletons are value-layer targets before TTIR import when their
bodies use supported value features.

The canonical Phase 8 executable pipeline for structured value control flow is:

```text
value input with scf
  -> --convert-scf-to-cf
  -> --vc4-verify-value-surface
  -> --convert-vc4-value-to-vc4kernel
  -> --verify-vc4kernel
```

Pure `cf` inputs may enter `--vc4-verify-value-surface` directly. VC4Value
hardware/static runners that encounter `scf` input must preserve the
after-scf-to-cf intermediate IR before constructing VC4Kernel.

The Phase 8 V1 executable `cf` set is:

```text
cf.br
cf.cond_br with scalar i1 condition
func.return
```

Value block arguments are staged by type. The Phase 8 V1 candidate lowerable
set is exactly:

```text
index
i1
i32
f32
vector<16xi1>
vector<16xindex>
vector<16xi32>
vector<16xf32>
```

`index` and `vector<16xindex>` must lower to target i32 carriers before
verified VC4Kernel. `vector<16xi1>` masks must lower to accepted VC4Kernel
predicate or scalar condition forms. Memref block arguments, VPM tile block
arguments, vector-valued branch conditions, `cf.switch`, `scf.index_switch`,
`scf.parallel`, `scf.forall`, `scf.reduce`, irreducible CFG, and
exception-like control flow remain rejected or staged in Phase 8R. Irreducible
CFG is probe/classify only and is not required for sane Triton Phase 8.5
control-flow import.

Natural loops, block arguments, liveness, spills, and branch layout must be
preserved honestly. The planner must not reshape kernels to hide backend bugs.

Verified VC4Kernel output contains no producer dialect operations. It may
contain accepted VC4Kernel operations, accepted scalar `arith`, and
`cf.br`/scalar-i1 `cf.cond_br`; it must contain no raw `scf`, `vector`,
`memref`, `func`, `vc4value`, TTIR, `ssavc4`, or scheduled `vc4` operations.

## 18. Contract/GEMM planning

The first contract/GEMM planning shape is:

```text
C[pid_m, pid_n*16 + lane] += A[pid_m,k] * B[k,pid_n*16+lane]
BLOCK_M=1
BLOCK_N=16
BLOCK_K=4 first
```

`vector.contract` is the generic value input. There is no `vc4value.dot` and no
tile DSL.

The planner should lower the first row-fragment shape through vector fragments,
f32/i32 policy as appropriate, reductions, and later VPM/VDR/VDW tile planning.
No tensor-core, TF32, native f16 arithmetic, or direct TTIR-to-VC4Kernel path is
assumed.

## 19. Cooperative and resource planning

`num_warps` and `num_stages` are planner metadata. They are not SIMT source
semantics in the value layer and do not expose physical QPU identity.

Later internal VC4Kernel cooperative/resource planning may use metadata to
choose barriers, semaphores, VPM rows, staging rows, double buffering, and
runtime descriptors.

Resource metadata must remain internally consistent through VC4Kernel, SSAVC4,
scheduled VC4, artifact emission, generated C, manifests, and runtime
descriptors.

## 20. Fixture claim and verification discipline

Future value/Triton hardware fixtures must inherit the mixed fixture claim
discipline from locked VC4Kernel acceptance.

Any `saw_*`, `no_*`, or equivalent claim must be classified as one of:

```text
CHECKED_OUTPUT
CHECKED_AUDIT
PHASE_GUARD
```

Claims must be tied to checked output, checked audit evidence, or an explicit
phase guard. They must not rely on fixture names, paths, public names, status
strings, decorative operations, or generated-output special cases.

Every accepted value feature must eventually have verifier, conversion,
lower-half, hardware where required, mixed-suite, and claim-contract coverage
appropriate to its risk.

## 21. Phase-by-phase implementation hooks

Future implementation hooks:

- Phase 5 elementwise: fragment constants, splats, ALU, comparisons, and
  select;
- Phase 9 multi-axis launch identity: `vc4value.grid_rank` 1/2/3,
  `program_id` axes 0/1/2, `num_programs` axes 0/1/2, flattened rank-1 memory
  only, and canonical linearized tail masks only;
- Phase 10 masks/memory legality: mask classifier, transfer read/write
  legality, TMU safe-offset loads, and VDW preserve stores;
- later gather/strided: gather-load and affine strided memory planning;
- Phase 11 reductions: i32 exact and f32 finite-tree/approx reduction plans;
- Phase 13 math: explicit approximate SFU and exact/default diagnostics;
- Phase 15 subword/f16: exact mode-table subword paths and f16 storage
  conversion plus f32 compute;
- Phase 17 shuffle/broadcast: rotate, lane broadcast composite, and direct
  arbitrary permutation rejects;
- Phase 18 VPM tile: VPM allocation, VDR, VDW, dynamic coordinates/selectors,
  and resource metadata;
- Phase 19 contract: row-fragment GEMM/GEMV planning through `vector.contract`.

Each phase must preserve the value-surface boundary, the locked VC4Kernel
surface, and the SSAVC4 lower-half boundary.

## 22. Phase 6 TTIR-to-value planning handoff

Phase 6 pins real TTIR source to Triton 3.7.0. The compiler support matrix for
Triton is based on emitted TTIR snapshots, not on Python-level metaprogramming.
The Triton Python examples under `examples/triton/phase6/kernels/` are demo
source provenance; the generated `.ttir.mlir` files are the source of truth for
future importer support.

The Phase 7 elementwise candidate TTIR forms map conceptually into the existing
value-layer planning boundary:

- `tt.func` / `tt.return` -> `func.func` / return boundary plus value kernel
  ABI metadata;
- axis-0 `tt.get_program_id` -> `vc4value.program_id`;
- `tt.make_range` / arange -> lane range, `vector.step`, splat, and offset
  arithmetic;
- `tt.splat` -> `vector.splat`;
- tensor `arith` ops -> value `arith` and vector elementwise operations;
- masked `tt.load` with zero `other` -> contiguous/tail
  `vector.transfer_read` planning;
- masked `tt.store` -> contiguous/tail `vector.transfer_write` planning.

Future staged TTIR forms remain outside Phase 7 V1 lowering:

- `tl.sum` / reductions -> later `vector.reduction` planning;
- `tl.exp` and other math -> later explicit math/SFU policy;
- `tl.dot` -> later `vector.contract` planning;
- block pointers and tensor descriptors -> later memory descriptor planning;
- atomics, cache/eviction modifiers, and volatile -> initial rejects or future
  profile work until semantics are specified.

The path remains TTIR -> standard value layer -> VC4Kernel -> SSAVC4 ->
scheduled VC4. Phase 6 does not implement TTIR-to-value semantic lowering, and
`READY_FOR_TRITON=NO` remains true until later phases add and prove the importer
and hardware path.

## 23. Phase 7 TTIR elementwise hardware handoff

Phase 7 proves that the real Phase 6 elementwise TTIR corpus can enter the
existing Phase 5 value planner without bypassing it. Phase 7.5 locks the
accepted semantic importer as the optional C++ `vc4-triton-opt` path:

```text
real emitted TTIR
  -> vc4-triton-opt --convert-triton-to-vc4-value
  -> func/vc4value/vector/memref/arith value IR
  -> --convert-vc4-value-to-vc4kernel
  -> vc4kernel
  -> ssavc4
  -> scheduled vc4
  -> generated artifacts/runtime
  -> real VC4 hardware
```

The supported examples are:

- `vector_add_b16`;
- `saxpy_select_b16`;
- `i32_add_select_b16`.

The TTIR importer output contract remains value-only. Generated value IR may
contain `func`, `vc4value`, `vector`, `memref`, and `arith`, but not `tt`,
`ttg`, `gpu`, `vc4kernel`, `ssavc4`, or scheduled `vc4` operations. Verified
`vc4kernel` output must contain no producer dialect operations.

Phase 7 hardware covers contiguous rank-1 i32/f32 memory, zero-other masked
loads, canonical tail stores, vector<16> arithmetic, compare/select, overlaunch
tail masks, TMU load planning, and VDW inactive-preserve stores at
active_qpus=12 and lanes=16. The mixed TTIR acceptance suite provides the
cumulative Phase 7 hardware gate and claim audit.

Future value planning remains responsible for reductions, dot/contract,
gather/scatter, block pointers, subword/f16 storage, math/SFU policy, non-16
block splitting, axes 1/2, and control flow. These are staged planning gaps
unless a later proof explicitly classifies a specific form as impossible.

Phase 8 does not add TTIR control-flow import. TTIR control flow remains
staged_future_ttir_import, and `READY_FOR_TRITON=NO` remains true.
READY_FOR_TRITON remains NO.

## 24. Phase 10.4 value mask/memory classifier

Phase 10.4 implements the value-layer executable classifier for the locked
Phase 10 mask and memory legality contract. Transfer masks now pass through a
central structural classifier before either `vector.transfer_read` or
`vector.transfer_write` can lower.

Accepted transfer masks are:

- full: no transfer mask, or clean all-active canonical mask;
- empty: clean zero-active canonical mask;
- tail: `vector.create_mask` for `vector<16xi1>` with the active count clamped
  to `[0, 16]` and a lane-zero start.

Compute masks produced by vector comparisons remain valid for
`arith.select`/fragment select, but they are not accepted as memory transfer
predicates. Sparse or unknown transfer masks reject deterministically in the
value-to-VC4Kernel pass.

The central memory legality classifier accepts only rank-1 identity
`#vc4value.global` memrefs with `i32` or `f32` elements, `vector<16xi32>` or
`vector<16xf32>` transfer values, scalar base indices, rank-1 identity transfer
maps, zero `transfer_read` padding, inactive-zero TMU loads, and
inactive-preserve VDW stores. Rank-2, strided, sparse, gather/scatter,
block-pointer, nonzero-other, subword/f16, and non-identity map forms remain
staged.

VALUE_MASK_CLASSIFIER_IMPLEMENTED=YES
VALUE_MEMORY_LEGALITY_CLASSIFIER_IMPLEMENTED=YES
SPARSE_TRANSFER_WRITE_REJECTS=PASS
SPARSE_TRANSFER_READ_REJECTS=PASS
NONZERO_LOAD_OTHER_REJECTS=PASS
VALUE_MASK_MEMORY_STATIC=PASS
READY_FOR_PHASE10_5_VALUE_HARDWARE_ISOLATION=YES
READY_FOR_TRITON=NO

## 25. Phase 10.5 value mask/memory hardware isolation

Phase 10.5 proves the Phase 10.4 value mask classifier and memory legality
subset on real VC4 hardware with targeted isolation fixtures. The fixtures run
with `active_qpus=12`, strict CPU oracles, sentinel preservation checks,
expected JSON result checking, and the existing VC4Value hardware runner.

The hardware-proven subset covers full, empty, and clamped tail transfer masks,
compute compare/select masks that do not feed memory predicates, zero-other
masked loads with inactive-zero behavior, TMU safe-offset policy, and
inactive-preserve stores. Sparse transfer masks remain static rejects.

VALUE_MASK_MEMORY_HARDWARE_ISOLATION=PASS
VALUE_MASK_FULL_HARDWARE=PASS
VALUE_MASK_EMPTY_HARDWARE=PASS
VALUE_MASK_TAIL_HARDWARE=PASS
VALUE_COMPUTE_MASK_SELECT_HARDWARE=PASS
VALUE_INACTIVE_ZERO_LOAD_HARDWARE=PASS
VALUE_INACTIVE_PRESERVE_STORE_HARDWARE=PASS
READY_FOR_PHASE10_6_VALUE_MIXED_ACCEPTANCE=YES
READY_FOR_TRITON=NO

## 26. Phase 10.6 value mask/memory mixed acceptance

Phase 10.6 extends the cumulative VC4Value mixed hardware suite with
`mixed_value_mask_memory_axis_cf_vc4value`. The fixture keeps memory rank-1 and
flattened while combining Phase 5 i32/f32 elementwise behavior, Phase 8 scalar
control flow, Phase 9 multi-axis launch identity, and Phase 10 mask/memory
legality in one natural kernel.

The mixed fixture checks full, empty, and tail transfer masks, compute-mask
select that does not feed memory predicates, inactive-zero load behavior,
inactive-preserve store behavior, repeated launches including `n=0`, strict CPU
oracles, sentinels, and `active_qpus=12`. The full VC4Value mixed suite passes
with zero mismatches, zero sentinel mismatches, and zero launch failures.

VALUE_MASK_MEMORY_MIXED_ACCEPTANCE=PASS
VALUE_MIXED_REGRESSION=PASS
READY_FOR_PHASE10_7_TTIR_IMPORTER_MASK_MEMORY_STATIC=YES
READY_FOR_TRITON=NO

## 27. Phase 10 final mask/memory legality lock

Phase 10 final-lock keeps the value-to-VC4Kernel contract unchanged from the
Phase 10.4 classifier implementation and records that the full value/TTIR
vertical slice has passed static checks, value hardware isolation, value mixed
acceptance, TTIR importer static lowering, TTIR hardware isolation, and TTIR
mixed acceptance.

The central value mask classifier accepts full, empty, and lane-zero clamped
tail masks for memory transfers. Compute masks remain compute-only. Sparse or
unknown transfer masks remain staged/rejected. For the Phase 10 executable
memory-transfer subset, the central memory legality classifier accepts rank-1
identity global i32/f32 `vector<16>` transfers with zero load padding and
inactive-zero/inactive-preserve policies. This is not the global value-layer
type or shape limit.

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

## 28. Phase 11.3 value strided/ranked memory contract

PHASE11_VALUE_STRIDED_RANKED_MEMORY_CONTRACT=LOCKED
RANK2_ROW_SLICE_IDENTITY_SURFACE=ACCEPTED
RANK2_ROW_SLICE_STRIDED_OUTER_DYNAMIC_SURFACE=ACCEPTED
MEMREF_DIM_METADATA_TO_SCALAR_ARG_CONTRACT=LOCKED
HIDDEN_MEMREF_DESCRIPTOR_ALLOWED=NO
GATHER_LANE_STRIDE_STAGED=YES
READY_FOR_PHASE11_4_VALUE_RANKED_STRIDED_STATIC=YES
READY_FOR_TRITON=NO

Phase 11 extends the planned value memory contract beyond the locked Phase 10
rank-1 identity executable subset. This section is a planning contract only;
main executable lowering is intentionally not implemented in Phase 11.3.

Accepted Phase 11 value forms:

- `RANK1_FLATTENED_SCALAR_STRIDED_ADDRESS`: rank-1 global i32/f32 memref,
  scalar address expression `row * stride + col_block * 16`, rank-1 identity
  transfer, contiguous vector lanes, and Phase 10 full/empty/tail masks.
- `RANK2_ROW_SLICE_IDENTITY`: rank-2 identity global i32/f32 memref with
  `shape_args` for dynamic dimensions, scalar `[row, col]` transfer indices,
  and `vector<16xT>` mapped to the innermost dimension.
- `RANK2_ROW_SLICE_STRIDED_OUTER_DYNAMIC`: rank-2 global i32/f32 memref with
  `strided<[?, 1], offset: 0>`, `shape_args` for dynamic dimensions, one
  `stride_args` entry naming the dynamic outer row stride, static inner stride
  1, and static offset 0.
- `MEMREF_DIM_METADATA_LOWERING`: `memref.dim` on public global memref dynamic
  dimensions resolves to explicit scalar extent args, never hidden memref
  descriptors.

Planned Phase 11.4 static coverage:

- rank-1 flattened stride address lowers;
- rank-2 identity row-slice transfer lowers;
- rank-2 strided outer row-slice transfer lowers;
- `memref.dim` maps to shape args;
- nonunit inner stride rejects;
- lane-varying stride/gather rejects;
- column-slice rejects;
- hidden descriptor extraction rejects.

Staged forms remain non-unit inner stride, lane-varying stride/gather, scatter,
column/vertical slice, rank greater than 2, rank-2 vector tiles, block pointers,
boundary-check and padding-option semantics, reductions, dot, f16/subword
storage lowering, numeric casts, SFU/math expansion, hidden descriptors, and
sparse/unknown transfer masks outside the Phase 10 accepted set.

## 29. Phase 11.4 value ranked/strided static lowering lock

VALUE_RANK2_ROW_SLICE_TO_VC4KERNEL_STATIC=PASS
VALUE_MEMREF_DIM_METADATA_LOWERING=PASS
VALUE_STRIDED_RANKED_ADDRESS_PLANNER=YES
GATHER_LANE_STRIDE_STAGED=YES
HIDDEN_MEMREF_DESCRIPTOR_REJECTED=YES
VALUE_STRIDED_RANKED_MEMORY_STATIC=PASS
READY_FOR_PHASE11_5_VALUE_HARDWARE_ISOLATION=YES
READY_FOR_TRITON=NO

Phase 11.4 implements the static value-to-VC4Kernel executable subset for the
Phase 11 memory skeletons. `vector.transfer_read` and `vector.transfer_write`
now flow through a central `MemoryAddressPlan` helper that reuses the Phase 10
mask classifier and then computes one scalar element base index before the
existing byte-offset, TMU inactive-zero load, and VDW inactive-preserve store
paths.

Accepted static-lowering forms are:

- rank-1 flattened scalar-computed addresses, including `row * stride + col`
  expressions already present in value IR, with contiguous rank-1 transfer
  lanes;
- rank-2 identity row slices over `memref<?x?xT, #vc4value.global>` for
  `T=i32/f32`, where the row stride is the dim-1 extent named by
  `vc4value.shape_args`;
- rank-2 `strided<[?, 1], offset: 0>` row slices, where the outer row stride is
  the scalar named by `vc4value.stride_args`;
- metadata-only `memref.dim` on public global memref dynamic dimensions,
  resolved to explicit scalar extent arguments without descriptor loads.

Still-staged or rejected forms include lane-varying stride/gather, scatter,
column/vertical slices, non-unit inner stride, nonzero layout offsets, hidden
memref descriptors, rank-2 vector/tile transfers, block pointers, reductions,
dot/GEMV/GEMM, f16/subword storage lowering, numeric casts, and sparse/unknown
transfer masks beyond the Phase 10 full/empty/tail mask set.
