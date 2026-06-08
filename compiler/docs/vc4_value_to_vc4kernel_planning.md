# VC4 Value-to-VC4Kernel Planning Guide

## 1. Purpose and non-goals

This document specifies how later lowering phases should plan standard
value-surface patterns into the locked VC4Kernel target surface.

It is not an implementation. It does not add accepted value features, does not
add a `vc4value` dialect, does not add a verifier, does not add conversion
passes, does not add TTIR import, does not run hardware, and does not change the
locked VC4Kernel surface.

The guide is a planning contract for future phases. It describes the decisions
those phases must make before creating VC4Kernel IR from:

```text
func + tiny vc4value + vector + memref + arith + math + scf/cf
```

Feature classification remains in
`compiler/docs/vc4_value_ttir_feature_taxonomy.json`.

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

`vc4value.program_id {axis = 0|1|2}` maps to the VC4Kernel program-id operation
for the same logical axis.

`vc4value.num_programs {axis = 0|1|2}` maps to VC4Kernel num-programs behavior
and associated resource/launch metadata.

Axes are logical launch-grid axes only. They are not physical QPU IDs and must
not expose physical scheduling to the value layer.

Index values may remain `index` in the value surface, but the planner must prove
the target i32 conversion policy before constructing VC4Kernel scalar i32
operands.

## 4. ABI and memref lowering plan

A value memref base pointer lowers to a raw i32 uniform for VC4Kernel. Dynamic
sizes, strides, offsets, and other ABI values lower to scalar arguments or
metadata after the value-layer ABI plan proves their type and range.

The initial ABI is contiguous rank-1 global memory:

```text
memref<?xf32>
memref<?xi32>
```

Later rank-2 strided ABI support must preserve rank, shape, pitch, stride,
layout, memory space, and element type until the planner chooses a target memory
path.

Read-only, write-only, and inout attributes are planning hints. They can guide
TMU, VDR, and VDW path selection, but they are not permission to weaken source
memory semantics or skip required inactive-lane handling.

## 5. Lane and fragment planning

`vector.step` maps to VC4Kernel lane identity / `lane_range` equivalent when the
vector shape is `vector<16xindex>` or can be safely converted to the target i32
lane carrier.

`vector<16xi32>` and `vector<16xf32>` map to VC4Kernel fragment carrier types.
The value planner must not invent wider VC4Kernel fragment carriers. Larger
blocks split or loop over `vector<16>` fragments.

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

`scf` may appear in value input. Raw `scf` must not survive into verified
VC4Kernel.

Before or during value-to-VC4Kernel lowering, `scf` must canonicalize to
`cf`/block arguments, predicated forms, or another specified intermediate that
the VC4Kernel planner can lower safely.

Natural loops, block arguments, liveness, spills, and branch layout must be
preserved honestly. The planner must not reshape kernels to hide backend bugs.

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
- Phase 9 masks/memory legality: mask classifier, transfer read/write legality,
  TMU safe-offset loads, and VDW preserve stores;
- Phase 10 gather/strided: gather-load and affine strided memory planning;
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
