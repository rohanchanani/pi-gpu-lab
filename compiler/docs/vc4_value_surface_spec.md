# VC4 Standard Value Surface Specification

## 1. Purpose and layer boundary

The VC4 standard value surface is the producer-facing layer above VC4Kernel. It
uses standard MLIR value semantics plus a tiny amount of `vc4value`
launch/policy plumbing:

```text
func + tiny vc4value + vector + memref + arith + math + scf/cf
```

The value surface lowers to VC4Kernel. It does not lower to SSAVC4 or scheduled
VC4, and it does not bypass VC4Kernel. The only accepted path remains:

```text
standard value layer -> vc4kernel -> ssavc4 -> scheduled vc4
  -> artifacts/runtime/hardware
```

This document specifies the planned source contract for later handwritten value
lowering. It does not implement any dialect, verifier, import path, conversion,
runtime behavior, or hardware fixture.

The value surface is tracked alongside the Phase 1 taxonomy in
`compiler/docs/vc4_value_ttir_feature_taxonomy.json`. Taxonomy rows classify
features; this document explains the human-readable value-layer contract those
rows are expected to preserve.

Phase 3.5 refines the vector abstraction boundary in
`compiler/docs/vc4_value_surface_abstraction_policy.md`: the value surface is a
target-profiled standard MLIR subset, while `vector<16xT>` is the Phase 5 V1
lowerable fragment-normal form rather than the global value-layer type limit.

No vector dialect ops are legal inside verified VC4Kernel. VC4Kernel may use
`vector<16xT>` carrier types for target fragment values, but those carrier types
are not permission for producer dialect operations to appear in verified
VC4Kernel hardware fixtures.

## 2. Non-goals and hard exclusions

The value surface explicitly forbids:

- direct Triton-to-VC4Kernel as the first producer path;
- direct VC4KernelToVC4;
- VC4Tile resurrection;
- TMU, VDR, VDW, or VPM operations in the value surface;
- `vc4value` tile, memory, fragment, lane-id, or barrier operations;
- producer dialect operations inside verified VC4Kernel;
- claiming Triton support before Phase 7.

The value layer must not become a second target dialect. It should preserve
source value semantics so the planner can choose among accepted VC4Kernel
memory, fragment, predicate, math, resource, and artifact mechanisms later.

Documentation in Phase 1 may say a feature is specified, planned, supportable,
or temporarily rejected. It must not claim executable value/Triton support from
the existence of this document.

## 3. Allowed dialect set

The allowed initial value-surface dialect set is exactly:

```text
builtin, func, vc4value, vector, memref, arith, math, scf, cf
```

The following dialect families are not accepted in the initial verified value
path:

```text
tt, ttg, gpu, linalg, tensor, nvgpu, nvvm, rocdl, spirv, iree, stablehlo, mhlo
```

`linalg` may be useful as a future lowering input, but it must canonicalize to
the value surface first. Tensor IR may also be useful above the value surface,
but it must lower into the target-profiled value subset first. The initial
verified path consumes `vector` and `memref` forms, not `tensor` or `linalg`
ops.

Real TTIR is a future producer input. It must lower into this value surface
after the handwritten path is proven; TTIR ops are not part of the value surface
itself.

## 4. Tiny vc4value launch/policy plumbing

`vc4value` is specified as tiny launch and policy plumbing. Phase 2 implements
this; Phase 1 only specifies it.

Accepted initial ops:

```mlir
vc4value.program_id   {axis = 0|1|2} : index
vc4value.num_programs {axis = 0|1|2} : index
```

Accepted attrs:

```text
vc4value.kernel
vc4value.grid_rank
vc4value.math_policy / approx policy marker if needed later
vc4value.target_profile if needed later
```

Explicitly forbidden `vc4value` operations or concepts:

```text
vc4value.load
vc4value.store
vc4value.tile
vc4value.vpm
vc4value.tmu
vc4value.vdr
vc4value.vdw
vc4value.fragment
vc4value.lane_id
vc4value.barrier
```

If `vc4value` starts to expose memory hardware, tiles, fragments, lanes, or
barriers, it has violated the value-surface boundary.

## 5. Kernel wrapper model

The value kernel wrapper is `func.func` with `vc4value` metadata attributes.
Conceptual syntax:

```mlir
func.func @kernel(%x: memref<?xf32, #vc4value.global>, ...)
    attributes {vc4value.kernel, vc4value.grid_rank = 1} {
  ...
}
```

The exact memory-space spelling may be finalized in Phase 4. The semantic
decision is already fixed: the value layer uses logical `memref` arguments, and
VC4Kernel receives raw i32 base pointer uniforms plus scalar sizes, strides, and
other ABI values after value-to-VC4Kernel planning.

The wrapper must preserve enough type, rank, layout, and metadata information
for later lowering. It must not expose raw TMU, VDR, VDW, VPM, or physical QPU
details.

## 6. Program ID and launch-grid semantics

`vc4value.program_id` axes 0, 1, and 2 are logical launch-grid axes. They are
not physical QPU IDs and must not be used to infer physical scheduling.

The conceptual decomposition from a linear logical request id is row-major:

```text
pid0 = linear_pid % grid0
pid1 = (linear_pid / grid0) % grid1
pid2 = linear_pid / (grid0 * grid1)
```

`vc4value.num_programs {axis = n}` returns the corresponding logical grid
extent. The value surface should keep these as `index` values until a target
profile or ABI rule proves conversion to i32 is safe.

## 7. Memref ABI model

The value layer uses logical `memref` rank, element type, layout, memory space,
and access attributes. It does not expose raw VC4 memory mechanisms.

VC4Kernel receives raw i32 base pointer uniforms plus scalar sizes, strides,
offsets, and policy values after lowering. That raw pointer ABI is a lower
target contract, not the producer-facing value syntax.

Initial V1 value memory supports:

```text
memref<?xf32>
memref<?xi32>
```

for contiguous 1D global buffers. The recommended conceptual memory space is
`#vc4value.global`, but Phase 4 may finalize the exact TableGen spelling.

Later phases may support rank-2 strided memrefs. Those phases must preserve
rank, shape, layout, pitch, and stride semantics long enough for the planner to
choose TMU, VDR/VPM, VDW, or a diagnostic.

## 8. Vector type and lane model

The value surface admits fixed, non-scalable rank-1 and rank-2 vectors with
allowed element types. `vector<16xT>` is the Phase 5 V1 lowerable
fragment-normal form for executable elementwise lowering; it is not the global
value-layer type limit.

Surface-admissible vector forms include:

```text
vector<16xi32>        ; Phase 5 V1 lowerable arithmetic carrier
vector<16xf32>        ; Phase 5 V1 lowerable arithmetic carrier
vector<16xindex>      ; value-level address/program-id support form
vector<16xi1>         ; mask/predicate support form
vector<NxT>           ; staged split/tail form when N != 16
vector<MxNxT>         ; staged tile/contract form
```

Allowed vector element types are exactly:

```text
i1, index, i8, i16, i32, f16, f32
```

`i1` is for masks and predicate sources. `index` is for value-level address and
program-id arithmetic. `i8` and `i16` are subword storage and data-movement
surface types, not native arithmetic promises. `f16` is a storage-conversion
surface type: f16 storage conversion plus f32 compute is accepted downstream,
but native f16 arithmetic remains rejected. `i32` and `f32` are the first
executable arithmetic carriers for Phase 5 V1.

Fixed rank-1 `vector<NxT>` where `N != 16` is surface-admissible and staged for
later splitting into `vector<16>` fragments plus tails or loops. Fixed rank-2
`vector<MxNxT>` is surface-admissible and staged for later tile, contract, and
VPM planning. Scalable vectors remain deterministic rejects in the initial
value surface.

`vector.step` is the preferred arange/lane representation. The common first V1
lane range is:

```mlir
%lane = vector.step : vector<16xindex>
```

`BLOCK_SIZE=16` is the first lowerable block size. Larger or non-16 blocks are
not permanent value-surface rejects; they are staged until splitting, tail
masks, loops, or rank-2 tile planning are implemented. The value surface must
not invent a wider VC4Kernel fragment carrier or assume physical lanes beyond
the locked SIMD-16 target model.

## 9. Mask model

The value mask categories are:

- full: all lanes active;
- empty: no lanes active;
- tail: active prefix, usually `lane < n`;
- rect: dense rectangular tile mask;
- sparse: arbitrary lane pattern;
- unknown: not classified by the legality analysis yet.

Compute masks may be supportable through predicate and select composition. Store
masks are stricter. Sparse or unknown stores reject unless a future
hardware-proven phase adds support or a separate value-level emulation plan is
specified. Sparse VDW stores remain deterministic rejects at the locked
VC4Kernel target surface.

Every memory-affecting mask must be classified before lowering. The value
surface must not silently map an unknown mask to a VC4Kernel store path.

## 10. Transfer read/write model

The initial value memory primitives are `vector.transfer_read` and
`vector.transfer_write`.

The value layer does not name TMU, VDR, VDW, or VPM. The planner chooses the
target mechanism later based on memref rank/layout, vector shape, mask class,
element type, and reuse/tile structure.

For reads, inactive-load zero behavior is a VC4Kernel planning requirement, not
a value op. A later lowering may use the locked TMU safe-offset inactive-load
contract when the access pattern is legal.

For writes, inactive-store preserve behavior is a VC4Kernel planning
requirement, not a value op. A later lowering may use the locked VDW
inactive-preserve store contract when the store mask is full, tail, or dense
rectangular.

The value surface keeps memory semantics source-like. Target memory path choice
belongs to value-to-VC4Kernel planning.

## 11. Elementwise arithmetic, comparison, and select

The initial elementwise families are:

- i32 add, sub, mul, bitwise operations, shifts, and min/max where specified;
- f32 add, sub, mul, and basic target arithmetic where specified;
- i32 comparisons;
- finite-policy f32 comparisons;
- select;
- constants and splats.

These are specified value forms. They may later lower to locked VC4Kernel
fragment constants, splats, selects, add-pipe and mul-pipe ALU opcodes, and
comparison predicates.

Finite f32 comparisons require explicit finite policy. NaN, infinity, exception
flag, signed-zero, and full default IEEE behavior are not silently claimed by
the value surface.

## 12. Reductions

i32 reductions are the straightforward initial reduction family, subject to
operation-specific identity and mask rules.

f32 reductions require explicit finite-tree or approximate policy. Strict IEEE
f32 reductions must not silently lower to VC4 finite-tree reduction. A later
phase may specify an emulation path, but until then strict IEEE f32 reductions
are a temporary staging reject, not a permanent hardware impossibility claim.

Masked reductions must specify inactive lane identity behavior. Tail and
rectangular reduction cases must preserve source semantics before lowering to
any target fragment reduction or rotate-derived composite.

## 13. Math policy

The value surface distinguishes these math policies:

- strict/exact: source-default exact or strict behavior;
- finite: finite-only behavior, excluding NaN/Inf/default IEEE edge claims;
- triton_fast: source or profile opts into fast/relaxed semantics;
- approx_sfu: explicit target approximate SFU policy.

Approximate SFU lowering is explicit policy only. Exact/default math does not
silently lower to SFU.

There is no direct SFU sqrt in the locked target contract. Approximate sqrt may
only be specified as an explicit approximate composite, such as an rsqrt-based
expansion, when the source policy permits it. Exact sqrt is a temporary staging
reject until an exact or emulated plan is specified.

Exp/log base semantics must be explicit. VC4 SFU proof is for target
approximate base-2 forms; source `math.exp`, `math.log`, `math.exp2`, and
`math.log2` must not be conflated.

## 14. Subword and f16 storage model

i8 and i16 storage/subword paths are table-backed by the taxonomy and later
planning docs. The value layer may express element storage through memref and
vector types, but the target packed mode remains a planner decision.

Dynamic subword selector means byte or halfword selector inside an accepted
static packed mode. It is not dynamic width, dynamic subword mode, dynamic
orientation, or dynamic layout.

f16 storage conversion plus f32 compute is specified as the accepted value
model. Source f16 storage may lower through target f16 storage conversion when
the explicit policy allows f32 compute.

Native f16 arithmetic is rejected at the locked VC4Kernel target surface.
Native bf16/fp8 arithmetic or conversion is also rejected. Future value-level
emulation would require a separate specification and proof; this document does
not claim it.

## 15. Shuffle, rotate, and lane broadcast model

Dynamic rotate is specified where a value pattern maps to the locked rotate
operation: one scalar rotate amount, modulo-16 lane semantics, and no
lane-varying rotate amount.

Lane broadcast is a composite idiom, not a new first-class VC4Kernel operation.
The canonical integer composite is:

```text
lane_range -> compare lane == selected_lane -> select value or zero
  -> fragment_reduce add
```

For f32 bit-preserving broadcast, the value should bitcast the f32 payload to
i32, use the i32 composite, and bitcast back. Numeric finite f32 broadcast may
be specified separately under finite policy.

Arbitrary direct shuffle/permutation is rejected at VC4Kernel. Value-level
emulation may be future supportable, but it must be specified as an emulation
plan rather than a direct VC4Kernel shuffle or permutation.

## 16. Control-flow boundary

`scf` may appear in value input. Raw `scf` must not survive into verified
VC4Kernel.

Before or during value-to-VC4Kernel lowering, structured control must
canonicalize into forms the target planner can handle, such as `cf` branches,
block arguments, predication, or other specified intermediate forms. Natural
loops and block args should be preserved until the lowering can prove correct
allocation, spill, and branch-layout behavior.

The lowering must not reshape kernels merely to hide backend bugs.

## 17. Contract/GEMM placeholder model

`tt.dot` and `vector.contract` are ML-critical, but they are not V1 executable
value features.

The first natural planned shape is row-fragment GEMM:

```text
BLOCK_M = 1
BLOCK_N = 16
small K staged through planned memory/tile movement
```

The value surface should express this as `vector.contract` or an equivalent
canonical vector/memref pattern. The planner may later choose fragment ALU,
reductions, VPM, VDR, and VDW mechanisms. This document does not implement or
claim contract lowering.

## 18. Diagnostics and deterministic rejects

Diagnostics must distinguish:

- valid but not implemented yet;
- temporary reject because no staged lowering exists;
- deterministic reject by value-surface policy;
- deterministic reject by locked VC4Kernel contract;
- permanent reject requiring hardware or semantic impossibility proof.

Temporary rejects are expected during staged bring-up. They must not be
described as permanent without proof that VC4 cannot preserve the required
semantics even through slow emulation.

Deterministic rejects must be explicit for:

- direct VC4Kernel sparse VDW stores;
- direct arbitrary VC4Kernel shuffle/permutation;
- dynamic subword width/mode/orientation/layout;
- native f16 arithmetic;
- native bf16/fp8 arithmetic/conversion;
- exact/default math paths that would otherwise silently lower to approximate
  SFU;
- TTIR, TTGIR, GPU, or vendor-specific dialect ops in the initial value path.

Diagnostics should reference `compiler/docs/vc4_value_ttir_feature_taxonomy.json`
for classification and the locked VC4Kernel docs for target-surface rejects.

## 19. Initial executable subset by phase

These are planned later implementation scopes, not claims that this document
implements them:

- Phase 2 tiny `vc4value`;
- Phase 3 verifier;
- Phase 4 ABI;
- Phase 5 elementwise value lowering;
- Phase 8 control;
- Phase 9 masks/memory legality;
- Phase 11 reductions;
- Phase 13 math;
- Phase 15 subword/f16 storage;
- Phase 17 shuffle/rotate/broadcast;
- Phase 18 tile planner;
- Phase 19 contract.

Each phase must preserve the lower path through VC4Kernel and must not weaken
existing VC4Kernel verifier, audit, fixture, or mixed-claim contracts.

## 20. Relationship to TTIR and future producer imports

Real TTIR ingestion begins only after the handwritten value path is proven.
READY_FOR_TRITON remains NO now.

Triton source, real TTIR, and any other producer import must lower into the
standard value surface first. They must not bypass the value surface, lower
directly to VC4Kernel as the first path, introduce producer dialect ops inside
verified VC4Kernel, or rely on VC4Tile history.

Future TTIR profile docs must classify each TTIR construct against
`compiler/docs/vc4_value_ttir_feature_taxonomy.json` and this value-surface
contract before implementation begins.
