# VC4Kernel Surface v2 Stage Plan

**Status:** P0 Surface v2 pre-vector/pre-Triton plan.
**Date:** 2026-06-06.
**Scope:** documentation, matrix, and audit lock for the `vc4kernel` target-kernel planning surface above SSAVC4. P0 does not implement new execution semantics.

---

## VC4Kernel Surface v2 pre-vector lock

Surface v2 locks the next planning sequence after the accepted dynamic rectangular, spill, runtime GEMV/GEMM, blocked GEMV/GEMM, and lower-half branch-layout accounting baseline.

Normative boundary:

```text
standard value layer:
  owns vector/memref/arith/math/scf/cf producer semantics.

vc4kernel:
  owns VC4 target execution-plan semantics.

only lower path:
  vc4kernel -> ssavc4 -> scheduled vc4 -> artifacts/runtime/hardware.
```

There is no direct VC4KernelToVC4 path, no VC4Tile resurrection, and no producer tile DSL. `vc4kernel` is not a replacement for vector, memref, arith, math, scf, or cf.

Current special-case ops are migration targets, not compatibility promises:

```text
fragment_add/sub/mul/shl:
  historical legacy migration target removed_in_p1 after replacement by the P1 general ALU surface.

fragment_reduce add-only:
  historical migration target removed_in_p4 after replacement by the P4 general reduction surface.

tmu_load_fragment implicit safe-address behavior:
  removed_in_p7 after replacement by the explicit safe inactive-load policy;
  active sources use explicit scalar safe_offset plus inactive_load<zero>.

vdw_store_fragment implicit inactive-store behavior:
  migration target for P8 explicit inactive-store policy.
```

Do not keep a special-case op merely because it already exists.

---

## Exact P0-P13 ordering

```text
P0  spec/matrix/audit
P1  general ALU
P2  bitcast/constants
P3  comparisons
P4  reductions
P5  scalar arith
P6  memory/coherency
P7  TMU safe inactive load
P8  VDW inactive store v1
P9  pack/unpack/subword
P10 SFU/fastmath
P11 dynamic rotate/shuffle
P12 dynamic VPM/VDR/VDW coordinates
P13 final support matrix/pre-vector lock

Deferred:
  arbitrary sparse VDW stores
```

P1-P13 are planned implementation phases. A phase is not accepted until it has verifier coverage, VC4KernelToSSAVC4 coverage, lower-half coverage where needed, and hardware proof or deterministic-reject proof.

---

## Phase Contracts

### P0 spec/matrix/audit

Inventory the accepted baseline and lock the Surface v2 plan. P0 is docs, spec, matrix, and audit only. It must not change compiler semantics, add execution ops, start vector/memref/Triton/IREE/TTIR lowering, or run hardware as a substitute for missing documentation.

### P1 general ALU

Plan a general hardware-faithful ALU surface for VC4 add-pipe and mul-pipe semantics. `fragment_add`, `fragment_sub`, `fragment_mul`, and `fragment_shl` are historical legacy migration targets removed_in_p1. P1 must not preserve one-off fragment ALU ops as the long-term API merely because they existed.

### P2 bitcast/constants

Plan bitcast and constant materialization inside the target execution-plan boundary. Producer-level constant folding and value-layer construction remain outside `vc4kernel`.

### P3 comparisons

Plan general comparisons and predicate production. Current comparison special cases are not the complete long-term surface.

### P4 reductions

Plan general reductions. The historical add-only reduction form is removed_in_p4 and is not a compatibility promise.

### P5 scalar arith

Plan the scalar arithmetic subset that belongs in `vc4kernel` execution planning. Generic scalar/vector arithmetic remains owned by the standard value layer until lowered into target planning semantics.

### P6 memory/coherency

Lock memory path and coherency policy:

```text
TMU:
  global-to-register read path.

VDR:
  global-to-VPM path.

VDW:
  VPM/register-staged-to-global path.
```

Compiler spill slots written by VDW must reload through the coherent VDR->VPM path, not TMU, unless a future architecture-backed invalidation/coherency mechanism is specified and proven.

### P7 TMU safe inactive load

Replace implicit TMU safe-address behavior with explicit safe offset / inactive-load policy. Inactive lanes zero-fill. Empty masks avoid unsafe requests. No inactive lane may issue an out-of-bounds memory request.

### P8 VDW inactive store v1

VDW v1 supports:

```text
full:
  all lanes/elements active.

tail:
  contiguous prefix active; inactive destination lanes preserved.

rect:
  active rows/columns stored; inactive destination region preserved.
```

Arbitrary sparse VDW masks deterministic-reject until a later hardware-proven phase. Do not silently decompose sparse stores into read-modify-write stores in P8.

Surface v2 final P8 requires `inactive_store =
#vc4kernel.inactive_store<preserve>` on active VC4Kernel VDW global-store ops.
The attr-absent implicit preserve form and any sparse read-modify-write
fallback are removed_in_p8. P9 may extend subword/pack modes later; P8 applies
to the existing 32-bit/none executable store modes.

### P9 pack/unpack/subword

Plan pack/unpack and sub-32 VPM modes. P9b/P9c hardware-prove fragment
pack/unpack. P9d hardware-proves QPU VPM read/write w8/w16 packed and laned
modes with explicit width/subword attrs and deterministic rejects for
meaningless combinations. VDR/VDW DMA subword modes remain separate P9e work
until their MODEW/stride behavior is hardware-proven.

### P10 SFU/fastmath

Default math is exact/conservative. Approximate SFU-derived math is opt-in only through an explicit fastmath/approx contract. `sqrt` may lower via `rsqrt` only under that contract unless an exact tested sequence exists. NaN, Inf, and signed-zero exactness must not be promised without explicit tests.

### P11 dynamic rotate/shuffle

P11 accepts dynamic rotate as the only rotate-derived op surface. Lane
broadcast is accepted as a composite idiom over `lane_range`, scalar splat,
`fragment_cmp` eq, `fragment_select` payload-or-zero, and i32 `fragment_reduce`
add. Exact f32 payload broadcast uses `fragment_bitcast` to i32, the i32
composite, then bitcast back. There is no `vc4kernel.fragment_broadcast_lane`
op, and arbitrary shuffle/permutation remains a deterministic reject.

### P12 dynamic VPM/VDR/VDW coordinates

P12 locks dynamic VPM/VDR/VDW coordinates and byte/halfword selectors only
where lower-half support, range verification, and hardware proof exist.
Word-X and subword selector are separate fields; dynamic selector is not
dynamic subword mode; setup-field masks are not modulo semantics. VDR and VDW
are asymmetric and keep separate setup logic. QPU horizontal w32 dynamic word-X
rejects as not meaningful; DMA laned modes are hardware-forbidden; vector
coordinate operands, sparse VDW/scatter, dynamic orientation/width/subword
attrs, out-of-range constants, and unencodable pitch/stride cases without a
proven fallback deterministic-reject. No hardware-backed coordinate/selector
mode remains deferred in P12.

### P13 final support matrix/pre-vector lock

Close the final Surface v2 support matrix. Every feature must be classified as implemented with proof, planned, migration target, deterministic reject, or deferred. Only after P13 may the project begin the next vector/pre-Triton surface work.

---

## Forbidden Permanent Surface

The following are forbidden as permanent `vc4kernel` Surface v2 features:

```text
tile_broadcast
tile_dot
tile_matmul
tile_contract
fragment_contract
producer-level layout algebra
arbitrary sparse VDW store before its deferred phase
integer div/mod unless a library sequence is designed
atomics
tile-buffer color/Z/stencil
texture filtering
cube maps
varyings for compute v1
```

---

## Verification Philosophy

Every phase after P0 must preserve the accepted baseline and add evidence rather than weakening existing checks.

Required evidence per feature:

```text
verifier coverage
VC4KernelToSSAVC4 coverage
lower-half coverage where needed
hardware fixture proof or deterministic-reject proof
```

Standing integrity rules:

```text
no direct vc4kernel -> scheduled vc4 conversion
no fixture-name/public_name/candidate/path/status special cases
no host-side result substitution
no stale generated artifacts as acceptance evidence
no weakened hardware oracles, result checkers, sentinels, timeout behavior, or CPU references
no producer/vector/Triton lowering before the pre-vector lock is accepted
```

P0 verification is local build and lit only. Hardware is not required for P0 because it changes docs/matrix/audit artifacts only.
