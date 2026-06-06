# VC4Kernel Surface v2 Decision Traceability Matrix

**Date:** 2026-06-06
**Purpose:** P0 matrix linking locked Surface v2 decisions to planned phases, required documentation/spec impact, and required proof. P1-P13 entries are planned unless explicitly marked accepted baseline.

---

## VC4Kernel Surface v2 pre-vector lock

| Decision | Locked answer | Phase | Required proof |
|---|---|---:|---|
| Layer boundary | Standard value layer owns vector/memref/arith/math/scf/cf; VC4Kernel owns VC4 target execution-plan semantics. | P0 | Spec text and verifier boundary tests when implementation changes. |
| Lower path | Only `vc4kernel -> ssavc4 -> scheduled vc4 -> artifacts/runtime/hardware`. | P0, all | Static scans and conversion tests showing no direct VC4KernelToVC4 path. |
| Producer tile DSL | No producer tile DSL and no VC4Tile resurrection. | P0, all | Static scans and forbidden-op diagnostics. |
| Compatibility bias | Existing special-case ops are migration targets, not compatibility promises. | P0-P4 | Updated spec and eventual migration tests. |
| Surface evidence | Every feature needs verifier, VC4KernelToSSAVC4, lower-half coverage where needed, and hardware proof or deterministic-reject proof. | P0-P13 | Final P13 support matrix. |

---

## Phase Ordering

| Phase | Name | Status in P0 | Scope |
|---:|---|---|---|
| P0 | spec/matrix/audit | accepted baseline setup | Documentation, inventory, support matrix planning only. |
| P1 | general ALU | planned | General hardware-faithful ALU surface; migrate fragment add/sub/mul/shl. |
| P2 | bitcast/constants | planned | Bitcast and constant materialization inside target planning boundary. |
| P3 | comparisons | planned | General comparisons and predicate production. |
| P4 | reductions | planned | General reductions; migrate add-only `fragment_reduce`. |
| P5 | scalar arith | planned | Scalar arithmetic subset belonging in VC4Kernel planning. |
| P6 | memory/coherency | planned | TMU/VDR/VDW/VPM paths, coherency, and spill reload policy. |
| P7 | TMU safe inactive load | planned | Explicit safe offset / inactive-load policy. |
| P8 | VDW inactive store v1 | planned | Full/tail/rect preserve; sparse deterministic reject. |
| P9 | pack/unpack/subword | planned | Pack/unpack and sub-32 VPM modes with executable rejects until proven. |
| P10 | SFU/fastmath | planned | Explicit fastmath/approx contract for SFU-derived math. |
| P11 | dynamic rotate/shuffle | planned | Dynamic rotate/shuffle only if hardware-proven. |
| P12 | dynamic VPM/VDR/VDW coordinates | planned | Dynamic coordinates/pitch/stride only where proven. |
| P13 | final support matrix/pre-vector lock | planned | Close feature matrix before vector/pre-Triton work. |
| Deferred | arbitrary sparse VDW stores | deferred | No silent decomposition before a later hardware-proven phase. |

---

## Locked Policy Matrix

| Feature or policy | Current classification | Required Surface v2 outcome | Required proof |
|---|---|---|---|
| `fragment_add/sub/mul/shl` | removed_in_p1 historical legacy migration | Replace long-term special cases with P1 general ALU. | Verifier, conversion, lower-half, hardware or deterministic-reject evidence per admitted opcode. |
| Bitcast/constants | planned | P2 target-planning semantics without producer-dialect admission. | Dialect verifier and conversion tests. |
| Comparisons | planned | P3 general comparison surface. | Predicate verifier/conversion tests and hardware proof where executable. |
| `fragment_reduce` add-only | migration target | P4 general reductions. | Reduction verifier/conversion/hardware matrix. |
| Scalar arith | planned | P5 scoped scalar arithmetic subset. | Verifier and lowering tests; no vector producer ops admitted. |
| TMU memory path | accepted baseline plus P7 migration | Explicit safe inactive-load policy; inactive lanes zero-fill. | Verifier/conversion tests and hardware zero-fill proof. |
| VDR global-to-VPM path | accepted baseline | Preserve natural shared-memory load path, including dynamic rect baseline. | Existing dynamic VDR canaries plus future matrix entries. |
| VDW VPM/register-to-global path | accepted baseline plus P8 migration | Full/tail/rect preserve in v1; sparse deterministic reject. | Sentinel hardware proof for admitted classes; deterministic-reject lit for sparse. |
| Spill coherency | accepted baseline policy | VDW-written compiler spill slots reload through coherent VDR->VPM path, not TMU, unless future invalidation is proven. | Lower-half tests and hardware canaries. |
| Pack/unpack/subword | planned | P9 hardware-faithful surface with executable rejects until proven. | Verifier reject tests and later hardware proof. |
| SFU/fastmath | planned | P10 explicit fastmath/approx opt-in; default exact/conservative. | Verifier/policy attrs, conversion tests, hardware proof; no untested NaN/Inf/signed-zero promises. |
| `sqrt` via `rsqrt` | planned under P10 only | Allowed only under explicit fastmath/approx unless exact sequence is implemented and tested. | Contract tests and numerical hardware fixtures. |
| Dynamic rotate/shuffle | planned | P11 only if downstream and hardware proof exists. | Verifier/conversion/hardware proof. |
| Dynamic VPM/VDR/VDW coordinates | planned with accepted dynamic rect baseline | P12 admission only where range verification and lower-half proof exist. | Dynamic-coordinate verifier/lowering/hardware matrix. |
| Sparse VDW store | deferred deterministic reject | Must reject until a later hardware-proven phase; no silent RMW decomposition. | Deterministic-reject lit and static policy scans. |

---

## Forbidden Permanent Surface

| Forbidden surface | Required handling |
|---|---|
| `tile_broadcast` | Forbidden permanent surface. |
| `tile_dot` | Forbidden permanent surface. |
| `tile_matmul` | Forbidden permanent surface. |
| `tile_contract` | Forbidden permanent surface. |
| `fragment_contract` | Forbidden permanent surface for Surface v2. |
| Producer-level layout algebra | Keep above VC4Kernel in the standard value layer. |
| Arbitrary sparse VDW store before deferred phase | Deterministic reject. |
| Integer div/mod | Forbidden unless a library sequence is designed. |
| Atomics | Forbidden for compute v1. |
| Tile-buffer color/Z/stencil | Forbidden for compute v1. |
| Texture filtering / cube maps / varyings | Forbidden for compute v1. |

---

## Accepted Baseline to Preserve

| Baseline area | Evidence class to preserve |
|---|---|
| Dynamic rectangular VDR/VDW | Existing dynamic rect lowering, canaries, and runtime pitch/stride fixtures. |
| Runtime GEMV/GEMM | Existing accepted naive/blocked runtime fixtures and source shapes. |
| Lower-half branch-layout accounting | Existing branch-layout static audit and closure tests. |
| Spill and VPM row accounting | Existing spill transport, edge-copy, and VPM row resource tests. |
| Hardware discipline | Existing sentinels, expected results, CPU references, timeout/power-cycle behavior, and freshness checks. |

---

## P13 Matrix Rule

P13 must publish a final support matrix. Each Surface v2 feature must be marked with exactly one precise status:

```text
accepted baseline
implemented and hardware-proven
implemented with deterministic-reject proof
planned
migration target
deferred
```

No P13 entry may rely on comments alone, fixture-name dispatch, public-name dispatch, candidate-path dispatch, generated-output strings, or host-side result substitution.
