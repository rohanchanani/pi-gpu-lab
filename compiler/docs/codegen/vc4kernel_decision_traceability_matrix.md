# VC4Kernel Decision Traceability Matrix

**Date:** 2026-05-31  
**Purpose:** companion checklist linking the design decisions from the corrected planning discussion to specification changes, implementation slices, tests, and hardware evidence.

---

## Decision matrix

| Decision | Final answer | Spec impact | Implementation stage | Required proof |
|---|---|---|---|---|
| Lower-half compatibility | Not a design input. Refactor lower half to the natural final model. | Normative rule + design principles | P0, all later | No compatibility-only fields/aliases in final scans |
| NVIDIA resource analogy | Separate compile config, compiler-computed resource metadata, and runtime launch metadata. | Resource sections | P2, P3 | Resource metadata lit + manifest tests |
| User-authored resource dict | Remove from source `vc4kernel`. Compute resources from body/planner. | Kernel attrs/resource sections | P1, P2 | `invalid-user-resource-metadata.mlir`; computed `vc4.resource` checks |
| `lane_id` | Remove. Use `lane_range`. | Op inventory/identity | P1 | No `vc4kernel.lane_id` in source except invalid tests |
| `block_id` | Remove. Use `program_id`; use `warp_id` only for cooperative local warp identity. | Op inventory/identity | P1 | No `vc4kernel.block_id` in source except invalid tests |
| `program_id` | Keep as logical program/request id. Not physical QPU number. Supports axis 0, 1, and 2 with `num_programs`. | Identity/ABI | P1, P8, dynamic rectangular transfer / runtime GEMM-GEMV checkpoint | Program-id writeback fixture; 3D identity fixture |
| `warp_id` | Keep for cooperative-block local warp id. | Identity/ABI/barrier | P1, P8, P13 | Barrier fixture and ABI checks |
| `lane_range` | Keep; lowers to element_number-derived vector. | Identity | P1, P8 | lane-range lit and vector-store hardware |
| Scalar `f32` formals | First-class scalar uniform reads. | Type/ABI/lowering | P4, P8 | f32 uniform/splat lit + SAXPY hardware |
| Scalar `i1` | Condition plan, not ordinary persistent data by default. | arith/control/lowering | P4, P5 | cmp/select/cond_br lit |
| `fragment_cmp` | Keep. Produces `general_mask` unless proven structured. | Predicate/fragment sections | P1, P5 | fragment-cmp/select lit, masked hardware paths |
| Predicate model | `!vc4kernel.pred<16>` supports full, empty, tail_prefix, rect_row, general_mask. | Predicate sections | P1, P5 | General-mask lit; no vector<16xi1> |
| TMU masked loads | Support full, empty, tail, general masks; inactive lanes zero; no unsafe inactive requests. | Memory/lowering | P10 | Tail/general zero-fill hardware |
| `vdw_store_fragment` | Keep as planning op, lower through hidden VPM staging. | Memory/resource | P7, P9 | full/tail/general store hardware |
| General masked VDW store | Preserve inactive memory by fallback, e.g. old-row TMU load + select + staging + VDW. | Memory/predicate | P9 | sentinel-preservation hardware |
| Explicit VPM ops | Keep for real shared-memory algorithms. | Memory/VPM | P1, P11 | VPM horizontal/vertical hardware |
| Hidden VPM rows | Include in computed resource allocation. | Resource/VPM | P7 | resource lit + hardware stores |
| VPM ownership | Runtime-assigned `vpm_base_row`, not physical QPU number. | ABI/resource/runtime | P3, P7 | manifest and multi-resident VPM tests |
| Barrier semaphores | Runtime-assigned `semaphore_base`; v1 uses 4 semaphores/block. | ABI/resource/barrier | P3, P13 | barrier hardware fixture |
| VPM/VDR/VDW modes | Hardware-derived schema; horizontal/vertical/strided/pitched 32-bit executable in v1. | attrs/memory/lower half | P6, P11 | horizontal and vertical hardware fixtures |
| Sub-32 modes | Schema allowed, executable use rejected until future milestone. | attrs/precision | P6 | invalid executable subword tests |
| i32 fragment multiply | Full 32-bit modular semantics; `mul24` fast path only if proven safe; otherwise software fallback. | fragment/lowering | P12 | fast-path and fallback hardware fixtures |
| f32 fragment multiply | Hardware fmul. | fragment/lowering | P8 | fragment arithmetic and SAXPY hardware |
| Reductions | Rotate/ALU tree after masking inactive lanes to zero. | fragment/lowering | P8 | reduce hardware fixture |
| Direct shortcuts | Forbidden: no direct vc4kernel->vc4, no QASM, no host substitution, no fixture dispatch. | acceptance/integrity | all | static scans + hardware freshness |

---

## Dynamic rectangular transfer decisions

| Decision | Hardware basis | Upstream motivation | Implementation checkpoint | Hardware proof fixture names |
|---|---|---|---|---|
| 3D `program_id(axis)` / `num_programs(axis)` | Runtime launch metadata can supply logical grid coordinates independently of physical QPU number; generated launchers already accept `vc4_dim3`. | Triton and vector lowering need CUDA-like grid axes for tiled GEMM/GEMV, not only a flattened axis-0 request. | Dynamic rectangular transfer / runtime GEMM-GEMV checkpoint. | `program_id_3d_writeback_vc4kernel`, `num_programs_3d_vc4kernel` |
| Dynamic VDR rect load | VC4 VDR/VCD is the hardware-natural global-to-VPM path; lowering may use one DMA when setup fields are encodable or row-by-row fallback otherwise. | `vector.transfer_read` and blocked GEMM/GEMV need shared/VPM tile fills with runtime active rows/cols and runtime leading dimensions. | Dynamic rectangular transfer / runtime GEMM-GEMV checkpoint. | `vdr_rect_runtime_pitch_zero_fill_vc4kernel`, `vdr_rect_row_by_row_fallback_vc4kernel` |
| Dynamic VDW rect store | VC4 VDW is the hardware-natural VPM-to-global path; inactive rows/cols require preserve-destination semantics. | `vector.transfer_write` and blocked GEMM/GEMV need runtime-shaped tile writeback without padding or fixture-specific layouts. | Dynamic rectangular transfer / runtime GEMM-GEMV checkpoint. | `vdw_rect_runtime_stride_preserve_vc4kernel`, `vdw_rect_row_by_row_fallback_vc4kernel` |
| Runtime leading dimensions for GEMM/GEMV | Global row pitch is a runtime scalar in general kernels; static VDR/VDW attrs only cover specialized fixed-shape movement. | CUDA/Triton-style GEMM/GEMV use runtime `m`, `n`, `k` and leading dimensions while tile sizes remain compile-time. | Dynamic rectangular transfer / runtime GEMM-GEMV checkpoint. | `gemv_naive_vc4kernel`, `gemv_blocked_vpm_vc4kernel`, `gemm_naive_vc4kernel`, `gemm_blocked_vpm_vc4kernel` |
| Row-by-row fallback for unencodable pitch/stride | VC4 setup encodings constrain single-DMA pitches; emitting one row at a time preserves semantics when a full rectangular setup cannot encode runtime pitch/stride. | Producer lowering should target semantic rectangular movement and let the compiler pick a legal hardware plan. | Dynamic rectangular transfer / runtime GEMM-GEMV checkpoint. | `vdr_rect_row_by_row_fallback_vc4kernel`, `vdw_rect_row_by_row_fallback_vc4kernel` |
| GEMM/GEMV blocked VPM reuse | VPM is the shared-memory resource; VDR/VDW should move global rectangles directly to/from VPM when possible. | Future `vector.contract` lowering needs real shared-memory blocking/tiling, not padded matrices or TMU-to-register-to-VPM as the primary path. | Dynamic rectangular transfer / runtime GEMM-GEMV checkpoint. | `gemv_blocked_vpm_vc4kernel`, `gemm_blocked_vpm_vc4kernel` |

---

## Current-to-target delta checklist

Starting point after pre-lowering cleanups:

```text
- VC4Kernel exists.
- FunctionOpInterface is implemented.
- Source lit works for dialect and conversion.
- VC4Tile is gone from source-controlled tree.
- Existing verifier/tests match the old strict spec.
```

Needed deltas before corrected surface is locked:

```text
- delete lane_id and block_id ops
- delete source resource dict requirement
- add warps_per_block source attr
- add/adjust VPM mode attrs: horizontal/vertical, w32/w16/w8, none/packed/laned
- add y/x coordinate operands to VPM ops
- add vdw_store_vpm if needed for full rectangular VPM->global DMA
- change predicate verifier to allow general_mask where consumer can lower it
- update invalid tests around fragment_cmp/general masks
- add invalid user resource metadata test
- revise docs and operation inventory
```

Needed deltas before lowering is accepted:

```text
- semantic vc4.resource emitted by VC4KernelToSSAVC4
- runtime vpm_base_row and semaphore_base builtins
- scalar f32 uniform support
- scalar i1 condition plan support
- predicate/general mask lowering
- full VPM allocator with hidden staging rows
- hardware-derived VPM/VDR/VDW lower-half schema
- vdw_store_fragment staging and preserve fallback
- masked TMU load
- i32 imul32 with mul24 fast path/fallback
- cooperative barrier through semaphore_base
- full hardware fixture matrix
```

---

## Acceptance gate summary

Corrected surface lock requires:

```text
ninja -C compiler/build vc4-opt
run_lit -sv compiler/test/Dialect/VC4Kernel
run_lit -sv compiler/test/Conversion/VC4KernelToSSAVC4
ninja -C compiler/build check-vc4
git diff --check
static scans show no lane_id/block_id/thread_id/direct-vc4/vc4tile in normative source
```

Full Stage 1 acceptance requires the above plus:

```text
ninja -C compiler/build vc4-codegen
SSAVC4 and SSAVC4ToVC4 lit suites
CodeGen lit suites
fresh hardware fixture matrix
CPU-reference comparison
sentinel/zero-fill checks
anti-shortcut scans
```

---

## Suggested first three Codex prompt themes

1. **Spec correction docs:** replace the old strict spec and update nearby docs, no code changes.
2. **Correct VC4Kernel surface:** delete lane_id/block_id/resource dict, update ODS/verifier/tests.
3. **Semantic resource/lower-half schema:** introduce computed `vc4.resource`, runtime builtins, and lower-half mode enums in SSAVC4/VC4 tests.

Only after these three should implementation proceed into predicate lowering, memory path lowering, and hardware fixtures.
