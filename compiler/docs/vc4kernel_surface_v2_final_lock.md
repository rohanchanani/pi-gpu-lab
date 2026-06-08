# VC4Kernel Surface v2 Final Lock

This page is the post-P13 handoff record for the locked VC4Kernel target-kernel surface. It is source-controlled context for producer value-layer work above VC4Kernel.

## Final Stack Boundary

The only accepted compiler path is:

```text
standard value layer -> vc4kernel -> ssavc4 -> scheduled vc4 -> artifacts/runtime/hardware
```

VC4Kernel is the stable target-kernel contract above SSAVC4. It may use MLIR vector types such as `vector<16xi32>` and `vector<16xf32>` as fragment carrier types, but verified VC4Kernel hardware fixtures do not contain vector dialect operations or other producer dialect operations.

VC4Kernel never lowers straight to scheduled VC4; SSAVC4 remains the required lower-half boundary. The retired tile DSL is not part of this stack.

## Final Feature Categories

| Category | Meaning |
| --- | --- |
| `ACCEPTED_HARDWARE_PROVEN` | Executable VC4Kernel semantics with verifier, conversion, lower-half, hardware, mixed-suite, and claim-audit evidence. |
| `INTERNAL_ONLY` | Resource, audit, manifest, artifact, or runtime contract that does not add executable VC4Kernel kernel semantics. |
| `DETERMINISTIC_REJECT_HARDWARE_FORBIDDEN` | Rejected because VC4 hardware does not provide the requested compute-kernel mode. |
| `DETERMINISTIC_REJECT_STATIC_SURFACE_POLICY` | Rejected because VC4Kernel Surface v2 requires a static spelling, explicit policy, or explicit mode. |
| `DETERMINISTIC_REJECT_NOT_MEANINGFUL` | Rejected because the requested field or mode has no meaning for that operation. |
| `OUT_OF_SCOPE_NON_COMPUTE_HARDWARE` | Fixed-function graphics, texture filtering, tile-buffer, varying, or display hardware outside the VC4Kernel compute target. |

No hardware-backed VC4Kernel compute-kernel feature is outside these categories.

## Accepted And Rejected Modes

| Surface row | Final category | Proof or reject path |
| --- | --- | --- |
| Kernel ABI, return, program id, number of programs, warp identity, lane identity | `ACCEPTED_HARDWARE_PROVEN` | Verifier, conversion, launch identity fixtures, mixed acceptance, support matrix. |
| Predicates: full, empty, tail, rectangular, and/or/not, any/all | `ACCEPTED_HARDWARE_PROVEN` | Predicate verifier/conversion tests and mixed fixtures with checked output. |
| Fragment const, bitcast, splat, select | `ACCEPTED_HARDWARE_PROVEN` | P2 proof links, conversion tests, mixed claims. |
| Add-pipe and mul-pipe fragment ALU opcodes | `ACCEPTED_HARDWARE_PROVEN` | P1 proof links, hardware fixtures, mixed claims. |
| i32 and finite f32 comparisons | `ACCEPTED_HARDWARE_PROVEN` | P3 proof links; f32 comparisons require finite policy. |
| i32 and finite f32 reductions | `ACCEPTED_HARDWARE_PROVEN` | P4 proof links; f32 reductions require finite-tree policy. |
| Scalar arithmetic, control, and address subset | `ACCEPTED_HARDWARE_PROVEN` | P5 proof links and mixed scalar/control fixture. |
| TMU safe-offset inactive loads | `ACCEPTED_HARDWARE_PROVEN` | Explicit `safe_offset` and inactive-zero policy; old implicit form rejected. |
| VDW register-fragment preserve stores | `ACCEPTED_HARDWARE_PROVEN` | P8 preserve-store proof and mixed tail/guard checks. |
| VPM allocation, QPU read/write, w32/subword/dynamic coords/selectors | `ACCEPTED_HARDWARE_PROVEN` | P9/P12 isolated fixtures, conversion tests, mixed selector fixtures. |
| VDR to VPM w32/subword/dynamic coords/selectors | `ACCEPTED_HARDWARE_PROVEN` | P9/P12 isolated fixtures and mixed VPM pipeline fixtures. |
| VDW from VPM w32/subword/dynamic coords/selectors | `ACCEPTED_HARDWARE_PROVEN` | P9/P12 isolated fixtures and mixed memory-output fixtures. |
| Pack/unpack and f16 storage conversion | `ACCEPTED_HARDWARE_PROVEN` | P9 and P13 f16 storage fixtures; native f16 arithmetic is rejected. |
| Approximate SFU policy | `ACCEPTED_HARDWARE_PROVEN` | Explicit approximate policy only; exact/default math is rejected. |
| Dynamic rotate and rotate-derived composites | `ACCEPTED_HARDWARE_PROVEN` | P11 dynamic rotate proof and mixed swizzle/scan fixtures. |
| Barriers, semaphores, cooperative metadata | `ACCEPTED_HARDWARE_PROVEN` | Cooperative barrier fixtures and runtime resource checks. |
| Runtime resource metadata and libpi descriptors | `ACCEPTED_HARDWARE_PROVEN` | Artifact audit, generated C, and mixed resource metadata. |
| Artifact emission, manifest JSON, generated C/QASM | `ACCEPTED_HARDWARE_PROVEN` | Final artifact boundary audit and lit coverage. |
| Producer dialect operations inside VC4Kernel | `DETERMINISTIC_REJECT_STATIC_SURFACE_POLICY` | Verifier reject corpus. |
| Removed fragment special spellings | `DETERMINISTIC_REJECT_STATIC_SURFACE_POLICY` | Verifier reject corpus and audit. |
| Old TMU signature or safe-address inference | `DETERMINISTIC_REJECT_STATIC_SURFACE_POLICY` | Verifier reject corpus and audit. |
| Sparse VDW stores | `DETERMINISTIC_REJECT_STATIC_SURFACE_POLICY` | Verifier reject corpus and support matrix. |
| DMA laned subword modes | `DETERMINISTIC_REJECT_HARDWARE_FORBIDDEN` | P9/P12 reject corpus. |
| Dynamic orientation, width, or subword mode attrs | `DETERMINISTIC_REJECT_STATIC_SURFACE_POLICY` | P12 selector reject corpus. |
| Vector or lane-varying coordinate operands | `DETERMINISTIC_REJECT_STATIC_SURFACE_POLICY` | P12 selector reject corpus. |
| Horizontal w32 dynamic word-X | `DETERMINISTIC_REJECT_NOT_MEANINGFUL` | P12 verifier reject. |
| Arbitrary shuffle or arbitrary permutation | `DETERMINISTIC_REJECT_STATIC_SURFACE_POLICY` | P11 reject corpus. |
| Texture filtering, cube maps, fixed-function graphics, tile-buffer modes | `OUT_OF_SCOPE_NON_COMPUTE_HARDWARE` | Support matrix classification. |

## P12 Dynamic Coordinate And Selector Contract

P12 locks row, word-X, and subword selector as separate coordinate fields. A dynamic subword selector is a byte/halfword selector inside an accepted packed mode. It is not dynamic width, dynamic subword mode, dynamic orientation, or dynamic layout.

Setup-field masks isolate hardware fields only. They are not modulo semantics and do not legalize out-of-range operands.

VPM QPU read/write selector semantics are separately proven for packed and laned modes and for horizontal and vertical modes. QPU normalized subword readback is distinct from raw DMA carrier placement.

VDR selector semantics are:

| Width/mode | Selector contract |
| --- | --- |
| w32 | No selector. |
| w16 packed | Selector `H` in `[0, 1]`; encodes `MODEW = 2 + H`. |
| w8 packed | Selector `B` in `[0, 3]`; encodes `MODEW = 4 + B`. |

VDW selector semantics are separately proved and are not inferred from VDR symmetry. VDW proof covers setup composition, preserve behavior, and memory output.

The final proof graph covers dynamic row, dynamic word-X, dynamic subword selector, VDR destination fields, VDW source fields, combined VDR/VPM/VDW roundtrip, runtime pitch/stride selector cases, double-buffered/ping-pong tiles, and spill-pressure cases.

## Final Mixed Suite

The final mixed hardware suite is the routine P13 and producer-layer gate:

| Fixture | Main role |
| --- | --- |
| `mixed_elementwise_tmu_alu_cmp_vdw_vc4kernel` | Elementwise TMU, ALU, comparison, VDW preserve. |
| `mixed_i32_control_reduce_scalar_address_vc4kernel` | Scalar control/address and i32 reductions. |
| `mixed_f32_reduce_tmu_loop_spill_vdw_vc4kernel` | f32 reductions, TMU loop, spill pressure, VDW preserve. |
| `mixed_blocked_gemv_vpm_fullstack_vc4kernel` | Blocked GEMV VPM pipeline. |
| `mixed_blocked_gemm_vpm_fullstack_vc4kernel` | Blocked GEMM VPM pipeline. |
| `mixed_vertical_rect_vpm_vdw_preserve_vc4kernel` | Vertical rectangular VPM/VDW preserve. |
| `mixed_cooperative_barrier_vpm_transpose_vc4kernel` | Cooperative barrier and VPM transpose. |
| `mixed_lower_half_spill_dma_branch_ssavc4` | Lower-half spill, DMA, branch layout. |
| `mixed_subword_vpm_pack_unpack_roundtrip_vc4kernel` | P9 subword and pack/unpack roundtrip. |
| `mixed_quantized_gemv_subword_vpm_vc4kernel` | Quantized GEMV subword VPM path. |
| `mixed_sfu_norm_reduce_vpm_vc4kernel` | SFU norm/reduce with VPM. |
| `mixed_sfu_activation_tmu_vdw_vc4kernel` | SFU activation with TMU and VDW. |
| `mixed_dynamic_rotate_reduction_scan_vc4kernel` | Dynamic rotate and reduction scan. |
| `mixed_shuffle_vpm_tile_swizzle_vc4kernel` | Rotate-derived VPM swizzle. |
| `dynamic_vpm_pingpong_coord_selector_loop_vc4kernel` | P12 dynamic ping-pong coordinate/selector loop. |
| `dynamic_vpm_pingpong_qpu_read_vc4kernel` | Checked dynamic ping-pong QPU readback. |
| `dynamic_vpm_double_buffered_subword_compute_vc4kernel` | Double-buffered dynamic subword compute. |
| `dynamic_vpm_coord_selector_forced_spill_vc4kernel` | Dynamic selector under spill pressure. |
| `mixed_dynamic_vpm_coordinates_rect_loop_vc4kernel` | Runtime rectangular loop with dynamic VPM/VDR/VDW fields. |
| `mixed_double_buffered_vpm_tiles_vc4kernel` | Double-buffered VPM tile acceptance. |
| `mixed_surface_lock_elementwise_full_vc4kernel` | P13 full-surface elementwise mix. |
| `mixed_surface_lock_vpm_pipeline_full_vc4kernel` | P13 full-surface VPM pipeline mix. |
| `mixed_surface_lock_dynamic_pingpong_qpu_compute_vc4kernel` | P13 checked dynamic ping-pong QPU compute. |
| `mixed_surface_lock_cooperative_barrier_full_vc4kernel` | P13 cooperative full-surface mix. |
| `mixed_f16_storage_conversion_vc4kernel` | f16 storage conversion plus f32 compute. |

Final P13h acceptance command:

```bash
VC4_CODEGEN_STATE_ROOT=.vc4_auto/codegen_p13h_final_mixed_acceptance \
  compiler/test/CodeGen/VC4Kernel/Hardware/MixedAcceptance/run_mixed_acceptance.sh \
  --manifest compiler/test/CodeGen/VC4Kernel/Hardware/MixedAcceptance/mixed_acceptance_manifest.json
```

The P13h run produced 25 `VC4_TEST_RESULT` lines, all `status=PASS`, with zero mismatch, sentinel, and launch-failure fields where present. The P13 full-surface fixtures emitted `saw_final_surface_mix=1`, and the checked ping-pong QPU fixture emitted `saw_checked_qpu_readback=1`.

## Fixture Claim Contract

Every mixed `saw_*`, `no_*`, or equivalent result field must have an entry in:

```text
compiler/test/CodeGen/VC4Kernel/Hardware/MixedAcceptance/mixed_fixture_claims.json
```

The entry must classify the claim as `CHECKED_OUTPUT`, `CHECKED_AUDIT`, or `PHASE_GUARD`. A mixed fixture may not claim a feature through dead IR, decorative operations, broad fixture names, status strings, or generated-output path checks. The claim audit is the source-controlled guard for that rule.

## Producer Value-Layer Handoff

Vector, memref, arith, math, scf, and cf lowering must treat VC4Kernel as a closed target-kernel contract. Producer lowering may create only accepted VC4Kernel operations and must use the deterministic reject rules above when source semantics do not fit the contract.

The first vector-layer work should start with a value-layer boundary/spec and handwritten value lowering to the locked VC4Kernel surface. It should not start from TTIR.

Triton direct-to-VC4Kernel remains forbidden. Triton lowering must enter through the standard value layer first, then lower to VC4Kernel through the same checked producer boundary.

No implementation package should be generated without post-P13 context from:

```bash
bash compiler/docs/scripts/collect_vc4_post_p13_context.sh
```

## Readiness Lines

```text
VC4KERNEL_SURFACE_V2_FINAL_LOCKED=YES
VC4KERNEL_MIXED_FIXTURE_CLAIMS_AUDITED=YES
VC4KERNEL_NO_DEAD_FEATURE_CLAIMS=YES
READY_FOR_VECTOR_SURFACE_DESIGN_AND_HANDWRITTEN_VECTOR_LOWERING=YES
READY_FOR_TRITON=NO
```
