# VC4Kernel Surface v2 Decision Traceability Matrix

This traceability document points from each final Surface v2 decision to the canonical proof source. The machine-readable source of truth is `compiler/docs/vc4kernel_surface_v2_support_matrix.json`; this document records the human-readable grouping.

## Final Surface Groups

| Surface group | Final category | Proof source |
| --- | --- | --- |
| Kernel, return, ABI, program id, number of programs, warp identity, lane identity | `accepted_hardware_proven` | Support matrix row `baseline_kernel_return_launch_identity` |
| Predicate algebra: full, empty, tail, rect, and, or, not, any, all | `accepted_hardware_proven` | Support matrix row `baseline_predicate_algebra` |
| Fragment const, bitcast, splat, select | `accepted_hardware_proven` | Support matrix rows `p2_fragment_constants_splat`, `p2_fragment_bitcast`, `p3_fragment_select` |
| Add-pipe ALU opcodes | `accepted_hardware_proven` | Support matrix row `p1_general_fragment_add_alu` |
| Mul-pipe ALU opcodes | `accepted_hardware_proven` | Support matrix row `p1_general_fragment_mul_alu` |
| i32 and finite f32 comparisons | `accepted_hardware_proven` | Support matrix rows `p3_fragment_cmp_signed_i32`, `p10_finite_f32_cmp` |
| i32 and finite f32 reductions | `accepted_hardware_proven` | Support matrix rows `p4_fragment_reduce_general`, `p10_finite_f32_reduce` |
| Scalar arith/control/address subset | `accepted_hardware_proven` | Support matrix row `p5_scalar_arith_control_address_subset` |
| TMU safe-offset inactive loads | `accepted_hardware_proven` | Support matrix row `p7_tmu_safe_inactive_load_policy` |
| VDW register-fragment preserve stores | `accepted_hardware_proven` | Support matrix row `p8_vdw_fragment_store_preserve_full_tail_rect` |
| VPM alloc/QPU read/write w32, subword, dynamic coords/selectors | `accepted_hardware_proven` | Support matrix rows `baseline_vpm_vdr_vdw_rect_paths`, `p9_vpm_subword_modes`, `p12_dynamic_vpm_read_write_coordinates` |
| VDR to VPM w32, subword, dynamic coords/selectors | `accepted_hardware_proven` | Support matrix rows `p9_vdr_vdw_subword_modes`, `p12_dynamic_vdr_vdw_coordinates` |
| VDW from VPM w32, subword, dynamic coords/selectors | `accepted_hardware_proven` | Support matrix rows `p8_vdw_fragment_store_preserve_full_tail_rect`, `p9_vdr_vdw_subword_modes`, `p12_dynamic_vdr_vdw_coordinates` |
| Pack/unpack | `accepted_hardware_proven` | Support matrix rows `p9_pack_unpack_subword_ops`, `p9_vpm_subword_modes` |
| SFU approximate policy | `accepted_hardware_proven` | Support matrix row `p10_fastmath_approx_contract` |
| Dynamic rotate and rotate-derived shuffle policy | `accepted_hardware_proven` | Support matrix rows `p11_dynamic_rotate`, `p11_fragment_broadcast_lane_if_supported` |
| Barriers, semaphores, cooperative resource metadata | `accepted_hardware_proven` | Support matrix rows `p6_memory_resource_metadata`, `p8_5_mixed_acceptance_policy` |
| Runtime resource metadata, libpi descriptors, artifacts, generated C | `accepted_hardware_proven` or `internal_only` | Support matrix rows `p6_memory_resource_metadata`, `p13_surface_support_matrix_final` |

## Deterministic Reject Groups

| Reject group | Category | Proof source |
| --- | --- | --- |
| Removed fragment special-case spellings | `unsupported_source_layer_inside_vc4kernel` | Support matrix rows `p1_remove_legacy_fragment_add_sub_mul_shl`, `p4_remove_legacy_fragment_contract`, `p7_remove_legacy_tmu_signature_and_safe_inference` |
| Producer dialect operations inside verified VC4Kernel | `unsupported_source_layer_inside_vc4kernel` | Strict spec plus verifier lit proofs in support matrix rows |
| Dynamic orientation, width, subword mode attrs | `static_surface_policy` | Support matrix row `p12_dynamic_vpm_read_write_coordinates` |
| Vector or lane-varying coordinate operands | `static_surface_policy` | Support matrix rows `p12_dynamic_vpm_read_write_coordinates`, `p12_dynamic_vdr_vdw_coordinates` |
| QPU horizontal w32 dynamic word-X | `not_meaningful` | P12 verifier and audit proofs |
| DMA laned selector modes | `hardware_forbidden` | P12 verifier and audit proofs |
| Sparse VDW general-mask stores | `static_surface_policy` | Support matrix rows `p8_sparse_vdw_store_deterministic_reject`, `sparse_vdw_store_general_masks_reject` |
| Exact/default SFU math | `static_surface_policy` | Support matrix row `p10_exact_default_math_no_sfu` |
| Arbitrary shuffle/permutation | `static_surface_policy` | Support matrix row `p11_arbitrary_shuffle_permutation_reject` |
| Fixed-function graphics, tile-buffer color/Z/stencil, texture filtering, cube maps, varyings | `out_of_scope_non_compute_hardware` | Strict spec out-of-scope classification |

## P12 Selector Traceability

P12 proof rows keep row, word-X, and subword selector as separate fields. Dynamic subword selector is byte/halfword selection and does not alter static width, subword mode, or orientation. VDR raw carrier placement, QPU-normalized subword readback, VDW memory output, and mixed dataflow are different proof views.

The proof set covers QPU VPM dynamic row, dynamic word-X, dynamic subword selector for packed/laned horizontal/vertical modes; VDR dynamic destination row, word-X, and horizontal/vertical selector; VDW dynamic source row, word-X, and horizontal/vertical selector; VDR-to-VPM-to-VDW selector roundtrip; runtime pitch/stride selector cases; double-buffered ping-pong VPM tiles; and dynamic coordinate/selector use under spill pressure.

## Final Lock Conditions

The final lock is valid when:

- The matrix checker accepts only final status categories.
- The mixed fixture claim audit accepts all `saw_*` and `no_*` fields.
- The mixed manifest checker, P12 audit, relevant lit tests, `vc4-opt`, `vc4-codegen`, and `check-vc4` pass.
- The final mixed suite remains the hardware gate for producer-layer work that lowers into VC4Kernel.
