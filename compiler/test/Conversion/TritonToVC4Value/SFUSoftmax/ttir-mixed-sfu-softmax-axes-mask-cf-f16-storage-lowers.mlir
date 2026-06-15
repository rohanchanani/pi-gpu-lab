// REQUIRES: vc4-has-triton-cpp-frontend
// RUN: %vc4_triton_opt "%vc4_repo_root/examples/triton/phase15_sfu_softmax/generated/mixed_ttir_sfu_softmax_axes_mask_cf_f16_storage_b16.ttir.mlir" --convert-triton-to-vc4-value -o - | FileCheck %s --check-prefix=VALUE --implicit-check-not='tt.' --implicit-check-not='ttg.' --implicit-check-not='vc4kernel.' --implicit-check-not='ssavc4.' --implicit-check-not='builtin.unrealized_conversion_cast' --implicit-check-not='memref.load'
// RUN: %vc4_triton_opt "%vc4_repo_root/examples/triton/phase15_sfu_softmax/generated/mixed_ttir_sfu_softmax_axes_mask_cf_f16_storage_b16.ttir.mlir" --convert-triton-to-vc4-value -o %t.value.mlir
// RUN: %vc4_opt_triton %t.value.mlir --vc4-verify-value-surface --convert-scf-to-cf --convert-vc4-value-to-vc4kernel --verify-vc4kernel -o - | FileCheck %s --check-prefix=VC4KERNEL --implicit-check-not='math.exp' --implicit-check-not='arith.divf' --implicit-check-not='vector.reduction'
// RUN: %vc4_opt_triton %t.value.mlir --vc4-verify-value-surface --convert-scf-to-cf --convert-vc4-value-to-vc4kernel --verify-vc4kernel -o %t.vc4kernel.mlir
// RUN: %vc4_opt_triton %t.vc4kernel.mlir --convert-vc4kernel-to-ssavc4 -o %t.ssavc4.mlir
// RUN: %vc4_opt_triton %t.ssavc4.mlir --convert-ssavc4-to-vc4 --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o %t.vc4.mlir

// VALUE-LABEL: func.func @mixed_ttir_sfu_softmax_axes_mask_cf_f16_storage_b16
// VALUE-SAME: vc4value.arg_name = "scale"
// VALUE-SAME: vc4value.grid_rank = 2
// VALUE-SAME: vc4value.math_policy = "approx_sfu"
// VALUE-SAME: vc4value.max_policy = "finite"
// VALUE-SAME: vc4value.reduction_policy = "finite_tree"
// VALUE: vc4value.program_id
// VALUE-SAME: axis = 0
// VALUE: vc4value.program_id
// VALUE-SAME: axis = 1
// VALUE: vector.transfer_read
// VALUE-SAME: memref<?xf16, #vc4value.global>, vector<16xf16>
// VALUE: arith.extf
// VALUE: vector.broadcast
// VALUE-SAME: f32 to vector<16xf32>
// VALUE: scf.if
// VALUE: vector.reduction <maxnumf>
// VALUE: math.exp
// VALUE-SAME: vc4value.fp_domain = "finite"
// VALUE-SAME: vc4value.math_policy = "approx_sfu"
// VALUE: vector.reduction <add>
// VALUE: arith.divf
// VALUE-SAME: vc4value.fp_domain = "finite"
// VALUE-SAME: vc4value.math_policy = "approx_sfu"
// VALUE: memref.store

// VC4KERNEL-LABEL: vc4kernel.kernel @mixed_ttir_sfu_softmax_axes_mask_cf_f16_storage_b16
// VC4KERNEL: vc4kernel.program_id
// VC4KERNEL-SAME: axis = 0
// VC4KERNEL: vc4kernel.program_id
// VC4KERNEL-SAME: axis = 1
// VC4KERNEL: vc4kernel.vdr_load_rect_to_vpm
// VC4KERNEL: vc4kernel.fragment_unpack
// VC4KERNEL-SAME: source = #vc4kernel.subword_type<f16>
// VC4KERNEL: vc4kernel.splat
// VC4KERNEL: vc4kernel.fragment_reduce
// VC4KERNEL-SAME: kind = #vc4kernel.reduce<fmax>
// VC4KERNEL: 1.44269502
// VC4KERNEL-NEXT: vc4kernel.fragment_alu.mul
// VC4KERNEL-NEXT: vc4kernel.fragment_sfu
// VC4KERNEL-SAME: kind = #vc4kernel.sfu_kind<exp>
// VC4KERNEL: vc4kernel.fragment_reduce
// VC4KERNEL-SAME: kind = #vc4kernel.reduce<add>
// VC4KERNEL-NEXT: vc4kernel.fragment_sfu
// VC4KERNEL-SAME: kind = #vc4kernel.sfu_kind<recip>
