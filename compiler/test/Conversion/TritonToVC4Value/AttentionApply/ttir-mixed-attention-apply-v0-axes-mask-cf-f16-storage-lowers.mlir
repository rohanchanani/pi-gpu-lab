// REQUIRES: vc4-triton-cpp-frontend

// RUN: %vc4_triton_opt "%vc4_repo_root/examples/triton/phase16_attention_apply_v0/generated/mixed_ttir_attention_apply_v0_axes_mask_cf_f16_storage_b16.ttir.mlir" --convert-triton-to-vc4-value -o - | FileCheck %s --implicit-check-not='tt.' --implicit-check-not='ttg.' --implicit-check-not='triton_gpu.' --implicit-check-not='gpu.' --implicit-check-not='nvgpu.' --implicit-check-not='nvvm.' --implicit-check-not='vc4kernel.' --implicit-check-not='ssavc4.' --implicit-check-not='vc4.qpu'

// CHECK-LABEL: func.func @mixed_ttir_attention_apply_v0_axes_mask_cf_f16_storage_b16_kernel
// CHECK-SAME: memref<?xf16, #vc4value.global>
// CHECK-SAME: memref<?xf32, #vc4value.global>
// CHECK-SAME: vc4value.math_policy = "approx_sfu"
// CHECK: vc4value.program_id {axis = 0 : i32}
// CHECK: vc4value.program_id {axis = 1 : i32}
// CHECK: scf.if
// CHECK: vector.reduction <maxnumf>
// CHECK: math.exp {{.*}}vc4value.math_policy = "approx_sfu"
// CHECK: vector.reduction <add>
// CHECK: vector.transfer_read
// CHECK: arith.extf
// CHECK: arith.mulf
// CHECK: memref.store
