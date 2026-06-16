// REQUIRES: vc4-triton-cpp-frontend

// RUN: %vc4_triton_opt "%vc4_repo_root/examples/triton/phase17_online_softmax_state/generated/mixed_ttir_online_attention_apply_axes_mask_cf_f16_storage_b16.ttir.mlir" --convert-triton-to-vc4-value -o - | FileCheck %s --implicit-check-not='tt.' --implicit-check-not='ttg.' --implicit-check-not='triton_gpu.' --implicit-check-not='gpu.' --implicit-check-not='nvgpu.' --implicit-check-not='nvvm.' --implicit-check-not='vc4kernel.' --implicit-check-not='ssavc4.' --implicit-check-not='vc4.qpu'

// CHECK-LABEL: func.func @mixed_ttir_online_attention_apply_axes_mask_cf_f16_storage_b16_kernel
// CHECK-SAME: memref<?xf16, #vc4value.global>
// CHECK-SAME: memref<?xf16, #vc4value.global>
// CHECK-SAME: vc4value.grid_rank = 3
// CHECK-SAME: vc4value.kernel
// CHECK: vc4value.program_id {axis = 0 : i32}
// CHECK: vc4value.program_id {axis = 1 : i32}
// CHECK: vc4value.program_id {axis = 2 : i32}
// CHECK: scf.if
// CHECK: scf.for {{.*}} iter_args
// CHECK-SAME: -> (f32, f32, f32)
// CHECK: vector.transfer_read {{.*}} vector<16xf16>
// CHECK: arith.extf {{.*}} vector<16xf16> to vector<16xf32>
// CHECK: vector.broadcast %{{.*}} : f32 to vector<16xf32>
// CHECK: vector.reduction <maxnumf>
// CHECK: math.exp {{.*}}vc4value.math_policy = "approx_sfu"
// CHECK: vector.transfer_read {{.*}} vector<16xf16>
// CHECK: arith.extf {{.*}} vector<16xf16> to vector<16xf32>
// CHECK: vector.reduction <add>
// CHECK: scf.yield
// CHECK: memref.store
