// REQUIRES: vc4-triton-cpp-frontend

// RUN: %vc4_triton_opt "%vc4_repo_root/examples/triton/phase17_online_softmax_state/generated/ttir_online_softmax_normalizer_f32_b16.ttir.mlir" --convert-triton-to-vc4-value -o - | FileCheck %s --implicit-check-not='tt.' --implicit-check-not='ttg.' --implicit-check-not='triton_gpu.' --implicit-check-not='gpu.' --implicit-check-not='nvgpu.' --implicit-check-not='nvvm.' --implicit-check-not='vc4kernel.' --implicit-check-not='ssavc4.' --implicit-check-not='vc4.qpu'

// CHECK-LABEL: func.func @ttir_online_softmax_normalizer_f32_b16_kernel
// CHECK-SAME: vc4value.kernel
// CHECK-SAME: vc4value.math_policy = "approx_sfu"
// CHECK-SAME: vc4value.max_policy = "finite"
// CHECK-SAME: vc4value.reduction_policy = "finite_tree"
// CHECK: vc4value.program_id {axis = 0 : i32}
// CHECK: scf.for {{.*}} iter_args
// CHECK-SAME: -> (f32, f32)
// CHECK: vector.create_mask
// CHECK: vector.transfer_read
// CHECK: arith.select
// CHECK: vector.reduction <maxnumf>
// CHECK: arith.maxnumf {{.*}}vc4value.max_policy = "finite"
// CHECK: math.exp {{.*}}vc4value.math_policy = "approx_sfu"
// CHECK: math.exp {{.*}}vc4value.math_policy = "approx_sfu"
// CHECK: vector.reduction <add>
// CHECK: scf.yield
// CHECK: arith.divf {{.*}}vc4value.math_policy = "approx_sfu"
// CHECK: memref.store
