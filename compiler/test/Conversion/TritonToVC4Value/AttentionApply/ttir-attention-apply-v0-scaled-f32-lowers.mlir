// REQUIRES: vc4-triton-cpp-frontend

// RUN: %vc4_triton_opt "%vc4_repo_root/examples/triton/phase16_attention_apply_v0/generated/ttir_attention_apply_v0_scaled_f32_b16.ttir.mlir" --convert-triton-to-vc4-value -o - | FileCheck %s --implicit-check-not='tt.' --implicit-check-not='ttg.' --implicit-check-not='triton_gpu.' --implicit-check-not='gpu.' --implicit-check-not='nvgpu.' --implicit-check-not='nvvm.' --implicit-check-not='vc4kernel.' --implicit-check-not='ssavc4.' --implicit-check-not='vc4.qpu'

// CHECK-LABEL: func.func @ttir_attention_apply_v0_scaled_f32_b16_kernel
// CHECK-SAME: %{{.*}}: f32
// CHECK-SAME: vc4value.kernel
// CHECK-SAME: vc4value.math_policy = "approx_sfu"
// CHECK: vector.transfer_read
// CHECK: vector.broadcast %{{.*}} : f32 to vector<16xf32>
// CHECK: arith.mulf
// CHECK: vector.reduction <maxnumf>
// CHECK: math.exp {{.*}}vc4value.math_policy = "approx_sfu"
// CHECK: vector.reduction <add>
// CHECK: vector.transfer_read
// CHECK: arith.mulf
// CHECK: vector.reduction <add>
// CHECK: memref.store
