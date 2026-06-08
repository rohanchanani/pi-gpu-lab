// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s

func.func @read_f32_full(%in: memref<64xf32, #vc4value.global> {vc4value.arg_name = "in", vc4value.direction = "in"},
                         %base: index {vc4value.arg_name = "base"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %zero = arith.constant 0.000000e+00 : f32
  %v = vector.transfer_read %in[%base], %zero {in_bounds = [true]} : memref<64xf32, #vc4value.global>, vector<16xf32>
  return
}

// CHECK-LABEL: vc4kernel.kernel @read_f32_full
// CHECK: vc4kernel.pred.full
// CHECK: vc4kernel.tmu_load_fragment
// CHECK-SAME: coherency = #vc4kernel.coherency<readonly_tmu>
// CHECK-SAME: inactive_load = #vc4kernel.inactive_load<zero>
// CHECK-SAME: memory_path = #vc4kernel.memory_path<tmu_global_read>
// CHECK-SAME: : i32, vector<16xi32>, <16>, i32 -> vector<16xf32>
// CHECK-NOT: func.func
// CHECK-NOT: vc4value.
// CHECK-NOT: vector.
// CHECK-NOT: memref.
// CHECK-NOT: scf.
// CHECK-NOT: tensor.
// CHECK-NOT: linalg.
// CHECK-NOT: gpu.
// CHECK-NOT: tt.
