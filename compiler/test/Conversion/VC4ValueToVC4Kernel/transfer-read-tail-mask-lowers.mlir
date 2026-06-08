// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s

func.func @read_tail(%in: memref<64xi32, #vc4value.global> {vc4value.arg_name = "in", vc4value.direction = "in"},
                     %base: index {vc4value.arg_name = "base"},
                     %n: index {vc4value.arg_name = "n"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %zero = arith.constant 0 : i32
  %mask = vector.create_mask %n : vector<16xi1>
  %v = vector.transfer_read %in[%base], %zero, %mask {in_bounds = [true]} : memref<64xi32, #vc4value.global>, vector<16xi32>
  return
}

// CHECK-LABEL: vc4kernel.kernel @read_tail
// CHECK: vc4kernel.pred.tail
// CHECK-SAME: : i32, i32 -> <16>
// CHECK: vc4kernel.fragment_select
// CHECK-SAME: vector<16xi32>, vector<16xi32> -> vector<16xi32>
// CHECK: vc4kernel.tmu_load_fragment
// CHECK-SAME: inactive_load = #vc4kernel.inactive_load<zero>
// CHECK-SAME: memory_path = #vc4kernel.memory_path<tmu_global_read>
// CHECK-SAME: : i32, vector<16xi32>, <16>, i32 -> vector<16xi32>
// CHECK-NOT: func.func
// CHECK-NOT: vc4value.
// CHECK-NOT: vector.
// CHECK-NOT: memref.
// CHECK-NOT: scf.
// CHECK-NOT: tensor.
// CHECK-NOT: linalg.
// CHECK-NOT: gpu.
// CHECK-NOT: tt.
