// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s

func.func @write_tail(%out: memref<64xi32, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out"},
                      %base: index {vc4value.arg_name = "base"},
                      %n: index {vc4value.arg_name = "n"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %value = arith.constant dense<23> : vector<16xi32>
  %mask = vector.create_mask %n : vector<16xi1>
  vector.transfer_write %value, %out[%base], %mask {in_bounds = [true]} : vector<16xi32>, memref<64xi32, #vc4value.global>
  return
}

// CHECK-LABEL: vc4kernel.kernel @write_tail
// CHECK: vc4kernel.pred.tail
// CHECK-SAME: : i32, i32 -> <16>
// CHECK: vc4kernel.vdw_store_fragment
// CHECK-SAME: inactive_store = #vc4kernel.inactive_store<preserve>
// CHECK-SAME: memory_path = #vc4kernel.memory_path<vdw_global_store>
// CHECK-SAME: : i32, vector<16xi32>, vector<16xi32>, <16>
// CHECK-NOT: func.func
// CHECK-NOT: vc4value.
// CHECK-NOT: vector.
// CHECK-NOT: memref.
// CHECK-NOT: scf.
// CHECK-NOT: tensor.
// CHECK-NOT: linalg.
// CHECK-NOT: gpu.
// CHECK-NOT: tt.
