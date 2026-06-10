// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s

func.func @grid2_linear_address(
    %out: memref<256xi32, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out"},
    %n: index {vc4value.arg_name = "n"})
    attributes {vc4value.kernel, vc4value.grid_rank = 2 : i32} {
  %pid0 = vc4value.program_id {axis = 0 : i32} : index
  %pid1 = vc4value.program_id {axis = 1 : i32} : index
  %nx = vc4value.num_programs {axis = 0 : i32} : index
  %c16 = arith.constant 16 : index
  %row_base = arith.muli %pid1, %nx : index
  %block_id = arith.addi %row_base, %pid0 : index
  %idx = arith.muli %block_id, %c16 : index
  %mask = vector.create_mask %n : vector<16xi1>
  %value = arith.constant dense<7> : vector<16xi32>
  vector.transfer_write %value, %out[%idx], %mask {in_bounds = [true]} : vector<16xi32>, memref<256xi32, #vc4value.global>
  return
}

// CHECK-LABEL: vc4kernel.kernel @grid2_linear_address
// CHECK: vc4kernel.program_id
// CHECK-SAME: axis = 0
// CHECK: vc4kernel.program_id
// CHECK-SAME: axis = 1
// CHECK: vc4kernel.num_programs
// CHECK-SAME: axis = 0
// CHECK: vc4kernel.pred.tail
// CHECK: vc4kernel.vdw_store_fragment
// CHECK-NOT: func.func
// CHECK-NOT: vc4value.
// CHECK-NOT: vector.
// CHECK-NOT: memref.
// CHECK-NOT: tt.
