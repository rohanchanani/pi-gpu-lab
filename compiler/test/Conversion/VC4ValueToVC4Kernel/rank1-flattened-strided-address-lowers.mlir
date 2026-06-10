// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s

func.func @rank1_flattened_strided_address_lowers(
    %in: memref<?xi32, #vc4value.global> {vc4value.arg_name = "in", vc4value.direction = "in", vc4value.shape_args = ["total"]},
    %out: memref<?xi32, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out", vc4value.shape_args = ["total"]},
    %total: index {vc4value.arg_name = "total", vc4value.scalar_role = "extent"},
    %lda: index {vc4value.arg_name = "lda", vc4value.scalar_role = "stride"})
    attributes {vc4value.kernel, vc4value.grid_rank = 2 : i32} {
  %pid_col = vc4value.program_id {axis = 0 : i32} : index
  %row = vc4value.program_id {axis = 1 : i32} : index
  %c16 = arith.constant 16 : index
  %col = arith.muli %pid_col, %c16 : index
  %row_base = arith.muli %row, %lda : index
  %idx = arith.addi %row_base, %col : index
  %remaining = arith.subi %total, %idx : index
  %mask = vector.create_mask %remaining : vector<16xi1>
  %zero = arith.constant 0 : i32
  %v = vector.transfer_read %in[%idx], %zero, %mask {in_bounds = [true]} : memref<?xi32, #vc4value.global>, vector<16xi32>
  vector.transfer_write %v, %out[%idx], %mask {in_bounds = [true]} : vector<16xi32>, memref<?xi32, #vc4value.global>
  return
}

// CHECK-LABEL: vc4kernel.kernel @rank1_flattened_strided_address_lowers
// CHECK: vc4kernel.program_id
// CHECK-SAME: axis = 0
// CHECK: vc4kernel.program_id
// CHECK-SAME: axis = 1
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: arith.shli
// CHECK: vc4kernel.fragment_const
// CHECK-SAME: dense<[0, 4, 8, 12
// CHECK: vc4kernel.tmu_load_fragment
// CHECK: vc4kernel.vdw_store_fragment
// CHECK-NOT: func.func
// CHECK-NOT: vc4value.
// CHECK-NOT: vector.
// CHECK-NOT: memref.
