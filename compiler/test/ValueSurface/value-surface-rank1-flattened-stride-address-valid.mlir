// RUN: vc4-opt %s --vc4-verify-value-surface | FileCheck %s

builtin.module {
  // CHECK-LABEL: func.func @rank1_flattened_stride_address
  func.func @rank1_flattened_stride_address(
      %in: memref<?xf32, #vc4value.global> {vc4value.arg_name = "in", vc4value.direction = "in", vc4value.shape_args = ["total"]},
      %out: memref<?xf32, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out", vc4value.shape_args = ["total"]},
      %total: index {vc4value.arg_name = "total", vc4value.scalar_role = "extent"},
      %row: index {vc4value.arg_name = "row", vc4value.scalar_role = "value"},
      %stride: index {vc4value.arg_name = "stride", vc4value.scalar_role = "stride"})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    %pid = vc4value.program_id {axis = 0 : i32} : index
    %c16 = arith.constant 16 : index
    %row_base = arith.muli %row, %stride : index
    %col_base = arith.muli %pid, %c16 : index
    %idx = arith.addi %row_base, %col_base : index
    %remaining = arith.subi %total, %idx : index
    // CHECK: vector.create_mask
    %mask = vector.create_mask %remaining : vector<16xi1>
    %zero = arith.constant 0.000000e+00 : f32
    // CHECK: vector.transfer_read
    %v = vector.transfer_read %in[%idx], %zero, %mask {in_bounds = [true]} : memref<?xf32, #vc4value.global>, vector<16xf32>
    // CHECK: vector.transfer_write
    vector.transfer_write %v, %out[%idx], %mask {in_bounds = [true]} : vector<16xf32>, memref<?xf32, #vc4value.global>
    func.return
  }
}
