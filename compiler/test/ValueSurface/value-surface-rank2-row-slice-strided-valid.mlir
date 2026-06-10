// RUN: vc4-opt %s --vc4-verify-value-surface | FileCheck %s

builtin.module {
  // CHECK-LABEL: func.func @rank2_row_slice_strided
  func.func @rank2_row_slice_strided(
      %in: memref<?x?xf32, strided<[?, 1], offset: 0>, #vc4value.global> {vc4value.arg_name = "in", vc4value.direction = "in", vc4value.shape_args = ["rows", "cols"], vc4value.stride_args = ["row_stride"]},
      %out: memref<?x?xf32, strided<[?, 1], offset: 0>, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out", vc4value.shape_args = ["rows", "cols"], vc4value.stride_args = ["row_stride"]},
      %rows: index {vc4value.arg_name = "rows", vc4value.scalar_role = "extent"},
      %cols: index {vc4value.arg_name = "cols", vc4value.scalar_role = "extent"},
      %row_stride: index {vc4value.arg_name = "row_stride", vc4value.scalar_role = "stride"})
      attributes {vc4value.kernel, vc4value.grid_rank = 2 : i32} {
    %pid_col = vc4value.program_id {axis = 0 : i32} : index
    %row = vc4value.program_id {axis = 1 : i32} : index
    %c16 = arith.constant 16 : index
    %col = arith.muli %pid_col, %c16 : index
    %remaining = arith.subi %cols, %col : index
    // CHECK: vector.create_mask
    %mask = vector.create_mask %remaining : vector<16xi1>
    %zero = arith.constant 0.000000e+00 : f32
    // CHECK: vector.transfer_read
    %v = vector.transfer_read %in[%row, %col], %zero, %mask {in_bounds = [true]} : memref<?x?xf32, strided<[?, 1], offset: 0>, #vc4value.global>, vector<16xf32>
    // CHECK: vector.transfer_write
    vector.transfer_write %v, %out[%row, %col], %mask {in_bounds = [true]} : vector<16xf32>, memref<?x?xf32, strided<[?, 1], offset: 0>, #vc4value.global>
    func.return
  }
}
