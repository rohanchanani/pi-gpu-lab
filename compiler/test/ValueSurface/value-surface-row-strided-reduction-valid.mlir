// RUN: vc4-opt %s --vc4-verify-value-surface | FileCheck %s

builtin.module {
  // CHECK-LABEL: func.func @row_strided_reduction
  func.func @row_strided_reduction(
      %in: memref<?xf32, #vc4value.global> {vc4value.arg_name = "in", vc4value.direction = "in", vc4value.shape_args = ["total"]},
      %out: memref<?xf32, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out", vc4value.shape_args = ["rows"]},
      %total: index {vc4value.arg_name = "total", vc4value.scalar_role = "extent"},
      %rows: index {vc4value.arg_name = "rows", vc4value.scalar_role = "extent"},
      %row: index {vc4value.arg_name = "row", vc4value.scalar_role = "value"},
      %stride: index {vc4value.arg_name = "stride", vc4value.scalar_role = "stride"},
      %ncols: index {vc4value.arg_name = "ncols", vc4value.scalar_role = "extent"})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                  vc4value.fp_domain = "finite",
                  vc4value.reduction_policy = "finite_tree"} {
    %pid = vc4value.program_id {axis = 0 : i32} : index
    %c16 = arith.constant 16 : index
    %row_base = arith.muli %row, %stride : index
    %col_base = arith.muli %pid, %c16 : index
    %idx = arith.addi %row_base, %col_base : index
    %remaining = arith.subi %ncols, %col_base : index
    %mask = vector.create_mask %remaining : vector<16xi1>
    %zero = arith.constant 0.000000e+00 : f32
    // CHECK: vector.transfer_read
    %v = vector.transfer_read %in[%idx], %zero, %mask {in_bounds = [true]} : memref<?xf32, #vc4value.global>, vector<16xf32>
    // CHECK: vector.reduction
    %sum = vector.reduction <add>, %v : vector<16xf32> into f32
    // CHECK: memref.store
    memref.store %sum, %out[%row] : memref<?xf32, #vc4value.global>
    return
  }
}
