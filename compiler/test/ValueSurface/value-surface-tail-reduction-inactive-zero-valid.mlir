// RUN: vc4-opt %s --vc4-verify-value-surface | FileCheck %s

builtin.module {
  // CHECK-LABEL: func.func @tail_reduction_inactive_zero
  func.func @tail_reduction_inactive_zero(
      %in: memref<?xf32, #vc4value.global> {vc4value.arg_name = "in", vc4value.direction = "in", vc4value.shape_args = ["n"]},
      %out: memref<?xf32, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out", vc4value.shape_args = ["blocks"]},
      %n: index {vc4value.arg_name = "n", vc4value.scalar_role = "extent"},
      %blocks: index {vc4value.arg_name = "blocks", vc4value.scalar_role = "extent"})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                  vc4value.fp_domain = "finite",
                  vc4value.reduction_policy = "finite_tree"} {
    %pid = vc4value.program_id {axis = 0 : i32} : index
    %c16 = arith.constant 16 : index
    %base = arith.muli %pid, %c16 : index
    %remaining = arith.subi %n, %base : index
    // CHECK: vector.create_mask
    %mask = vector.create_mask %remaining : vector<16xi1>
    %zero = arith.constant 0.000000e+00 : f32
    // CHECK: vector.transfer_read
    %v = vector.transfer_read %in[%base], %zero, %mask {in_bounds = [true]} : memref<?xf32, #vc4value.global>, vector<16xf32>
    // CHECK: vector.reduction
    %sum = vector.reduction <add>, %v : vector<16xf32> into f32
    return
  }
}
