// RUN: vc4-opt %s --vc4-verify-value-surface | FileCheck %s

builtin.module {
  // CHECK-LABEL: func.func @compute_mask_select
  func.func @compute_mask_select(
      %in: memref<?xf32, #vc4value.global> {vc4value.arg_name = "in", vc4value.direction = "in", vc4value.shape_args = ["n"]},
      %out: memref<?xf32, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out", vc4value.shape_args = ["n"]},
      %n: index {vc4value.arg_name = "n"},
      %threshold: f32 {vc4value.arg_name = "threshold"})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    %pid = vc4value.program_id {axis = 0 : i32} : index
    %c16 = arith.constant 16 : index
    %base = arith.muli %pid, %c16 : index
    %remaining = arith.subi %n, %base : index
    %mask = vector.create_mask %remaining : vector<16xi1>
    %zero = arith.constant 0.000000e+00 : f32
    %v = vector.transfer_read %in[%base], %zero, %mask {in_bounds = [true]} : memref<?xf32, #vc4value.global>, vector<16xf32>
    %threshold_v = vector.broadcast %threshold : f32 to vector<16xf32>
    // CHECK: arith.cmpf
    %compute_mask = arith.cmpf olt, %v, %threshold_v : vector<16xf32>
    %zeros = arith.constant dense<0.000000e+00> : vector<16xf32>
    // CHECK: arith.select
    %selected = arith.select %compute_mask, %v, %zeros : vector<16xi1>, vector<16xf32>
    vector.transfer_write %selected, %out[%base], %mask {in_bounds = [true]} : vector<16xf32>, memref<?xf32, #vc4value.global>
    return
  }
}
