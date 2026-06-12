// RUN: vc4-opt %s --vc4-verify-value-surface | FileCheck %s

builtin.module {
  // CHECK-LABEL: func.func @f16_load_extf_f32
  func.func @f16_load_extf_f32(
      %in: memref<?xf16, #vc4value.global> {vc4value.arg_name = "in", vc4value.direction = "in", vc4value.shape_args = ["n"]},
      %out: memref<?xf32, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out", vc4value.shape_args = ["n"]},
      %n: index {vc4value.arg_name = "n", vc4value.scalar_role = "extent"})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    %pid = vc4value.program_id {axis = 0 : i32} : index
    %c16 = arith.constant 16 : index
    %base = arith.muli %pid, %c16 : index
    %remaining = arith.subi %n, %base : index
    // CHECK: vector.create_mask
    %mask = vector.create_mask %remaining : vector<16xi1>
    %zero_h = arith.constant 0.000000e+00 : f16
    // CHECK: vector.transfer_read
    %loaded = vector.transfer_read %in[%base], %zero_h, %mask {in_bounds = [true]} : memref<?xf16, #vc4value.global>, vector<16xf16>
    // CHECK: arith.extf
    %wide = arith.extf %loaded : vector<16xf16> to vector<16xf32>
    %one = arith.constant dense<1.000000e+00> : vector<16xf32>
    // CHECK: arith.addf
    %sum = arith.addf %wide, %one : vector<16xf32>
    vector.transfer_write %sum, %out[%base], %mask {in_bounds = [true]} : vector<16xf32>, memref<?xf32, #vc4value.global>
    return
  }
}
