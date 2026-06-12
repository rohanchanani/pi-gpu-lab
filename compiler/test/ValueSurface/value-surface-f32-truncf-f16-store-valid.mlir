// RUN: vc4-opt %s --vc4-verify-value-surface | FileCheck %s

builtin.module {
  // CHECK-LABEL: func.func @f32_truncf_f16_store
  func.func @f32_truncf_f16_store(
      %in: memref<?xf32, #vc4value.global> {vc4value.arg_name = "in", vc4value.direction = "in", vc4value.shape_args = ["n"]},
      %out: memref<?xf16, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out", vc4value.shape_args = ["n"]},
      %n: index {vc4value.arg_name = "n", vc4value.scalar_role = "extent"})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                  vc4value.f16_storage_policy = "finite"} {
    %pid = vc4value.program_id {axis = 0 : i32} : index
    %c16 = arith.constant 16 : index
    %base = arith.muli %pid, %c16 : index
    %remaining = arith.subi %n, %base : index
    %mask = vector.create_mask %remaining : vector<16xi1>
    %zero = arith.constant 0.000000e+00 : f32
    %loaded = vector.transfer_read %in[%base], %zero, %mask {in_bounds = [true]} : memref<?xf32, #vc4value.global>, vector<16xf32>
    %bias = arith.constant dense<2.500000e-01> : vector<16xf32>
    // CHECK: arith.addf
    %sum = arith.addf %loaded, %bias : vector<16xf32>
    // CHECK: arith.truncf
    %narrow = arith.truncf %sum : vector<16xf32> to vector<16xf16>
    // CHECK: vector.transfer_write
    vector.transfer_write %narrow, %out[%base], %mask {in_bounds = [true]} : vector<16xf16>, memref<?xf16, #vc4value.global>
    return
  }
}
