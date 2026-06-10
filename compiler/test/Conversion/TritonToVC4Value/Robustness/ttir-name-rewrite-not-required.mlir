// REQUIRES: vc4-has-triton-cpp-frontend

// RUN: %vc4_triton_opt %s --convert-triton-to-vc4-value -o - | FileCheck %s

// CHECK-LABEL: func.func @name_rewrite_not_required
// CHECK-SAME: vc4value.arg_name = "alpha"
// CHECK-SAME: vc4value.arg_name = "beta"
// CHECK-SAME: vc4value.shape_args = ["omega"]
// CHECK-SAME: %{{[^:]+}}: index {vc4value.arg_name = "omega"}
// CHECK: vector.transfer_read
// CHECK: vector.transfer_write

#loc = loc("name_rewrite_not_required")
module {
  tt.func public @name_rewrite_not_required(%a: !tt.ptr<f32> loc("alpha"(#loc)), %b: !tt.ptr<f32> loc("beta"(#loc)), %limit: i32 loc("omega"(#loc))) attributes {noinline = false} {
    %pid = tt.get_program_id x : i32
    %c16 = arith.constant 16 : i32
    %pid16 = arith.muli %pid, %c16 : i32
    %lanes = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %base = tt.splat %pid16 : i32 -> tensor<16xi32>
    %offsets = arith.addi %base, %lanes : tensor<16xi32>
    %bound = tt.splat %limit : i32 -> tensor<16xi32>
    %mask = arith.cmpi slt, %offsets, %bound : tensor<16xi32>
    %x_base = tt.splat %a : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %x_ptrs = tt.addptr %x_base, %offsets : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %zero = arith.constant dense<0.000000e+00> : tensor<16xf32>
    %value = tt.load %x_ptrs, %mask, %zero : tensor<16x!tt.ptr<f32>>
    %out_base = tt.splat %b : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %out_ptrs = tt.addptr %out_base, %offsets : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %out_ptrs, %value, %mask : tensor<16x!tt.ptr<f32>>
    tt.return
  } loc(#loc)
}
