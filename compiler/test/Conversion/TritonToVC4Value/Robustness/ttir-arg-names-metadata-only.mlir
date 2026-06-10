// REQUIRES: vc4-has-triton-cpp-frontend

// RUN: %vc4_triton_opt %s --convert-triton-to-vc4-value -o - | FileCheck %s

// CHECK-LABEL: func.func @arg_names_metadata_only
// CHECK-SAME: vc4value.arg_name = "x_ptr"
// CHECK-SAME: vc4value.direction = "in"
// CHECK-SAME: vc4value.shape_args = ["length"]
// CHECK-SAME: vc4value.arg_name = "out_buffer"
// CHECK-SAME: vc4value.direction = "out"
// CHECK-SAME: vc4value.shape_args = ["length"]
// CHECK-SAME: %{{[^:]+}}: index {vc4value.arg_name = "length"}
// CHECK-SAME: %{{[^:]+}}: i32 {vc4value.arg_name = "foo"}
// CHECK-NOT: vc4value.shape_args = ["n"]

#loc = loc("arg_names_metadata_only")
module {
  tt.func public @arg_names_metadata_only(%x: !tt.ptr<f32> loc("x_ptr"(#loc)), %out: !tt.ptr<f32> loc("out_buffer"(#loc)), %n: i32 loc("length"(#loc)), %unused: i32 loc("foo"(#loc))) attributes {noinline = false} {
    %pid = tt.get_program_id x : i32
    %c16 = arith.constant 16 : i32
    %pid16 = arith.muli %pid, %c16 : i32
    %lanes = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %base = tt.splat %pid16 : i32 -> tensor<16xi32>
    %offsets = arith.addi %base, %lanes : tensor<16xi32>
    %bound = tt.splat %n : i32 -> tensor<16xi32>
    %mask = arith.cmpi slt, %offsets, %bound : tensor<16xi32>
    %x_base = tt.splat %x : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %x_ptrs = tt.addptr %x_base, %offsets : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %zero = arith.constant dense<0.000000e+00> : tensor<16xf32>
    %value = tt.load %x_ptrs, %mask, %zero : tensor<16x!tt.ptr<f32>>
    %out_base = tt.splat %out : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %out_ptrs = tt.addptr %out_base, %offsets : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %out_ptrs, %value, %mask : tensor<16x!tt.ptr<f32>>
    tt.return
  } loc(#loc)
}
