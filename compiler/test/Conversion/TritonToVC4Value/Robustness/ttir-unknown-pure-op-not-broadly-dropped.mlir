// REQUIRES: vc4-has-triton-cpp-frontend

// RUN: not %vc4_triton_opt %s --convert-triton-to-vc4-value -o - 2>&1 | FileCheck %s

// CHECK: arith.divsi
// CHECK: staged TTIR target-profile feature
// CHECK: READY_FOR_TRITON remains NO

module {
  tt.func public @unknown_pure_op_not_broadly_dropped(%x: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32) attributes {noinline = false} {
    %pid = tt.get_program_id x : i32
    %c16 = arith.constant 16 : i32
    %dead_div = arith.divsi %pid, %c16 : i32
    %dead_user = arith.muli %dead_div, %c16 : i32
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
  }
}
