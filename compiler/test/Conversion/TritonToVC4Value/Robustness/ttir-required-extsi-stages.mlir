// REQUIRES: vc4-has-triton-cpp-frontend

// RUN: not %vc4_triton_opt %s --convert-triton-to-vc4-value -o - 2>&1 | FileCheck %s

// CHECK: arith.extsi required use staged
// CHECK: READY_FOR_TRITON remains NO

module {
  tt.func public @required_extsi_stages(%x: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32) attributes {noinline = false} {
    %pid = tt.get_program_id x : i32
    %c16 = arith.constant 16 : i32
    %pid16 = arith.muli %pid, %c16 : i32
    %lanes = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %base = tt.splat %pid16 : i32 -> tensor<16xi32>
    %offsets = arith.addi %base, %lanes : tensor<16xi32>
    %bound = tt.splat %n : i32 -> tensor<16xi32>
    %mask = arith.cmpi slt, %offsets, %bound : tensor<16xi32>
    %wide_pid = arith.extsi %pid : i32 to i64
    %zero64 = arith.constant 0 : i64
    %cond = arith.cmpi sge, %wide_pid, %zero64 : i64
    scf.if %cond {
    }
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
