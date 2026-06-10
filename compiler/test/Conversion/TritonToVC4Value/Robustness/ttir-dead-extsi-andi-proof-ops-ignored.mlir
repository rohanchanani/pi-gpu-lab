// REQUIRES: vc4-has-triton-cpp-frontend

// RUN: %vc4_triton_opt %s --convert-triton-to-vc4-value -o - | FileCheck %s

// CHECK-LABEL: func.func @dead_extsi_andi_proof_ops_ignored
// CHECK: vector.transfer_read
// CHECK: vector.transfer_write
// CHECK-NOT: arith.extsi
// CHECK-NOT: arith.andi
// CHECK-NOT: arith.trunci
// CHECK-NOT: tensor<16xi64>
// CHECK-NOT: tt.

module {
  tt.func public @dead_extsi_andi_proof_ops_ignored(%x: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32) attributes {noinline = false} {
    %pid = tt.get_program_id x : i32
    %c16 = arith.constant 16 : i32
    %pid64 = arith.extsi %pid : i32 to i64
    %c16_64 = arith.extsi %c16 : i32 to i64
    %wide = arith.muli %pid64, %c16_64 : i64
    %max = arith.constant 2147483647 : i64
    %min = arith.constant -2147483648 : i64
    %ok_hi = arith.cmpi sle, %wide, %max : i64
    %ok_lo = arith.cmpi sge, %wide, %min : i64
    %ok = arith.andi %ok_hi, %ok_lo : i1
    %pid16 = arith.muli %pid, %c16 : i32
    %lanes = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %base = tt.splat %pid16 : i32 -> tensor<16xi32>
    %base64 = arith.extsi %base : tensor<16xi32> to tensor<16xi64>
    %lanes64 = arith.extsi %lanes : tensor<16xi32> to tensor<16xi64>
    %wide_offsets = arith.addi %base64, %lanes64 : tensor<16xi64>
    %vmax = arith.constant dense<2147483647> : tensor<16xi64>
    %vmin = arith.constant dense<-2147483648> : tensor<16xi64>
    %vok_hi = arith.cmpi sle, %wide_offsets, %vmax : tensor<16xi64>
    %vok_lo = arith.cmpi sge, %wide_offsets, %vmin : tensor<16xi64>
    %vok = arith.andi %vok_hi, %vok_lo : tensor<16xi1>
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
