// RUN: not %vc4_triton_opt %s --convert-triton-to-vc4-value -o %t 2>&1 | FileCheck %s

module {
  tt.func public @sparse_load_mask(%x_ptr: !tt.ptr<f32> loc("x_ptr"), %out_ptr: !tt.ptr<f32> loc("out_ptr"), %n: i32 loc("n")) {
    %pid = tt.get_program_id x : i32
    %lanes = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %c16 = arith.constant 16 : i32
    %base = arith.muli %pid, %c16 : i32
    %basev = tt.splat %base : i32 -> tensor<16xi32>
    %offsets = arith.addi %basev, %lanes : tensor<16xi32>
    %bound = tt.splat %n : i32 -> tensor<16xi32>
    %tail = arith.cmpi slt, %offsets, %bound : tensor<16xi32>
    %x_base = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %x_addr = tt.addptr %x_base, %offsets : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %zero = arith.constant dense<0.000000e+00> : tensor<16xf32>
    %x = tt.load %x_addr, %tail, %zero : tensor<16x!tt.ptr<f32>>
    %zero2 = arith.constant dense<0.000000e+00> : tensor<16xf32>
    %sparse = arith.cmpf ogt, %x, %zero2 : tensor<16xf32>
    %bad = tt.load %x_addr, %sparse, %zero : tensor<16x!tt.ptr<f32>>
    %out_base = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %out_addr = tt.addptr %out_base, %offsets : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %out_addr, %bad, %tail : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK: sparse or unknown tt.load memory mask
// CHECK: READY_FOR_TRITON remains NO
