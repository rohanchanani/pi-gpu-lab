// REQUIRES: vc4-has-triton-cpp-frontend

// RUN: not %vc4_triton_opt %s --convert-triton-to-vc4-value -o - 2>&1 | FileCheck %s

// CHECK: arith.index_cast
// CHECK: staged TTIR target-profile feature
// CHECK: READY_FOR_TRITON remains NO

module {
  tt.func public @unmapped_result_accounting_regression(%out: !tt.ptr<i32>, %n: i32) attributes {noinline = false} {
    %pid = tt.get_program_id x : i32
    %c16 = arith.constant 16 : i32
    %pid16 = arith.muli %pid, %c16 : i32
    %lanes = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %base = tt.splat %pid16 : i32 -> tensor<16xi32>
    %offsets = arith.addi %base, %lanes : tensor<16xi32>
    %bound = tt.splat %n : i32 -> tensor<16xi32>
    %mask = arith.cmpi slt, %offsets, %bound : tensor<16xi32>
    %idx = arith.index_cast %pid : i32 to index
    %back = arith.index_cast %idx : index to i32
    %value = tt.splat %back : i32 -> tensor<16xi32>
    %out_base = tt.splat %out : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
    %out_ptrs = tt.addptr %out_base, %offsets : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    tt.store %out_ptrs, %value, %mask : tensor<16x!tt.ptr<i32>>
    tt.return
  }
}
