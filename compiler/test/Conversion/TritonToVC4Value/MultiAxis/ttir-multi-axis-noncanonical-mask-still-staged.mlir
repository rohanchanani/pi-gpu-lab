// REQUIRES: vc4-has-triton-cpp-frontend
// RUN: not %vc4_triton_opt %s --convert-triton-to-vc4-value -o %t 2>&1 | FileCheck %s

#loc = loc(unknown)

module {
  tt.func public @multi_axis_noncanonical_mask(%out_ptr: !tt.ptr<i32> loc("out_ptr"(#loc)), %n_elements: i32 loc("n_elements"(#loc))) attributes {noinline = false} {
    %pid0 = tt.get_program_id x : i32 loc(#loc)
    %pid1 = tt.get_program_id y : i32 loc(#loc)
    %nprog0 = tt.get_num_programs x : i32 loc(#loc)
    %lanes = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32> loc(#loc)
    %row_base = arith.muli %pid1, %nprog0 : i32 loc(#loc)
    %block_id = arith.addi %row_base, %pid0 : i32 loc(#loc)
    %c16 = arith.constant 16 : i32 loc(#loc)
    %base = arith.muli %block_id, %c16 : i32 loc(#loc)
    %basev = tt.splat %base : i32 -> tensor<16xi32> loc(#loc)
    %offsets = arith.addi %basev, %lanes : tensor<16xi32> loc(#loc)
    %bound = tt.splat %n_elements : i32 -> tensor<16xi32> loc(#loc)
    %bad_mask = arith.cmpi slt, %lanes, %bound : tensor<16xi32> loc(#loc)
    %out = tt.splat %out_ptr : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>> loc(#loc)
    %addr = tt.addptr %out, %offsets : tensor<16x!tt.ptr<i32>>, tensor<16xi32> loc(#loc)
    %zero = arith.constant dense<0> : tensor<16xi32> loc(#loc)
    tt.store %addr, %zero, %bad_mask : tensor<16x!tt.ptr<i32>> loc(#loc)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)

// CHECK: sparse or unknown tt.store memory mask
// CHECK: staged TTIR target-profile feature
// CHECK: READY_FOR_TRITON remains NO
