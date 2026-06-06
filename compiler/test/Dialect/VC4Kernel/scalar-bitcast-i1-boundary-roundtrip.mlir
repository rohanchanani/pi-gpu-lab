// RUN: vc4-opt %s --verify-vc4kernel | FileCheck %s

module {
  vc4kernel.kernel @scalar_bitcast_i1_boundary_roundtrip(%x : i32, %y : i32) attributes {
    public_name = "scalar_bitcast_i1_boundary_roundtrip",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "x", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "y", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c1 = arith.constant 1 : i32
    // CHECK: arith.bitcast
    %as_f = arith.bitcast %x : i32 to f32
    // CHECK: arith.bitcast
    %as_i = arith.bitcast %as_f : f32 to i32
    %gt = arith.cmpi sgt, %as_i, %y : i32
    // CHECK: arith.trunci
    %low = arith.trunci %as_i : i32 to i1
    // CHECK: arith.andi
    %and = arith.andi %gt, %low : i1
    // CHECK: arith.ori
    %or = arith.ori %gt, %low : i1
    // CHECK: arith.xori
    %xor = arith.xori %gt, %low : i1
    // CHECK: arith.extui
    %and_i = arith.extui %and : i1 to i32
    %or_i = arith.extui %or : i1 to i32
    %xor_i = arith.extui %xor : i1 to i32
    %sum0 = arith.addi %and_i, %or_i : i32
    %sum1 = arith.addi %sum0, %xor_i : i32
    %sum2 = arith.addi %sum1, %c1 : i32
    %selected = arith.select %xor, %sum2, %as_i : i32
    vc4kernel.return
  }
}
