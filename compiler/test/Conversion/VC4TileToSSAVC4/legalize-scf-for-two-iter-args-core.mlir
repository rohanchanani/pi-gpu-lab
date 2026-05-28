// RUN: vc4-opt --legalize-vc4tile-core-cfg %s | FileCheck %s

// CHECK-NOT: scf.
// CHECK-NOT: index
// CHECK-NOT: arith.cmpi eq
// CHECK: arith.cmpi ult
// CHECK: cf.cond_br
// CHECK-NOT: scf.
// CHECK-NOT: index
// CHECK-NOT: arith.cmpi eq

vc4tile.kernel @legalize_scf_for_two_iter_args_core attributes {
  public_name = "legalize_scf_for_two_iter_args_core",
  arg_attrs = [{name = "out", kind = "scalar", direction = "by_value", type = "u32"}]
} {
^entry(%out: i32):
  %lb = arith.constant 0 : index
  %ub = arith.constant 4 : index
  %step = arith.constant 1 : index
  %zero = arith.constant 0 : i32
  %two = arith.constant 2 : i32
  %one = arith.constant 1 : i32
  %a, %b = scf.for %i = %lb to %ub step %step iter_args(%acc_a = %zero, %acc_b = %zero) -> (i32, i32) {
    %ii = arith.index_cast %i : index to i32
    %next_a = arith.addi %acc_a, %ii : i32
    %twice = arith.muli %ii, %two : i32
    %term_b = arith.addi %twice, %one : i32
    %next_b = arith.addi %acc_b, %term_b : i32
    scf.yield %next_a, %next_b : i32, i32
  }
  %total = arith.addi %a, %b : i32

  %lanes = vc4tile.lane_range : vector<16xi32>
  %total_vec = vector.broadcast %total : i32 to vector<16xi32>
  %value = arith.addi %lanes, %total_vec : vector<16xi32>
  %mask = vc4tile.core_mask_all : vector<16xi1>
  vc4tile.masked_store_global %out, %lanes, %value, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
  vc4tile.return
}
