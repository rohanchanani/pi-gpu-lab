// RUN: vc4-opt --legalize-vc4tile-core-cfg %s | FileCheck %s

// CHECK-NOT: scf.
// CHECK-NOT: index
// CHECK-NOT: arith.cmpi eq
// CHECK: arith.cmpi ult
// CHECK: cf.cond_br
// CHECK-NOT: scf.
// CHECK-NOT: index
// CHECK-NOT: arith.cmpi eq

vc4tile.kernel @legalize_scf_for_nested_core attributes {
  public_name = "legalize_scf_for_nested_core",
  arg_attrs = [{name = "out", kind = "scalar", direction = "by_value", type = "u32"}]
} {
^entry(%out: i32):
  %olb = arith.constant 0 : index
  %oub = arith.constant 3 : index
  %ilb = arith.constant 0 : index
  %iub = arith.constant 2 : index
  %step = arith.constant 1 : index
  %zero = arith.constant 0 : i32
  %ten = arith.constant 10 : i32
  %total = scf.for %oi = %olb to %oub step %step iter_args(%outer_acc = %zero) -> (i32) {
    %o = arith.index_cast %oi : index to i32
    %inner = scf.for %ii = %ilb to %iub step %step iter_args(%inner_acc = %outer_acc) -> (i32) {
      %j = arith.index_cast %ii : index to i32
      %os = arith.muli %o, %ten : i32
      %term = arith.addi %os, %j : i32
      %next = arith.addi %inner_acc, %term : i32
      scf.yield %next : i32
    }
    scf.yield %inner : i32
  }

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
