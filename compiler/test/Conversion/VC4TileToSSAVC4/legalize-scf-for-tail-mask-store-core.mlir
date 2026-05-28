// RUN: vc4-opt --legalize-vc4tile-core-cfg %s | FileCheck %s

// CHECK-NOT: scf.
// CHECK-NOT: index
// CHECK-NOT: arith.cmpi eq
// CHECK: arith.cmpi ult
// CHECK: cf.cond_br
// CHECK-NOT: scf.
// CHECK-NOT: index
// CHECK-NOT: arith.cmpi eq

vc4tile.kernel @legalize_scf_for_tail_mask_store_core attributes {
  public_name = "legalize_scf_for_tail_mask_store_core",
  arg_attrs = [{name = "out", kind = "scalar", direction = "by_value", type = "u32"}, {name = "n", kind = "scalar", direction = "by_value", type = "u32"}]
} {
^entry(%out: i32, %n: i32):
  %lb = arith.constant 0 : index
  %ub = arith.index_cast %n : i32 to index
  %step = arith.constant 1 : index
  %zero = arith.constant 0 : i32
  %total = scf.for %i = %lb to %ub step %step iter_args(%acc = %zero) -> (i32) {
    %ii = arith.index_cast %i : index to i32
    %next = arith.addi %acc, %ii : i32
    scf.yield %next : i32
  }
  %lanes = vc4tile.lane_range : vector<16xi32>
  %total_vec = vector.broadcast %total : i32 to vector<16xi32>
  %value = arith.addi %lanes, %total_vec : vector<16xi32>
  %mask = vc4tile.core_tail_mask %zero, %n : i32, i32 -> vector<16xi1>
  vc4tile.masked_store_global %out, %lanes, %value, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
  vc4tile.return
}
