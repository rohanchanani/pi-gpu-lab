// RUN: vc4-opt --legalize-vc4tile-core-cfg --verify-vc4tile-core --convert-vc4tile-to-ssavc4 %s | vc4-opt --convert-ssavc4-to-vc4 | FileCheck %s

// CHECK: vc4.func
// CHECK: vc4.qpu.branch
// CHECK: vc4.qpu.
// CHECK-NOT: scf.
// CHECK-NOT: vc4tile.

vc4tile.kernel @scf_for_semantics_emit_vc4tile attributes {
  public_name = "scf_for_semantics_emit_vc4tile",
  arg_attrs = [{name = "out", kind = "scalar", direction = "by_value", type = "u32"}]
} {
^entry(%out: i32):
  %lb = arith.constant 0 : index
  %ub = arith.constant 5 : index
  %step = arith.constant 2 : index
  %zero = arith.constant 0 : i32
  %ten = arith.constant 10 : i32
  %one = arith.constant 1 : i32
  %total = scf.for %i = %lb to %ub step %step iter_args(%acc = %zero) -> (i32) {
    %ii = arith.index_cast %i : index to i32
    %scaled = arith.muli %ii, %ten : i32
    %term = arith.addi %scaled, %one : i32
    %next = arith.addi %acc, %term : i32
    scf.yield %next : i32
  }

  %lanes = vc4tile.lane_range : vector<16xi32>
  %total_vec = vector.broadcast %total : i32 to vector<16xi32>
  %value = arith.addi %lanes, %total_vec : vector<16xi32>
  %mask = vc4tile.mask_all : vector<16xi1>
  vc4tile.masked_store_global %out, %lanes, %value, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
  vc4tile.return
}
