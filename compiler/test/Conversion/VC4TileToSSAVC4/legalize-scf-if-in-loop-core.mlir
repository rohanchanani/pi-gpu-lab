// RUN: vc4-opt --legalize-vc4tile-core-cfg %s | FileCheck %s

// CHECK-NOT: scf.
// CHECK-NOT: index
// CHECK-NOT: arith.cmpi eq
// CHECK: arith.cmpi ult
// CHECK: cf.cond_br
// CHECK-NOT: scf.
// CHECK-NOT: index
// CHECK-NOT: arith.cmpi eq

vc4tile.kernel @legalize_scf_if_in_loop_core attributes {
  public_name = "legalize_scf_if_in_loop_core",
  arg_attrs = [{name = "out", kind = "scalar", direction = "by_value", type = "u32"}]
} {
^entry(%out: i32):
  %lb = arith.constant 0 : index
  %ub = arith.constant 6 : index
  %step = arith.constant 1 : index
  %zero = arith.constant 0 : i32
  %three = arith.constant 3 : i32
  %ten = arith.constant 10 : i32
  %one = arith.constant 1 : i32
  %total = scf.for %i = %lb to %ub step %step iter_args(%acc = %zero) -> (i32) {
    %ii = arith.index_cast %i : index to i32
    %lt = arith.cmpi ult, %ii, %three : i32
    %term = scf.if %lt -> (i32) {
      scf.yield %ten : i32
    } else {
      scf.yield %one : i32
    }
    %next = arith.addi %acc, %term : i32
    scf.yield %next : i32
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
