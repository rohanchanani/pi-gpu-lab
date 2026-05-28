// RUN: vc4-opt --legalize-vc4tile-core-cfg %s | FileCheck %s

// CHECK-NOT: scf.
// CHECK-NOT: index
// CHECK-NOT: arith.cmpi eq
// CHECK: arith.cmpi ult
// CHECK: cf.cond_br
// CHECK-NOT: scf.
// CHECK-NOT: index
// CHECK-NOT: arith.cmpi eq

vc4tile.kernel @legalize_scf_for_vector_iter_arg_core attributes {
  public_name = "legalize_scf_for_vector_iter_arg_core",
  arg_attrs = [{name = "out", kind = "scalar", direction = "by_value", type = "u32"}]
} {
^entry(%out: i32):
  %lb = arith.constant 0 : index
  %ub = arith.constant 5 : index
  %step = arith.constant 1 : index
  %lanes0 = vc4tile.lane_range : vector<16xi32>
  %vec = scf.for %i = %lb to %ub step %step iter_args(%acc = %lanes0) -> (vector<16xi32>) {
    %ii = arith.index_cast %i : index to i32
    %s = vector.broadcast %ii : i32 to vector<16xi32>
    %next = arith.addi %acc, %s : vector<16xi32>
    scf.yield %next : vector<16xi32>
  }
  %mask = vc4tile.core_mask_all : vector<16xi1>
  vc4tile.masked_store_global %out, %lanes0, %vec, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
  vc4tile.return
}
