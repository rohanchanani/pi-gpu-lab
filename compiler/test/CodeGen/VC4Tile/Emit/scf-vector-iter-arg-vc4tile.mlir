// RUN: vc4-opt --legalize-vc4tile-core-cfg --verify-vc4tile-core --convert-vc4tile-to-ssavc4 %s | vc4-opt --convert-ssavc4-to-vc4 | FileCheck %s

// CHECK: vc4.func
// CHECK: vc4.qpu.branch
// CHECK: vc4.qpu.
// CHECK-NOT: scf.
// CHECK-NOT: vc4tile.

vc4tile.kernel @scf_vector_iter_arg_emit_vc4tile attributes {
  public_name = "scf_vector_iter_arg_emit_vc4tile",
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
