// RUN: vc4-opt %s --canonicalize-vc4tile-surface --plan-vc4tile-copies --legalize-vc4tile-core-cfg --verify-vc4tile-core --convert-vc4tile-to-ssavc4 -o - | FileCheck %s

// CHECK-LABEL: ssavc4.func @tile_reduce_tail_mask_lowering
// CHECK-NOT: vc4tile.tile_reduce
// CHECK: ssavc4.make_flags
// CHECK-SAME: kind = #ssavc4.flag_kind<compare>
// CHECK: ssavc4.cond_select
// CHECK-SAME: cond = #vc4.cond<cs>
// CHECK: ssavc4.make_flags
// CHECK-SAME: kind = #ssavc4.flag_kind<zero_test>
// CHECK: ssavc4.cond_select
// CHECK-SAME: cond = #vc4.cond<zc>
// CHECK: ssavc4.rotate
// CHECK-SAME: amount = 8 : i32
// CHECK: ssavc4.rotate
// CHECK-SAME: amount = 4 : i32
// CHECK: ssavc4.rotate
// CHECK-SAME: amount = 2 : i32
// CHECK: ssavc4.rotate
// CHECK-SAME: amount = 1 : i32
// CHECK: ssavc4.vdw.store
vc4tile.kernel @tile_reduce_tail_mask_lowering(%out : i32, %in : i32, %n : i32) attributes {
  public_name = "tile_reduce_tail_mask_lowering",
  arg_attrs = [
    {abi_name = "out", kind = "buffer", direction = "out", elem_type = "u32", type = "u32"},
    {abi_name = "in", kind = "buffer", direction = "in", elem_type = "u32", type = "u32"},
    {abi_name = "n", kind = "scalar", direction = "by_value", type = "u32"}
  ],
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  warps_per_block_max = 1 : i32,
  uses_shared_vpm = false,
  uses_barrier = false,
  semaphores_per_block = 0 : i32,
  vpm_rows_per_block = 0 : i32,
  vpm_bytes_per_block = 0 : i32
} {
  %zero = arith.constant 0 : i32
  %all = vc4tile.mask_all
  %tail = vc4tile.tail_mask %zero, %n : i32, i32
  %tile = "vc4tile.tile_load"(%in, %zero, %all) {shape = [1, 16], src_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, boundary = #vc4tile.boundary_policy<exact>, packing = #vc4tile.packing<none>} : (i32, i32, !vc4tile.predicate) -> vector<16xi32>
  %sum = "vc4tile.tile_reduce"(%tile, %tail) {shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<register>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, kind = #vc4tile.reduce_kind<add>, axis = 1 : i32, algorithm_hint = "predicated_rotate_add_tree"} : (vector<16xi32>, !vc4tile.predicate) -> vector<16xi32>
  "vc4tile.tile_store"(%sum, %out, %zero, %all) {shape = [1, 16], dst_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, boundary = #vc4tile.boundary_policy<exact>, packing = #vc4tile.packing<none>} : (vector<16xi32>, i32, i32, !vc4tile.predicate) -> ()
  vc4tile.return
}
