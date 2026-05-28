// RUN: vc4-opt %s --canonicalize-vc4tile-surface --plan-vc4tile-copies --legalize-vc4tile-core-cfg --verify-vc4tile-core --convert-vc4tile-to-ssavc4 -o - | FileCheck %s

// CHECK-LABEL: ssavc4.func @tile_dot_tail_mask_lowering
// CHECK-NOT: vc4tile.tile_dot
// CHECK-NOT: vc4tile.tile_reduce
// CHECK: ssavc4.alu.mul
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
// CHECK: ssavc4.alu.add
// CHECK: ssavc4.vdw.store
vc4tile.kernel @tile_dot_tail_mask_lowering(%out : i32, %lhs_base : i32, %rhs_base : i32, %n : i32) attributes {
  public_name = "tile_dot_tail_mask_lowering",
  arg_attrs = [
    {abi_name = "out", kind = "buffer", direction = "out", elem_type = "u32", type = "u32"},
    {abi_name = "lhs", kind = "buffer", direction = "in", elem_type = "u32", type = "u32"},
    {abi_name = "rhs", kind = "buffer", direction = "in", elem_type = "u32", type = "u32"},
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
  %lhs = "vc4tile.tile_load"(%lhs_base, %zero, %all) {shape = [1, 16], src_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (i32, i32, !vc4tile.predicate) -> vector<16xi32>
  %rhs = "vc4tile.tile_load"(%rhs_base, %zero, %all) {shape = [1, 16], src_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (i32, i32, !vc4tile.predicate) -> vector<16xi32>
  %result = "vc4tile.tile_dot"(%lhs, %rhs, %tail) {k = 16 : i32, shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<register>, element_type = i32, storage_type = i32, accumulator_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, lhs_role = #vc4tile.role<input>, rhs_role = #vc4tile.role<input>, acc_role = #vc4tile.role<accumulator>, algorithm_hint = "dot_1x16_tail_mask"} : (vector<16xi32>, vector<16xi32>, !vc4tile.predicate) -> vector<16xi32>
  "vc4tile.tile_store"(%result, %out, %zero, %all) {shape = [1, 16], dst_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (vector<16xi32>, i32, i32, !vc4tile.predicate) -> ()
  vc4tile.return
}
