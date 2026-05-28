// RUN: vc4-opt %s --canonicalize-vc4tile-surface --plan-vc4tile-copies --legalize-vc4tile-core-cfg --verify-vc4tile-core --convert-vc4tile-to-ssavc4 -o - | FileCheck %s

// CHECK-LABEL: ssavc4.func @tile_store_bounds_and_tail_mask_lowering
// CHECK-NOT: vc4tile.mask_and
// CHECK-NOT: vc4tile.tile_bounds_mask
// CHECK-NOT: vc4tile.tile_store
// CHECK-NOT: vc4tile.masked_store_global
// CHECK: ssavc4.cond_br
// CHECK-COUNT-4: ssavc4.vdw.store
// CHECK-NOT: ssavc4.vdw.store
vc4tile.kernel @tile_store_bounds_and_tail_mask_lowering(%out : i32, %rows : i32, %cols : i32, %n : i32) attributes {
  public_name = "tile_store_bounds_and_tail_mask_lowering",
  arg_attrs = [
    {abi_name = "out", kind = "buffer", direction = "out", elem_type = "u32", type = "u32"},
    {abi_name = "rows", kind = "scalar", direction = "by_value", type = "u32"},
    {abi_name = "cols", kind = "scalar", direction = "by_value", type = "u32"},
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
  %values = vc4tile.lane_range : vector<16xi32>
  %bounds = vc4tile.tile_bounds_mask %rows, %cols {shape = [4, 4], layout = #vc4tile.layout<row_major>} : i32, i32 -> vector<16xi1>
  %tail = vc4tile.tail_mask %zero, %n : i32, i32 -> vector<16xi1>
  %mask = vc4tile.mask_and %bounds, %tail : vector<16xi1>, vector<16xi1> -> vector<16xi1>
  "vc4tile.tile_store"(%values, %out, %zero, %mask) {shape = [4, 4], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, boundary = #vc4tile.boundary_policy<exact>, packing = #vc4tile.packing<none>, role = #vc4tile.role<output>} : (vector<16xi32>, i32, i32, vector<16xi1>) -> ()
  vc4tile.return
}
