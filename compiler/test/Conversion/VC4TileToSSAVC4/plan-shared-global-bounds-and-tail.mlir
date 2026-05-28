// RUN: vc4-opt %s --canonicalize-vc4tile-surface --plan-vc4tile-copies --legalize-vc4tile-core-cfg --verify-vc4tile-core --convert-vc4tile-to-ssavc4 -o - | FileCheck %s

// CHECK-LABEL: ssavc4.func @plan_shared_global_bounds_and_tail
// CHECK-NOT: vc4tile.copy_tile
// CHECK-NOT: vc4tile.mask_and
// CHECK-NOT: vc4tile.tile_bounds_mask
// CHECK-NOT: vc4tile.shared_store_global
// CHECK: ssavc4.cond_br
// CHECK-COUNT-4: ssavc4.vdw.store_vpm
// CHECK-NOT: ssavc4.vdw.store_vpm
// CHECK-NOT: ssavc4.vpm.read
// CHECK-NOT: ssavc4.vdw.store
vc4tile.kernel @plan_shared_global_bounds_and_tail(%out : i32, %rows : i32, %cols : i32, %n : i32) attributes {
  public_name = "plan_shared_global_bounds_and_tail",
  schedule_mode = #vc4tile.schedule_mode<cooperative_block>,
  arg_attrs = [
    {abi_name = "out", kind = "buffer", direction = "out", elem_type = "u32", type = "u32"},
    {abi_name = "rows", kind = "scalar", direction = "by_value", type = "u32"},
    {abi_name = "cols", kind = "scalar", direction = "by_value", type = "u32"},
    {abi_name = "n", kind = "scalar", direction = "by_value", type = "u32"}
  ],
  warps_per_block_max = 1 : i32,
  uses_shared_vpm = true,
  uses_barrier = false,
  semaphores_per_block = 0 : i32,
  vpm_rows_per_block = 4 : i32,
  vpm_bytes_per_block = 256 : i32
} {
  %zero = arith.constant 0 : i32
  %shared = "vc4tile.shared_tile_alloc"() {
    rows = 4 : i32, elem_bytes = 4 : i32, shape = [4, 16],
    element_type = i32, storage_type = i32, layout = #vc4tile.layout<vpm_row>,
    role = #vc4tile.role<scratch>, precision = #vc4tile.precision<exact_32>,
    packing = #vc4tile.packing<none>, memory_space = #vc4tile.memory_space<shared_vpm>
  } : () -> !vc4tile.shared_tile
  %bounds = vc4tile.tile_bounds_mask %rows, %cols {shape = [4, 4], layout = #vc4tile.layout<row_major>} : i32, i32 -> vector<16xi1>
  %tail = vc4tile.tail_mask %zero, %n : i32, i32 -> vector<16xi1>
  %mask = vc4tile.mask_and %bounds, %tail : vector<16xi1>, vector<16xi1> -> vector<16xi1>
  "vc4tile.copy_tile"(%shared, %zero, %out, %zero, %mask) {
    shape = [4, 4], src_space = #vc4tile.memory_space<shared_vpm>, dst_space = #vc4tile.memory_space<global>,
    src_layout = #vc4tile.layout<vpm_row>, dst_layout = #vc4tile.layout<row_major>,
    element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>,
    packing = #vc4tile.packing<none>, elem_bytes = 4 : i32, memory_pitch_bytes = 16 : i32
  } : (!vc4tile.shared_tile, i32, i32, i32, vector<16xi1>) -> ()
  vc4tile.return
}
