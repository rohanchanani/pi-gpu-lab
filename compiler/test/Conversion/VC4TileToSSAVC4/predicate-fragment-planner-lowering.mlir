// RUN: vc4-opt %s --canonicalize-vc4tile-surface --plan-vc4tile-copies --legalize-vc4tile-core-cfg --verify-vc4tile-core -o - | FileCheck %s

// CHECK-LABEL: vc4tile.kernel @predicate_fragment_planner_full_block
// CHECK-NOT: vc4tile.copy_tile
// CHECK: vc4tile.vdr_load_tile
// CHECK-SAME: nrows = 4
// CHECK-SAME: row_len = 4
// CHECK-NOT: vc4tile.mask_or
vc4tile.kernel @predicate_fragment_planner_full_block(%in : i32, %n : i32) attributes {
  public_name = "predicate_fragment_planner_full_block",
  schedule_mode = #vc4tile.schedule_mode<cooperative_block>,
  arg_attrs = [
    {abi_name = "in", kind = "buffer", direction = "in", elem_type = "u32", type = "u32"},
    {abi_name = "n", kind = "scalar", direction = "by_value", type = "u32"}
  ],
  warps_per_block_max = 1 : i32,
  uses_shared_vpm = true,
  uses_barrier = false,
  require_full_block_residency = true,
  semaphores_per_block = 4 : i32,
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
  %all = vc4tile.mask_all
  %tail = vc4tile.tail_mask %zero, %n : i32, i32
  %mask = vc4tile.mask_or %all, %tail
  "vc4tile.copy_tile"(%in, %shared, %zero, %mask) {
    shape = [4, 4], src_space = #vc4tile.memory_space<global>, dst_space = #vc4tile.memory_space<shared_vpm>,
    src_layout = #vc4tile.layout<row_major>, dst_layout = #vc4tile.layout<vpm_row>,
    element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>,
    packing = #vc4tile.packing<none>, elem_bytes = 4 : i32, memory_pitch_bytes = 16 : i32
  } : (i32, !vc4tile.shared_tile, i32, !vc4tile.predicate) -> ()
  vc4tile.return
}
