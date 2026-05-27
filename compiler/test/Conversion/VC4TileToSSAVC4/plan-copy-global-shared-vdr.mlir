// RUN: vc4-opt %s --canonicalize-vc4tile-surface --plan-vc4tile-copies --legalize-vc4tile-core-cfg --verify-vc4tile-core -o - | FileCheck %s --check-prefix=CORE
// RUN: vc4-opt %s --canonicalize-vc4tile-surface --plan-vc4tile-copies --legalize-vc4tile-core-cfg --verify-vc4tile-core --convert-vc4tile-to-ssavc4 -o - | FileCheck %s --check-prefix=SSAVC4

// CORE-LABEL: vc4tile.kernel @plan_copy_global_shared_vdr
// CORE-NOT: vc4tile.copy_tile
// CORE: vc4tile.vdr_load_tile
// CORE-SAME: nrows = 2 : i32
// CORE-SAME: row_len = 16 : i32
// SSAVC4-LABEL: ssavc4.func @plan_copy_global_shared_vdr
// SSAVC4: ssavc4.vdr.load
// SSAVC4-SAME: nrows = 2 : i32
// SSAVC4-SAME: row_len = 16 : i32
vc4tile.kernel @plan_copy_global_shared_vdr(%in : i32) attributes {
  public_name = "plan_copy_global_shared_vdr",
  schedule_mode = #vc4tile.schedule_mode<cooperative_block>,
  arg_attrs = [{abi_name = "in", kind = "buffer", direction = "in", elem_type = "u32", type = "u32"}],
  warps_per_block_max = 1 : i32,
  uses_shared_vpm = true,
  uses_barrier = false,
  require_full_block_residency = true,
  semaphores_per_block = 4 : i32,
  vpm_rows_per_block = 2 : i32,
  vpm_bytes_per_block = 128 : i32
} {
  %zero = arith.constant 0 : i32
  %shared = "vc4tile.shared_tile_alloc"() {rows = 2 : i32, elem_bytes = 4 : i32, shape = [2, 16], element_type = i32, storage_type = i32, layout = #vc4tile.layout<vpm_row>, role = #vc4tile.role<scratch>, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, memory_space = #vc4tile.memory_space<shared_vpm>} : () -> !vc4tile.shared_tile
  "vc4tile.copy_tile"(%in, %shared, %zero) {shape = [2, 16], src_space = #vc4tile.memory_space<global>, dst_space = #vc4tile.memory_space<shared_vpm>, src_layout = #vc4tile.layout<row_major>, dst_layout = #vc4tile.layout<vpm_row>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, elem_bytes = 4 : i32, memory_pitch_bytes = 64 : i32} : (i32, !vc4tile.shared_tile, i32) -> ()
  vc4tile.return
}
