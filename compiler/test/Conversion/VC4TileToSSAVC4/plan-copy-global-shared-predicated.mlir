// RUN: vc4-opt %s --canonicalize-vc4tile-surface --plan-vc4tile-copies --legalize-vc4tile-core-cfg --verify-vc4tile-core -o - | FileCheck %s --check-prefix=CORE
// RUN: vc4-opt %s --canonicalize-vc4tile-surface --plan-vc4tile-copies --legalize-vc4tile-core-cfg --verify-vc4tile-core --convert-vc4tile-to-ssavc4 -o - | FileCheck %s --check-prefix=SSAVC4

// CORE-LABEL: vc4tile.kernel @plan_copy_global_shared_bounds_predicated
// CORE-NOT: vc4tile.copy_tile
// CORE-NOT: vc4tile.vdr_load_tile
// CORE: vc4tile.masked_load_global
// CORE: vc4tile.shared_store
// CORE: cf.cond_br
// CORE: vc4tile.shared_store

// SSAVC4-LABEL: ssavc4.func @plan_copy_global_shared_bounds_predicated
// SSAVC4-NOT: ssavc4.vdr.load
// SSAVC4: ssavc4.tmu.request
// SSAVC4: ssavc4.vpm.write
vc4tile.kernel @plan_copy_global_shared_bounds_predicated(%in : i32, %rows : i32, %cols : i32) attributes {
  public_name = "plan_copy_global_shared_bounds_predicated",
  schedule_mode = #vc4tile.schedule_mode<cooperative_block>,
  arg_attrs = [
    {abi_name = "in", kind = "buffer", direction = "in", elem_type = "u32", type = "u32"},
    {abi_name = "rows", kind = "scalar", direction = "by_value", type = "u32"},
    {abi_name = "cols", kind = "scalar", direction = "by_value", type = "u32"}
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
  %mask = vc4tile.tile_bounds_mask %rows, %cols {shape = [4, 4], layout = #vc4tile.layout<row_major>} : i32, i32 -> vector<16xi1>
  "vc4tile.copy_tile"(%in, %shared, %zero, %mask) {
    shape = [4, 4], src_space = #vc4tile.memory_space<global>, dst_space = #vc4tile.memory_space<shared_vpm>,
    src_layout = #vc4tile.layout<row_major>, dst_layout = #vc4tile.layout<vpm_row>,
    element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>,
    packing = #vc4tile.packing<none>, elem_bytes = 4 : i32, memory_pitch_bytes = 16 : i32
  } : (i32, !vc4tile.shared_tile, i32, vector<16xi1>) -> ()
  vc4tile.return
}

// CORE-LABEL: vc4tile.kernel @plan_copy_global_shared_rect_predicated
// CORE-NOT: vc4tile.copy_tile
// CORE-NOT: vc4tile.vdr_load_tile
// CORE: vc4tile.masked_load_global
// CORE: vc4tile.shared_store
vc4tile.kernel @plan_copy_global_shared_rect_predicated(%in : i32) attributes {
  public_name = "plan_copy_global_shared_rect_predicated",
  schedule_mode = #vc4tile.schedule_mode<cooperative_block>,
  arg_attrs = [{abi_name = "in", kind = "buffer", direction = "in", elem_type = "u32", type = "u32"}],
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
  %mask = vc4tile.tile_rect_mask {active_rows = 3 : i32, active_cols = 2 : i32, shape = [4, 4], layout = #vc4tile.layout<row_major>} : vector<16xi1>
  "vc4tile.copy_tile"(%in, %shared, %zero, %mask) {
    shape = [4, 4], src_space = #vc4tile.memory_space<global>, dst_space = #vc4tile.memory_space<shared_vpm>,
    src_layout = #vc4tile.layout<row_major>, dst_layout = #vc4tile.layout<vpm_row>,
    element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>,
    packing = #vc4tile.packing<none>, elem_bytes = 4 : i32, memory_pitch_bytes = 16 : i32
  } : (i32, !vc4tile.shared_tile, i32, vector<16xi1>) -> ()
  vc4tile.return
}
