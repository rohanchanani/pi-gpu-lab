// RUN: vc4-opt %s --canonicalize-vc4tile-surface --plan-vc4tile-copies --legalize-vc4tile-core-cfg --verify-vc4tile-core --convert-vc4tile-to-ssavc4 | FileCheck %s

// CHECK-LABEL: ssavc4.func @shared_vpm_bounds_predicated_load_store
// CHECK-NOT: vc4tile.tile_bounds_mask
// CHECK-NOT: vc4tile.shared_store
// CHECK-NOT: vc4tile.shared_load
// CHECK: ssavc4.cond_select
// CHECK: ssavc4.vpm.write
// CHECK: ssavc4.vpm.read
// CHECK: ssavc4.cond_select
vc4tile.kernel @shared_vpm_bounds_predicated_load_store(%out : i32, %rows : i32, %cols : i32) attributes {
  public_name = "shared_vpm_bounds_predicated_load_store",
  schedule_mode = #vc4tile.schedule_mode<cooperative_block>,
  arg_attrs = [
    {abi_name = "out", kind = "buffer", direction = "out", elem_type = "u32", type = "u32"},
    {abi_name = "rows", kind = "scalar", direction = "by_value", type = "u32"},
    {abi_name = "cols", kind = "scalar", direction = "by_value", type = "u32"}
  ],
  warps_per_block_max = 1 : i32,
  uses_shared_vpm = true,
  uses_barrier = false,
  semaphores_per_block = 0 : i32,
  vpm_rows_per_block = 1 : i32,
  vpm_bytes_per_block = 64 : i32
} {
  %zero = arith.constant 0 : i32
  %values = vc4tile.lane_range : vector<16xi32>
  %shared = "vc4tile.shared_tile_alloc"() {
    rows = 1 : i32, elem_bytes = 4 : i32, shape = [1, 16],
    element_type = i32, storage_type = i32, layout = #vc4tile.layout<vpm_row>,
    role = #vc4tile.role<scratch>, precision = #vc4tile.precision<exact_32>,
    packing = #vc4tile.packing<none>, memory_space = #vc4tile.memory_space<shared_vpm>
  } : () -> !vc4tile.shared_tile
  %mask = vc4tile.tile_bounds_mask %rows, %cols {shape = [4, 4], layout = #vc4tile.layout<row_major>} : i32, i32 -> vector<16xi1>
  "vc4tile.copy_tile"(%values, %shared, %zero, %mask) {
    shape = [1, 16], src_space = #vc4tile.memory_space<register>, dst_space = #vc4tile.memory_space<shared_vpm>,
    src_layout = #vc4tile.layout<row_major>, dst_layout = #vc4tile.layout<vpm_row>,
    element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, elem_bytes = 4 : i32
  } : (vector<16xi32>, !vc4tile.shared_tile, i32, vector<16xi1>) -> ()
  %loaded = "vc4tile.copy_tile"(%shared, %zero, %mask) {
    shape = [1, 16], src_space = #vc4tile.memory_space<shared_vpm>, dst_space = #vc4tile.memory_space<register>,
    src_layout = #vc4tile.layout<vpm_row>, dst_layout = #vc4tile.layout<row_major>,
    element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, elem_bytes = 4 : i32
  } : (!vc4tile.shared_tile, i32, vector<16xi1>) -> vector<16xi32>
  %all = vc4tile.mask_all : vector<16xi1>
  "vc4tile.tile_store"(%loaded, %out, %zero, %all) {
    shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>,
    element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>
  } : (vector<16xi32>, i32, i32, vector<16xi1>) -> ()
  vc4tile.return
}

// CHECK-LABEL: ssavc4.func @shared_vpm_rect_predicated_load_store
// CHECK-NOT: vc4tile.tile_rect_mask
// CHECK-NOT: vc4tile.shared_store
// CHECK-NOT: vc4tile.shared_load
// CHECK: ssavc4.cond_select
// CHECK: ssavc4.vpm.write
// CHECK: ssavc4.vpm.read
// CHECK: ssavc4.cond_select
vc4tile.kernel @shared_vpm_rect_predicated_load_store(%out : i32) attributes {
  public_name = "shared_vpm_rect_predicated_load_store",
  schedule_mode = #vc4tile.schedule_mode<cooperative_block>,
  arg_attrs = [{abi_name = "out", kind = "buffer", direction = "out", elem_type = "u32", type = "u32"}],
  warps_per_block_max = 1 : i32,
  uses_shared_vpm = true,
  uses_barrier = false,
  semaphores_per_block = 0 : i32,
  vpm_rows_per_block = 1 : i32,
  vpm_bytes_per_block = 64 : i32
} {
  %zero = arith.constant 0 : i32
  %values = vc4tile.lane_range : vector<16xi32>
  %shared = "vc4tile.shared_tile_alloc"() {
    rows = 1 : i32, elem_bytes = 4 : i32, shape = [1, 16],
    element_type = i32, storage_type = i32, layout = #vc4tile.layout<vpm_row>,
    role = #vc4tile.role<scratch>, precision = #vc4tile.precision<exact_32>,
    packing = #vc4tile.packing<none>, memory_space = #vc4tile.memory_space<shared_vpm>
  } : () -> !vc4tile.shared_tile
  %mask = vc4tile.tile_rect_mask {active_rows = 3 : i32, active_cols = 2 : i32, shape = [4, 4], layout = #vc4tile.layout<row_major>} : vector<16xi1>
  "vc4tile.copy_tile"(%values, %shared, %zero, %mask) {
    shape = [1, 16], src_space = #vc4tile.memory_space<register>, dst_space = #vc4tile.memory_space<shared_vpm>,
    src_layout = #vc4tile.layout<row_major>, dst_layout = #vc4tile.layout<vpm_row>,
    element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, elem_bytes = 4 : i32
  } : (vector<16xi32>, !vc4tile.shared_tile, i32, vector<16xi1>) -> ()
  %loaded = "vc4tile.copy_tile"(%shared, %zero, %mask) {
    shape = [1, 16], src_space = #vc4tile.memory_space<shared_vpm>, dst_space = #vc4tile.memory_space<register>,
    src_layout = #vc4tile.layout<vpm_row>, dst_layout = #vc4tile.layout<row_major>,
    element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, elem_bytes = 4 : i32
  } : (!vc4tile.shared_tile, i32, vector<16xi1>) -> vector<16xi32>
  %all = vc4tile.mask_all : vector<16xi1>
  "vc4tile.tile_store"(%loaded, %out, %zero, %all) {
    shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>,
    element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>
  } : (vector<16xi32>, i32, i32, vector<16xi1>) -> ()
  vc4tile.return
}
