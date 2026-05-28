// RUN: vc4-opt %s --canonicalize-vc4tile-surface --plan-vc4tile-copies --legalize-vc4tile-core-cfg --verify-vc4tile-core --convert-vc4tile-to-ssavc4 -o - | FileCheck %s

// CHECK-LABEL: ssavc4.func @predicate_hardware_mapping_full_static_rect
// CHECK-NOT: vc4tile.copy_tile
// CHECK: ssavc4.vdr.load
// CHECK-SAME: nrows = 4 : i32
// CHECK-SAME: row_len = 4 : i32
vc4tile.kernel @predicate_hardware_mapping_full_static_rect(%in : i32) attributes {
  public_name = "predicate_hardware_mapping_full_static_rect",
  schedule_mode = #vc4tile.schedule_mode<cooperative_block>,
  arg_attrs = [
    {abi_name = "in", kind = "buffer", direction = "in", elem_type = "u32", type = "u32"}
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
  "vc4tile.copy_tile"(%in, %shared, %zero, %all) {
    shape = [4, 4],
    src_space = #vc4tile.memory_space<global>,
    dst_space = #vc4tile.memory_space<shared_vpm>,
    src_layout = #vc4tile.layout<row_major>,
    dst_layout = #vc4tile.layout<vpm_row>,
    element_type = i32, storage_type = i32,
    precision = #vc4tile.precision<exact_32>,
    packing = #vc4tile.packing<none>,
    elem_bytes = 4 : i32,
    memory_pitch_bytes = 16 : i32
  } : (i32, !vc4tile.shared_tile, i32, !vc4tile.predicate) -> ()
  vc4tile.return
}

// CHECK-LABEL: ssavc4.func @predicate_hardware_mapping_register_row_tail
// CHECK-NOT: vc4tile.tile_store
// CHECK: ssavc4.vdw.store {{.*}}, {{.*}}, %{{.*}} {
// CHECK-SAME: active_lanes = 16 : i32
vc4tile.kernel @predicate_hardware_mapping_register_row_tail(%out : i32, %n : i32) attributes {
  public_name = "predicate_hardware_mapping_register_row_tail",
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  arg_attrs = [
    {abi_name = "out", kind = "buffer", direction = "out", elem_type = "u32", type = "u32"},
    {abi_name = "n", kind = "scalar", direction = "by_value", type = "u32"}
  ],
  warps_per_block_max = 1 : i32,
  uses_shared_vpm = false,
  uses_barrier = false,
  semaphores_per_block = 0 : i32,
  vpm_rows_per_block = 0 : i32,
  vpm_bytes_per_block = 0 : i32
} {
  %zero = arith.constant 0 : i32
  %values = vc4tile.lane_range : vector<16xi32>
  %tail = vc4tile.tail_mask %zero, %n : i32, i32
  "vc4tile.tile_store"(%values, %out, %zero, %tail) {
    shape = [1, 16],
    dst_layout = #vc4tile.layout<row_major>,
    memory_space = #vc4tile.memory_space<global>,
    element_type = i32, storage_type = i32,
    precision = #vc4tile.precision<exact_32>,
    boundary = #vc4tile.boundary_policy<tail_predicated>,
    packing = #vc4tile.packing<none>
  } : (vector<16xi32>, i32, i32, !vc4tile.predicate) -> ()
  vc4tile.return
}

// CHECK-LABEL: ssavc4.func @predicate_hardware_mapping_shared_row_tail
// CHECK-NOT: vc4tile.copy_tile
// CHECK: ssavc4.vdw.store_vpm {{.*}}, {{.*}}, {{.*}}, %{{.*}} {
// CHECK-SAME: nrows = 1 : i32
// CHECK-SAME: row_len = 16 : i32
// CHECK-NOT: active_lanes =
vc4tile.kernel @predicate_hardware_mapping_shared_row_tail(%out : i32, %n : i32) attributes {
  public_name = "predicate_hardware_mapping_shared_row_tail",
  schedule_mode = #vc4tile.schedule_mode<cooperative_block>,
  arg_attrs = [
    {abi_name = "out", kind = "buffer", direction = "out", elem_type = "u32", type = "u32"},
    {abi_name = "n", kind = "scalar", direction = "by_value", type = "u32"}
  ],
  warps_per_block_max = 1 : i32,
  uses_shared_vpm = true,
  uses_barrier = false,
  require_full_block_residency = true,
  semaphores_per_block = 4 : i32,
  vpm_rows_per_block = 1 : i32,
  vpm_bytes_per_block = 64 : i32
} {
  %zero = arith.constant 0 : i32
  %shared = "vc4tile.shared_tile_alloc"() {
    rows = 1 : i32, elem_bytes = 4 : i32, shape = [1, 16],
    element_type = i32, storage_type = i32, layout = #vc4tile.layout<vpm_row>,
    role = #vc4tile.role<scratch>, precision = #vc4tile.precision<exact_32>,
    packing = #vc4tile.packing<none>, memory_space = #vc4tile.memory_space<shared_vpm>
  } : () -> !vc4tile.shared_tile
  %tail = vc4tile.tail_mask %zero, %n : i32, i32
  "vc4tile.copy_tile"(%shared, %out, %zero, %tail) {
    shape = [1, 16],
    src_space = #vc4tile.memory_space<shared_vpm>,
    dst_space = #vc4tile.memory_space<global>,
    src_layout = #vc4tile.layout<vpm_row>,
    dst_layout = #vc4tile.layout<row_major>,
    element_type = i32, storage_type = i32,
    precision = #vc4tile.precision<exact_32>,
    packing = #vc4tile.packing<none>,
    elem_bytes = 4 : i32,
    memory_pitch_bytes = 64 : i32
  } : (!vc4tile.shared_tile, i32, i32, !vc4tile.predicate) -> ()
  vc4tile.return
}

// CHECK-LABEL: ssavc4.func @predicate_hardware_mapping_register_shared_preserve
// CHECK-NOT: vc4tile.copy_tile
// CHECK: ssavc4.vpm.read
// CHECK: ssavc4.cond_select
// CHECK: ssavc4.vpm.write
vc4tile.kernel @predicate_hardware_mapping_register_shared_preserve(%rows : i32, %cols : i32) attributes {
  public_name = "predicate_hardware_mapping_register_shared_preserve",
  schedule_mode = #vc4tile.schedule_mode<cooperative_block>,
  arg_attrs = [
    {abi_name = "rows", kind = "scalar", direction = "by_value", type = "u32"},
    {abi_name = "cols", kind = "scalar", direction = "by_value", type = "u32"}
  ],
  warps_per_block_max = 1 : i32,
  uses_shared_vpm = true,
  uses_barrier = false,
  require_full_block_residency = true,
  semaphores_per_block = 4 : i32,
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
  %mask = vc4tile.tile_bounds_mask %rows, %cols {shape = [4, 4], layout = #vc4tile.layout<row_major>} : i32, i32
  "vc4tile.copy_tile"(%values, %shared, %zero, %mask) {
    shape = [4, 4],
    src_space = #vc4tile.memory_space<register>,
    dst_space = #vc4tile.memory_space<shared_vpm>,
    src_layout = #vc4tile.layout<row_major>,
    dst_layout = #vc4tile.layout<vpm_row>,
    element_type = i32, storage_type = i32,
    precision = #vc4tile.precision<exact_32>,
    packing = #vc4tile.packing<none>,
    elem_bytes = 4 : i32
  } : (vector<16xi32>, !vc4tile.shared_tile, i32, !vc4tile.predicate) -> ()
  vc4tile.return
}

// CHECK-LABEL: ssavc4.func @predicate_hardware_mapping_shared_register_zero_fill
// CHECK-NOT: vc4tile.copy_tile
// CHECK: ssavc4.vpm.read
// CHECK: ssavc4.cond_select
vc4tile.kernel @predicate_hardware_mapping_shared_register_zero_fill(%out : i32, %rows : i32, %cols : i32) attributes {
  public_name = "predicate_hardware_mapping_shared_register_zero_fill",
  schedule_mode = #vc4tile.schedule_mode<cooperative_block>,
  arg_attrs = [
    {abi_name = "out", kind = "buffer", direction = "out", elem_type = "u32", type = "u32"},
    {abi_name = "rows", kind = "scalar", direction = "by_value", type = "u32"},
    {abi_name = "cols", kind = "scalar", direction = "by_value", type = "u32"}
  ],
  warps_per_block_max = 1 : i32,
  uses_shared_vpm = true,
  uses_barrier = false,
  require_full_block_residency = true,
  semaphores_per_block = 4 : i32,
  vpm_rows_per_block = 1 : i32,
  vpm_bytes_per_block = 64 : i32
} {
  %zero = arith.constant 0 : i32
  %shared = "vc4tile.shared_tile_alloc"() {
    rows = 1 : i32, elem_bytes = 4 : i32, shape = [1, 16],
    element_type = i32, storage_type = i32, layout = #vc4tile.layout<vpm_row>,
    role = #vc4tile.role<scratch>, precision = #vc4tile.precision<exact_32>,
    packing = #vc4tile.packing<none>, memory_space = #vc4tile.memory_space<shared_vpm>
  } : () -> !vc4tile.shared_tile
  %mask = vc4tile.tile_bounds_mask %rows, %cols {shape = [4, 4], layout = #vc4tile.layout<row_major>} : i32, i32
  %loaded = "vc4tile.copy_tile"(%shared, %zero, %mask) {
    shape = [4, 4],
    src_space = #vc4tile.memory_space<shared_vpm>,
    dst_space = #vc4tile.memory_space<register>,
    src_layout = #vc4tile.layout<vpm_row>,
    dst_layout = #vc4tile.layout<row_major>,
    element_type = i32, storage_type = i32,
    precision = #vc4tile.precision<exact_32>,
    packing = #vc4tile.packing<none>,
    elem_bytes = 4 : i32
  } : (!vc4tile.shared_tile, i32, !vc4tile.predicate) -> vector<16xi32>
  %all = vc4tile.mask_all
  "vc4tile.tile_store"(%loaded, %out, %zero, %all) {
    shape = [1, 16],
    dst_layout = #vc4tile.layout<row_major>,
    memory_space = #vc4tile.memory_space<global>,
    element_type = i32, storage_type = i32,
    precision = #vc4tile.precision<exact_32>,
    packing = #vc4tile.packing<none>
  } : (vector<16xi32>, i32, i32, !vc4tile.predicate) -> ()
  vc4tile.return
}
