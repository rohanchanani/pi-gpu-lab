// RUN: vc4-opt %s --canonicalize-vc4tile-surface --plan-vc4tile-copies --legalize-vc4tile-core-cfg --verify-vc4tile-core --convert-vc4tile-to-ssavc4 -o - | FileCheck %s

// CHECK-LABEL: ssavc4.func @predicate_layout_affine_load_tail
// CHECK-NOT: vc4tile.tile_load
// CHECK: ssavc4.tmu.request
// CHECK: ssavc4.tmu.read
vc4tile.kernel @predicate_layout_affine_load_tail(%out : i32, %in : i32, %n : i32) attributes {
  public_name = "predicate_layout_affine_load_tail",
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  arg_attrs = [
    {abi_name = "out", kind = "buffer", direction = "out", elem_type = "u32", type = "u32"},
    {abi_name = "in", kind = "buffer", direction = "in", elem_type = "u32", type = "u32"},
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
  %tail = vc4tile.tail_mask %zero, %n : i32, i32
  %tile = "vc4tile.tile_load"(%in, %zero, %tail) {
    shape = [1, 16],
    src_layout = #vc4tile.layout<affine_2d>,
    strides = [2],
    memory_space = #vc4tile.memory_space<global>,
    element_type = i32,
    storage_type = i32,
    precision = #vc4tile.precision<exact_32>,
    boundary = #vc4tile.boundary_policy<tail_predicated>,
    packing = #vc4tile.packing<none>
  } : (i32, i32, !vc4tile.predicate) -> vector<16xi32>
  %all = vc4tile.mask_all
  "vc4tile.tile_store"(%tile, %out, %zero, %all) {
    shape = [1, 16],
    dst_layout = #vc4tile.layout<row_major>,
    memory_space = #vc4tile.memory_space<global>,
    element_type = i32,
    storage_type = i32,
    precision = #vc4tile.precision<exact_32>,
    boundary = #vc4tile.boundary_policy<exact>,
    packing = #vc4tile.packing<none>
  } : (vector<16xi32>, i32, i32, !vc4tile.predicate) -> ()
  vc4tile.return
}

// CHECK-LABEL: ssavc4.func @predicate_layout_pitched_row_major_store
// CHECK-NOT: vc4tile.tile_store
// CHECK: ssavc4.vdw.store
// CHECK: ssavc4.vdw.store
vc4tile.kernel @predicate_layout_pitched_row_major_store(%out : i32, %rows : i32, %cols : i32) attributes {
  public_name = "predicate_layout_pitched_row_major_store",
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  arg_attrs = [
    {abi_name = "out", kind = "buffer", direction = "out", elem_type = "u32", type = "u32"},
    {abi_name = "rows", kind = "scalar", direction = "by_value", type = "u32"},
    {abi_name = "cols", kind = "scalar", direction = "by_value", type = "u32"}
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
  %mask = vc4tile.tile_bounds_mask %rows, %cols {shape = [4, 4], layout = #vc4tile.layout<row_major>} : i32, i32
  "vc4tile.tile_store"(%values, %out, %zero, %mask) {
    shape = [4, 4],
    dst_layout = #vc4tile.layout<row_major>,
    memory_space = #vc4tile.memory_space<global>,
    element_type = i32,
    storage_type = i32,
    precision = #vc4tile.precision<exact_32>,
    boundary = #vc4tile.boundary_policy<exact>,
    packing = #vc4tile.packing<none>,
    memory_pitch_bytes = 52 : i32
  } : (vector<16xi32>, i32, i32, !vc4tile.predicate) -> ()
  vc4tile.return
}

// CHECK-LABEL: ssavc4.func @predicate_layout_transposed_full_store
// CHECK-NOT: vc4tile.transpose_view
// CHECK-NOT: vc4tile.tile_store
// CHECK: ssavc4.vdw.store_vpm
// CHECK-SAME: orientation = "vertical"
vc4tile.kernel @predicate_layout_transposed_full_store(%out : i32, %in : i32) attributes {
  public_name = "predicate_layout_transposed_full_store",
  schedule_mode = #vc4tile.schedule_mode<cooperative_block>,
  arg_attrs = [
    {abi_name = "out", kind = "buffer", direction = "out", elem_type = "u32", type = "u32"},
    {abi_name = "in", kind = "buffer", direction = "in", elem_type = "u32", type = "u32"}
  ],
  warps_per_block_max = 1 : i32,
  uses_shared_vpm = true,
  uses_barrier = false,
  require_full_block_residency = true,
  semaphores_per_block = 4 : i32,
  vpm_rows_per_block = 16 : i32,
  vpm_bytes_per_block = 1024 : i32
} {
  %zero = arith.constant 0 : i32
  %mask = vc4tile.mask_all
  %shared = "vc4tile.shared_tile_alloc"() {
    rows = 16 : i32,
    elem_bytes = 4 : i32,
    shape = [16, 16],
    element_type = i32,
    storage_type = i32,
    layout = #vc4tile.layout<vpm_row>,
    role = #vc4tile.role<scratch>,
    precision = #vc4tile.precision<exact_32>,
    packing = #vc4tile.packing<none>,
    memory_space = #vc4tile.memory_space<shared_vpm>
  } : () -> !vc4tile.shared_tile
  "vc4tile.copy_tile"(%in, %shared, %zero) {
    shape = [16, 16],
    src_space = #vc4tile.memory_space<global>,
    dst_space = #vc4tile.memory_space<shared_vpm>,
    src_layout = #vc4tile.layout<row_major>,
    dst_layout = #vc4tile.layout<vpm_row>,
    element_type = i32,
    storage_type = i32,
    precision = #vc4tile.precision<exact_32>,
    packing = #vc4tile.packing<none>,
    elem_bytes = 4 : i32,
    memory_pitch_bytes = 64 : i32
  } : (i32, !vc4tile.shared_tile, i32) -> ()
  %view = "vc4tile.transpose_view"(%shared) {permutation = [1, 0]} : (!vc4tile.shared_tile) -> !vc4tile.shared_tile
  "vc4tile.tile_store"(%view, %out, %zero, %mask) {
    shape = [1, 16],
    dst_layout = #vc4tile.layout<row_major>,
    memory_space = #vc4tile.memory_space<global>,
    element_type = i32,
    storage_type = i32,
    precision = #vc4tile.precision<exact_32>,
    packing = #vc4tile.packing<none>
  } : (!vc4tile.shared_tile, i32, i32, !vc4tile.predicate) -> ()
  vc4tile.return
}
