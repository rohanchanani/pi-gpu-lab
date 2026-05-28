// RUN: vc4-opt %s --canonicalize-vc4tile-surface --plan-vc4tile-copies -split-input-file -verify-diagnostics

vc4tile.kernel @predicate_hardware_mapping_reject_sparse_store(%out : i32, %n : i32) attributes {
  public_name = "predicate_hardware_mapping_reject_sparse_store",
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
  %tail = vc4tile.tail_mask %zero, %n : i32, i32 -> vector<16xi1>
  %not_tail = vc4tile.mask_not %tail : vector<16xi1> -> vector<16xi1>
  // expected-error@+1 {{register->global tile_store unsupported predicate fragment plan: predicate normalization could not produce dense fragments and sparse fallback is disabled}}
  "vc4tile.tile_store"(%values, %out, %zero, %not_tail) {
    shape = [1, 16],
    layout = #vc4tile.layout<row_major>,
    memory_space = #vc4tile.memory_space<global>,
    element_type = i32, storage_type = i32,
    precision = #vc4tile.precision<exact_32>,
    boundary = #vc4tile.boundary_policy<exact>,
    packing = #vc4tile.packing<none>
  } : (vector<16xi32>, i32, i32, vector<16xi1>) -> ()
  vc4tile.return
}

// -----

vc4tile.kernel @predicate_hardware_mapping_reject_dynamic_2d(%in : i32, %rows : i32, %cols : i32) attributes {
  public_name = "predicate_hardware_mapping_reject_dynamic_2d",
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
  %bounds = vc4tile.tile_bounds_mask %rows, %cols {shape = [4, 4], layout = #vc4tile.layout<row_major>} : i32, i32 -> vector<16xi1>
  %not_bounds = vc4tile.mask_not %bounds : vector<16xi1> -> vector<16xi1>
  // expected-error@+1 {{global->shared_vpm copy_tile unsupported predicate fragment plan: predicate normalization could not produce dense fragments and sparse fallback is disabled}}
  "vc4tile.copy_tile"(%in, %shared, %zero, %not_bounds) {
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
  } : (i32, !vc4tile.shared_tile, i32, vector<16xi1>) -> ()
  vc4tile.return
}
