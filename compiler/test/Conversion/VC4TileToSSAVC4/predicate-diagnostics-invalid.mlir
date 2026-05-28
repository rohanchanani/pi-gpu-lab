// RUN: vc4-opt %s --canonicalize-vc4tile-surface --plan-vc4tile-copies -split-input-file -verify-diagnostics

vc4tile.kernel @predicate_load_zero_fill_layout_diag(%in : i32, %rows : i32, %cols : i32) attributes {
  public_name = "predicate_load_zero_fill_layout_diag",
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
    element_type = i32, storage_type = i32, layout = #vc4tile.layout<vpm_col>,
    role = #vc4tile.role<scratch>, precision = #vc4tile.precision<exact_32>,
    packing = #vc4tile.packing<none>, memory_space = #vc4tile.memory_space<shared_vpm>
  } : () -> !vc4tile.shared_tile
  %mask = vc4tile.tile_bounds_mask %rows, %cols {shape = [4, 4], layout = #vc4tile.layout<row_major>} : i32, i32 -> vector<16xi1>
  // expected-error@+1 {{load predicate requires inactive zero-fill}}
  "vc4tile.copy_tile"(%in, %shared, %zero, %mask) {
    shape = [4, 4],
    src_space = #vc4tile.memory_space<global>,
    dst_space = #vc4tile.memory_space<shared_vpm>,
    src_layout = #vc4tile.layout<row_major>,
    dst_layout = #vc4tile.layout<vpm_col>,
    element_type = i32, storage_type = i32,
    precision = #vc4tile.precision<exact_32>,
    packing = #vc4tile.packing<none>,
    elem_bytes = 4 : i32,
    memory_pitch_bytes = 16 : i32
  } : (i32, !vc4tile.shared_tile, i32, vector<16xi1>) -> ()
  vc4tile.return
}

// -----

vc4tile.kernel @predicate_store_preserve_layout_diag(%out : i32, %rows : i32, %cols : i32) attributes {
  public_name = "predicate_store_preserve_layout_diag",
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
  vpm_rows_per_block = 4 : i32,
  vpm_bytes_per_block = 256 : i32
} {
  %zero = arith.constant 0 : i32
  %shared = "vc4tile.shared_tile_alloc"() {
    rows = 4 : i32, elem_bytes = 4 : i32, shape = [4, 16],
    element_type = i32, storage_type = i32, layout = #vc4tile.layout<vpm_col>,
    role = #vc4tile.role<scratch>, precision = #vc4tile.precision<exact_32>,
    packing = #vc4tile.packing<none>, memory_space = #vc4tile.memory_space<shared_vpm>
  } : () -> !vc4tile.shared_tile
  %mask = vc4tile.tile_bounds_mask %rows, %cols {shape = [4, 4], layout = #vc4tile.layout<row_major>} : i32, i32 -> vector<16xi1>
  // expected-error@+1 {{store predicate requires inactive global destination preservation}}
  "vc4tile.copy_tile"(%shared, %zero, %out, %zero, %mask) {
    shape = [4, 4],
    src_space = #vc4tile.memory_space<shared_vpm>,
    dst_space = #vc4tile.memory_space<global>,
    src_layout = #vc4tile.layout<col_major>,
    dst_layout = #vc4tile.layout<row_major>,
    element_type = i32, storage_type = i32,
    precision = #vc4tile.precision<exact_32>,
    packing = #vc4tile.packing<none>,
    elem_bytes = 4 : i32,
    memory_pitch_bytes = 16 : i32
  } : (!vc4tile.shared_tile, i32, i32, i32, vector<16xi1>) -> ()
  vc4tile.return
}

// -----

vc4tile.kernel @predicate_dynamic_vpm_alignment_diag(%out : i32, %row : i32) attributes {
  public_name = "predicate_dynamic_vpm_alignment_diag",
  schedule_mode = #vc4tile.schedule_mode<cooperative_block>,
  arg_attrs = [
    {abi_name = "out", kind = "buffer", direction = "out", elem_type = "u32", type = "u32"},
    {abi_name = "row", kind = "scalar", direction = "by_value", type = "u32"}
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
  %shared = "vc4tile.shared_tile_alloc"() {
    rows = 16 : i32, elem_bytes = 4 : i32, shape = [16, 16],
    element_type = i32, storage_type = i32, layout = #vc4tile.layout<vpm_col>,
    role = #vc4tile.role<scratch>, precision = #vc4tile.precision<exact_32>,
    packing = #vc4tile.packing<none>, memory_space = #vc4tile.memory_space<shared_vpm>
  } : () -> !vc4tile.shared_tile
  %mask = vc4tile.mask_all : vector<16xi1>
  // expected-error@+1 {{predicate fragment planning would require dynamic VPM alignment}}
  "vc4tile.copy_tile"(%shared, %row, %out, %zero, %mask) {
    shape = [2, 16],
    src_space = #vc4tile.memory_space<shared_vpm>,
    dst_space = #vc4tile.memory_space<global>,
    src_layout = #vc4tile.layout<col_major>,
    dst_layout = #vc4tile.layout<row_major>,
    element_type = i32, storage_type = i32,
    precision = #vc4tile.precision<exact_32>,
    packing = #vc4tile.packing<none>,
    elem_bytes = 4 : i32,
    memory_pitch_bytes = 64 : i32
  } : (!vc4tile.shared_tile, i32, i32, i32, vector<16xi1>) -> ()
  vc4tile.return
}
