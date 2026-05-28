// RUN: vc4-opt %s -split-input-file -verify-diagnostics

vc4tile.kernel @predicate_shape_mismatch(%out : i32) attributes {
  public_name = "predicate_shape_mismatch",
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  arg_attrs = [{abi_name = "out", kind = "buffer", direction = "out", elem_type = "u32", type = "u32"}],
  warps_per_block_max = 1 : i32,
  uses_shared_vpm = false,
  uses_barrier = false,
  semaphores_per_block = 0 : i32,
  vpm_rows_per_block = 0 : i32,
  vpm_bytes_per_block = 0 : i32
} {
  %zero = arith.constant 0 : i32
  %tile = arith.constant dense<0> : vector<16xi32>
  %mask = vc4tile.tile_rect_mask {active_rows = 3 : i32, active_cols = 2 : i32, shape = [4, 4], layout = #vc4tile.layout<row_major>}
  // expected-error@+1 {{semantic predicate shape mismatch: predicate shape [4, 4] does not match consumer shape [1, 16]}}
  "vc4tile.tile_store"(%tile, %out, %zero, %mask) {
    shape = [1, 16], dst_layout = #vc4tile.layout<row_major>,
    memory_space = #vc4tile.memory_space<global>,
    element_type = i32, storage_type = i32,
    precision = #vc4tile.precision<exact_32>,
    boundary = #vc4tile.boundary_policy<exact>,
    packing = #vc4tile.packing<none>
  } : (vector<16xi32>, i32, i32, !vc4tile.predicate) -> ()
  vc4tile.return
}

// -----

vc4tile.kernel @predicate_rank_mismatch_direct_tail(%in : i32, %n : i32) attributes {
  public_name = "predicate_rank_mismatch_direct_tail",
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  arg_attrs = [
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
  %mask = vc4tile.tail_mask %zero, %n : i32, i32
  // expected-error@+1 {{semantic predicate rank mismatch: vc4tile.tail_mask is a 1D tail interval}}
  %tile = "vc4tile.tile_load"(%in, %zero, %mask) {
    shape = [4, 4], src_layout = #vc4tile.layout<row_major>,
    memory_space = #vc4tile.memory_space<global>,
    element_type = i32, storage_type = i32,
    precision = #vc4tile.precision<exact_32>,
    boundary = #vc4tile.boundary_policy<tail_predicated>,
    packing = #vc4tile.packing<none>
  } : (i32, i32, !vc4tile.predicate) -> vector<16xi32>
  vc4tile.return
}

// -----

vc4tile.kernel @predicate_layout_mismatch_copy(%in : i32) attributes {
  public_name = "predicate_layout_mismatch_copy",
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
  %mask = vc4tile.tile_bounds_mask %zero, %zero {shape = [4, 4], layout = #vc4tile.layout<row_major>} : i32, i32
  // expected-error@+1 {{unsupported layout + predicate combination: semantic predicate layout mismatch}}
  "vc4tile.copy_tile"(%in, %shared, %zero, %mask) {
    shape = [4, 4],
    src_space = #vc4tile.memory_space<global>,
    dst_space = #vc4tile.memory_space<shared_vpm>,
    src_layout = #vc4tile.layout<col_major>,
    dst_layout = #vc4tile.layout<vpm_row>,
    element_type = i32, storage_type = i32,
    precision = #vc4tile.precision<exact_32>,
    packing = #vc4tile.packing<none>,
    elem_bytes = 4 : i32,
    memory_pitch_bytes = 16 : i32
  } : (i32, !vc4tile.shared_tile, i32, !vc4tile.predicate) -> ()
  vc4tile.return
}

// -----

vc4tile.kernel @predicate_dynamic_multirow_store(%out : i32, %n : i32) attributes {
  public_name = "predicate_dynamic_multirow_store",
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
  %mask = vc4tile.core_tail_mask %zero, %n : i32, i32 -> vector<16xi1>
  // expected-error@+1 {{unsupported 2D dynamic active-lane predicate}}
  vc4tile.shared_store_global %shared, %zero, %zero, %out, %zero, %mask {
    elem_bytes = 4 : i32,
    row_len = 4 : i32,
    nrows = 2 : i32,
    memory_pitch_bytes = 16 : i32,
    layout = #vc4tile.vpm_layout<row_major>
  } : !vc4tile.shared_tile, i32, i32, i32, i32, vector<16xi1>
  vc4tile.return
}

// -----

vc4tile.kernel @legacy_tile_load_layout_rejected(%in : i32) attributes {
  public_name = "legacy_tile_load_layout_rejected",
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  arg_attrs = [{abi_name = "in", kind = "buffer", direction = "in", elem_type = "u32", type = "u32"}],
  warps_per_block_max = 1 : i32,
  uses_shared_vpm = false,
  uses_barrier = false,
  semaphores_per_block = 0 : i32,
  vpm_rows_per_block = 0 : i32,
  vpm_bytes_per_block = 0 : i32
} {
  %zero = arith.constant 0 : i32
  %mask = vc4tile.mask_all
  // expected-error@+1 {{legacy movement layout attribute is not supported; use src_layout for tile_load and dst_layout for tile_store}}
  %tile = "vc4tile.tile_load"(%in, %zero, %mask) {
    shape = [1, 16],
    src_layout = #vc4tile.layout<row_major>,
    layout = #vc4tile.layout<row_major>,
    memory_space = #vc4tile.memory_space<global>,
    element_type = i32, storage_type = i32,
    precision = #vc4tile.precision<exact_32>,
    packing = #vc4tile.packing<none>
  } : (i32, i32, !vc4tile.predicate) -> vector<16xi32>
  vc4tile.return
}

// -----

vc4tile.kernel @legacy_tile_store_layout_rejected(%out : i32) attributes {
  public_name = "legacy_tile_store_layout_rejected",
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  arg_attrs = [{abi_name = "out", kind = "buffer", direction = "out", elem_type = "u32", type = "u32"}],
  warps_per_block_max = 1 : i32,
  uses_shared_vpm = false,
  uses_barrier = false,
  semaphores_per_block = 0 : i32,
  vpm_rows_per_block = 0 : i32,
  vpm_bytes_per_block = 0 : i32
} {
  %zero = arith.constant 0 : i32
  %tile = arith.constant dense<0> : vector<16xi32>
  %mask = vc4tile.mask_all
  // expected-error@+1 {{legacy movement layout attribute is not supported; use src_layout for tile_load and dst_layout for tile_store}}
  "vc4tile.tile_store"(%tile, %out, %zero, %mask) {
    shape = [1, 16],
    dst_layout = #vc4tile.layout<row_major>,
    layout = #vc4tile.layout<row_major>,
    memory_space = #vc4tile.memory_space<global>,
    element_type = i32, storage_type = i32,
    precision = #vc4tile.precision<exact_32>,
    packing = #vc4tile.packing<none>
  } : (vector<16xi32>, i32, i32, !vc4tile.predicate) -> ()
  vc4tile.return
}

// -----

vc4tile.kernel @legacy_copy_tile_source_layout_rejected(%out : i32) attributes {
  public_name = "legacy_copy_tile_source_layout_rejected",
  schedule_mode = #vc4tile.schedule_mode<cooperative_block>,
  arg_attrs = [{abi_name = "out", kind = "buffer", direction = "out", elem_type = "u32", type = "u32"}],
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
  %mask = vc4tile.mask_all
  // expected-error@+1 {{legacy source_layout attribute is not supported; use explicit src_layout/dst_layout transfer roles}}
  "vc4tile.copy_tile"(%shared, %zero, %out, %zero, %mask) {
    shape = [1, 16],
    src_space = #vc4tile.memory_space<shared_vpm>,
    dst_space = #vc4tile.memory_space<global>,
    src_layout = #vc4tile.layout<vpm_row>,
    dst_layout = #vc4tile.layout<row_major>,
    source_layout = #vc4tile.layout<vpm_row>,
    element_type = i32, storage_type = i32,
    precision = #vc4tile.precision<exact_32>,
    packing = #vc4tile.packing<none>,
    elem_bytes = 4 : i32
  } : (!vc4tile.shared_tile, i32, i32, i32, !vc4tile.predicate) -> ()
  vc4tile.return
}
