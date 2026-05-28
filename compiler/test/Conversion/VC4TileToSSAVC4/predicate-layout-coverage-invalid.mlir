// RUN: vc4-opt %s --canonicalize-vc4tile-surface --plan-vc4tile-copies -split-input-file -verify-diagnostics

vc4tile.kernel @predicate_layout_reject_global_register_col_major(%in : i32, %n : i32) attributes {
  public_name = "predicate_layout_reject_global_register_col_major",
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
  %tail = vc4tile.tail_mask %zero, %n : i32, i32
  // expected-error@+1 {{global->register tile_load unsupported predicate fragment plan: unsupported layout + predicate combination for global->register tile_load: load predicate requires inactive zero-fill over a row-major or affine_2d logical source}}
  %tile = "vc4tile.tile_load"(%in, %zero, %tail) {
    shape = [1, 16],
    src_layout = #vc4tile.layout<col_major>,
    memory_space = #vc4tile.memory_space<global>,
    element_type = i32,
    storage_type = i32,
    precision = #vc4tile.precision<exact_32>,
    boundary = #vc4tile.boundary_policy<tail_predicated>,
    packing = #vc4tile.packing<none>
  } : (i32, i32, !vc4tile.predicate) -> vector<16xi32>
  vc4tile.return
}

// -----

vc4tile.kernel @predicate_layout_reject_register_global_affine(%out : i32, %n : i32) attributes {
  public_name = "predicate_layout_reject_register_global_affine",
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
  // expected-error@+1 {{register->global tile_store unsupported predicate fragment plan: unsupported layout + predicate combination for register->global tile_store: store predicate requires inactive destination preservation over a row-major logical destination}}
  "vc4tile.tile_store"(%values, %out, %zero, %tail) {
    shape = [1, 16],
    dst_layout = #vc4tile.layout<affine_2d>,
    strides = [2],
    memory_space = #vc4tile.memory_space<global>,
    element_type = i32,
    storage_type = i32,
    precision = #vc4tile.precision<exact_32>,
    boundary = #vc4tile.boundary_policy<tail_predicated>,
    packing = #vc4tile.packing<none>
  } : (vector<16xi32>, i32, i32, !vc4tile.predicate) -> ()
  vc4tile.return
}

// -----

vc4tile.kernel @predicate_layout_reject_global_shared_col_major_source(%in : i32, %rows : i32, %cols : i32) attributes {
  public_name = "predicate_layout_reject_global_shared_col_major_source",
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
  %mask = vc4tile.tile_bounds_mask %rows, %cols {shape = [4, 4], layout = #vc4tile.layout<row_major>} : i32, i32
  // expected-error@+1 {{unsupported layout + predicate combination: semantic predicate layout mismatch; predicate layout #vc4tile.layout<row_major> does not match consumer src_layout #vc4tile.layout<col_major>}}
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

vc4tile.kernel @predicate_layout_reject_shared_global_transposed_nonfull(%out : i32, %rows : i32, %cols : i32) attributes {
  public_name = "predicate_layout_reject_shared_global_transposed_nonfull",
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
  vpm_rows_per_block = 16 : i32,
  vpm_bytes_per_block = 1024 : i32
} {
  %zero = arith.constant 0 : i32
  %shared = "vc4tile.shared_tile_alloc"() {
    rows = 16 : i32, elem_bytes = 4 : i32, shape = [16, 16],
    element_type = i32, storage_type = i32, layout = #vc4tile.layout<vpm_row>,
    role = #vc4tile.role<scratch>, precision = #vc4tile.precision<exact_32>,
    packing = #vc4tile.packing<none>, memory_space = #vc4tile.memory_space<shared_vpm>
  } : () -> !vc4tile.shared_tile
  %mask = vc4tile.tile_bounds_mask %rows, %cols {shape = [4, 4], layout = #vc4tile.layout<row_major>} : i32, i32
  %view = "vc4tile.transpose_view"(%shared) {permutation = [1, 0]} : (!vc4tile.shared_tile) -> !vc4tile.shared_tile
  // expected-error@+1 {{shared_vpm->global copy_tile unsupported predicate fragment plan: transposed or column-major predicate planning requires explicit logical-coordinate predicate rebasing before fragment mapping}}
  "vc4tile.tile_store"(%view, %out, %zero, %mask) {
    shape = [4, 4],
    dst_layout = #vc4tile.layout<row_major>,
    memory_space = #vc4tile.memory_space<global>,
    element_type = i32,
    storage_type = i32,
    precision = #vc4tile.precision<exact_32>,
    packing = #vc4tile.packing<none>,
    shared_row = 0 : i32
  } : (!vc4tile.shared_tile, i32, i32, !vc4tile.predicate) -> ()
  vc4tile.return
}

// -----

vc4tile.kernel @predicate_layout_reject_shared_register_transposed_nonfull(%out : i32, %rows : i32, %cols : i32) attributes {
  public_name = "predicate_layout_reject_shared_register_transposed_nonfull",
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
  vpm_rows_per_block = 16 : i32,
  vpm_bytes_per_block = 1024 : i32
} {
  %zero = arith.constant 0 : i32
  %shared = "vc4tile.shared_tile_alloc"() {
    rows = 16 : i32, elem_bytes = 4 : i32, shape = [16, 16],
    element_type = i32, storage_type = i32, layout = #vc4tile.layout<vpm_row>,
    role = #vc4tile.role<scratch>, precision = #vc4tile.precision<exact_32>,
    packing = #vc4tile.packing<none>, memory_space = #vc4tile.memory_space<shared_vpm>
  } : () -> !vc4tile.shared_tile
  %mask = vc4tile.tile_bounds_mask %rows, %cols {shape = [4, 4], layout = #vc4tile.layout<row_major>} : i32, i32
  // expected-error@+1 {{shared_vpm->register copy_tile unsupported predicate fragment plan: transposed or column-major predicate planning requires explicit logical-coordinate predicate rebasing before fragment mapping}}
  %loaded = "vc4tile.copy_tile"(%shared, %zero, %mask) {
    shape = [4, 4],
    src_space = #vc4tile.memory_space<shared_vpm>,
    dst_space = #vc4tile.memory_space<register>,
    src_layout = #vc4tile.layout<transposed_view>,
    dst_layout = #vc4tile.layout<row_major>,
    element_type = i32,
    storage_type = i32,
    precision = #vc4tile.precision<exact_32>,
    packing = #vc4tile.packing<none>,
    elem_bytes = 4 : i32
  } : (!vc4tile.shared_tile, i32, !vc4tile.predicate) -> vector<16xi32>
  %all = vc4tile.mask_all
  "vc4tile.tile_store"(%loaded, %out, %zero, %all) {
    shape = [1, 16],
    dst_layout = #vc4tile.layout<row_major>,
    memory_space = #vc4tile.memory_space<global>,
    element_type = i32,
    storage_type = i32,
    precision = #vc4tile.precision<exact_32>,
    packing = #vc4tile.packing<none>
  } : (vector<16xi32>, i32, i32, !vc4tile.predicate) -> ()
  vc4tile.return
}
