// RUN: vc4-opt %s --canonicalize-vc4tile-surface --plan-vc4tile-copies -split-input-file -verify-diagnostics

vc4tile.kernel @predicate_algebra_and_true(%in : i32, %n : i32) attributes {
  public_name = "predicate_algebra_and_true",
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
  %shared = "vc4tile.shared_tile_alloc"() {rows = 4 : i32, elem_bytes = 4 : i32, shape = [4, 16], element_type = i32, storage_type = i32, layout = #vc4tile.layout<vpm_row>, role = #vc4tile.role<scratch>, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, memory_space = #vc4tile.memory_space<shared_vpm>} : () -> !vc4tile.shared_tile
  %tail = vc4tile.tail_mask %zero, %n : i32, i32 -> vector<16xi1>
  %all = vc4tile.mask_all : vector<16xi1>
  %mask = vc4tile.mask_and %tail, %all : vector<16xi1>, vector<16xi1> -> vector<16xi1>
  // expected-error@+1 {{planned_fragment_class = row_fragment}}
  "vc4tile.copy_tile"(%in, %shared, %zero, %mask) {shape = [4, 4], src_space = #vc4tile.memory_space<global>, dst_space = #vc4tile.memory_space<shared_vpm>, src_layout = #vc4tile.layout<row_major>, dst_layout = #vc4tile.layout<vpm_row>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, elem_bytes = 4 : i32, memory_pitch_bytes = 16 : i32} : (i32, !vc4tile.shared_tile, i32, vector<16xi1>) -> ()
  vc4tile.return
}

// -----

vc4tile.kernel @predicate_algebra_or_false(%in : i32, %n : i32) attributes {
  public_name = "predicate_algebra_or_false",
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
  %shared = "vc4tile.shared_tile_alloc"() {rows = 4 : i32, elem_bytes = 4 : i32, shape = [4, 16], element_type = i32, storage_type = i32, layout = #vc4tile.layout<vpm_row>, role = #vc4tile.role<scratch>, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, memory_space = #vc4tile.memory_space<shared_vpm>} : () -> !vc4tile.shared_tile
  %tail = vc4tile.tail_mask %zero, %n : i32, i32 -> vector<16xi1>
  %all = vc4tile.mask_all : vector<16xi1>
  %false = vc4tile.mask_not %all : vector<16xi1> -> vector<16xi1>
  %mask = vc4tile.mask_or %false, %tail : vector<16xi1>, vector<16xi1> -> vector<16xi1>
  // expected-error@+1 {{planned_fragment_class = row_fragment}}
  "vc4tile.copy_tile"(%in, %shared, %zero, %mask) {shape = [4, 4], src_space = #vc4tile.memory_space<global>, dst_space = #vc4tile.memory_space<shared_vpm>, src_layout = #vc4tile.layout<row_major>, dst_layout = #vc4tile.layout<vpm_row>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, elem_bytes = 4 : i32, memory_pitch_bytes = 16 : i32} : (i32, !vc4tile.shared_tile, i32, vector<16xi1>) -> ()
  vc4tile.return
}

// -----

vc4tile.kernel @predicate_algebra_and_false(%in : i32, %n : i32) attributes {
  public_name = "predicate_algebra_and_false",
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
  %shared = "vc4tile.shared_tile_alloc"() {rows = 4 : i32, elem_bytes = 4 : i32, shape = [4, 16], element_type = i32, storage_type = i32, layout = #vc4tile.layout<vpm_row>, role = #vc4tile.role<scratch>, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, memory_space = #vc4tile.memory_space<shared_vpm>} : () -> !vc4tile.shared_tile
  %tail = vc4tile.tail_mask %zero, %n : i32, i32 -> vector<16xi1>
  %all = vc4tile.mask_all : vector<16xi1>
  %false = vc4tile.mask_not %all : vector<16xi1> -> vector<16xi1>
  %mask = vc4tile.mask_and %tail, %false : vector<16xi1>, vector<16xi1> -> vector<16xi1>
  // expected-error@+1 {{planned_fragment_class = empty}}
  "vc4tile.copy_tile"(%in, %shared, %zero, %mask) {shape = [4, 4], src_space = #vc4tile.memory_space<global>, dst_space = #vc4tile.memory_space<shared_vpm>, src_layout = #vc4tile.layout<row_major>, dst_layout = #vc4tile.layout<vpm_row>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, elem_bytes = 4 : i32, memory_pitch_bytes = 16 : i32} : (i32, !vc4tile.shared_tile, i32, vector<16xi1>) -> ()
  vc4tile.return
}

// -----

vc4tile.kernel @predicate_algebra_or_true(%in : i32, %n : i32) attributes {
  public_name = "predicate_algebra_or_true",
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
  %shared = "vc4tile.shared_tile_alloc"() {rows = 4 : i32, elem_bytes = 4 : i32, shape = [4, 16], element_type = i32, storage_type = i32, layout = #vc4tile.layout<vpm_row>, role = #vc4tile.role<scratch>, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, memory_space = #vc4tile.memory_space<shared_vpm>} : () -> !vc4tile.shared_tile
  %tail = vc4tile.tail_mask %zero, %n : i32, i32 -> vector<16xi1>
  %all = vc4tile.mask_all : vector<16xi1>
  %mask = vc4tile.mask_or %tail, %all : vector<16xi1>, vector<16xi1> -> vector<16xi1>
  "vc4tile.copy_tile"(%in, %shared, %zero, %mask) {shape = [4, 4], src_space = #vc4tile.memory_space<global>, dst_space = #vc4tile.memory_space<shared_vpm>, src_layout = #vc4tile.layout<row_major>, dst_layout = #vc4tile.layout<vpm_row>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, elem_bytes = 4 : i32, memory_pitch_bytes = 16 : i32} : (i32, !vc4tile.shared_tile, i32, vector<16xi1>) -> ()
  vc4tile.return
}

// -----

vc4tile.kernel @predicate_algebra_double_not(%in : i32) attributes {
  public_name = "predicate_algebra_double_not",
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
  %shared = "vc4tile.shared_tile_alloc"() {rows = 4 : i32, elem_bytes = 4 : i32, shape = [4, 16], element_type = i32, storage_type = i32, layout = #vc4tile.layout<vpm_row>, role = #vc4tile.role<scratch>, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, memory_space = #vc4tile.memory_space<shared_vpm>} : () -> !vc4tile.shared_tile
  %rect = vc4tile.tile_rect_mask {active_rows = 3 : i32, active_cols = 2 : i32, shape = [4, 4], layout = #vc4tile.layout<row_major>} : vector<16xi1>
  %not_rect = vc4tile.mask_not %rect : vector<16xi1> -> vector<16xi1>
  %mask = vc4tile.mask_not %not_rect : vector<16xi1> -> vector<16xi1>
  // expected-error@+1 {{planned_fragment_class = row_set_fragment}}
  "vc4tile.copy_tile"(%in, %shared, %zero, %mask) {shape = [4, 4], src_space = #vc4tile.memory_space<global>, dst_space = #vc4tile.memory_space<shared_vpm>, src_layout = #vc4tile.layout<row_major>, dst_layout = #vc4tile.layout<vpm_row>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, elem_bytes = 4 : i32, memory_pitch_bytes = 16 : i32} : (i32, !vc4tile.shared_tile, i32, vector<16xi1>) -> ()
  vc4tile.return
}

// -----

vc4tile.kernel @predicate_algebra_idempotent(%in : i32, %rows : i32, %cols : i32) attributes {
  public_name = "predicate_algebra_idempotent",
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
  %shared = "vc4tile.shared_tile_alloc"() {rows = 4 : i32, elem_bytes = 4 : i32, shape = [4, 16], element_type = i32, storage_type = i32, layout = #vc4tile.layout<vpm_row>, role = #vc4tile.role<scratch>, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, memory_space = #vc4tile.memory_space<shared_vpm>} : () -> !vc4tile.shared_tile
  %bounds = vc4tile.tile_bounds_mask %rows, %cols {shape = [4, 4], layout = #vc4tile.layout<row_major>} : i32, i32 -> vector<16xi1>
  %mask = vc4tile.mask_and %bounds, %bounds : vector<16xi1>, vector<16xi1> -> vector<16xi1>
  // expected-error@+1 {{planned_fragment_class = row_set_fragment}}
  "vc4tile.copy_tile"(%in, %shared, %zero, %mask) {shape = [4, 4], src_space = #vc4tile.memory_space<global>, dst_space = #vc4tile.memory_space<shared_vpm>, src_layout = #vc4tile.layout<row_major>, dst_layout = #vc4tile.layout<vpm_row>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, elem_bytes = 4 : i32, memory_pitch_bytes = 16 : i32} : (i32, !vc4tile.shared_tile, i32, vector<16xi1>) -> ()
  vc4tile.return
}

// -----

vc4tile.kernel @predicate_algebra_idempotent_or(%in : i32, %rows : i32, %cols : i32) attributes {
  public_name = "predicate_algebra_idempotent_or",
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
  %shared = "vc4tile.shared_tile_alloc"() {rows = 4 : i32, elem_bytes = 4 : i32, shape = [4, 16], element_type = i32, storage_type = i32, layout = #vc4tile.layout<vpm_row>, role = #vc4tile.role<scratch>, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, memory_space = #vc4tile.memory_space<shared_vpm>} : () -> !vc4tile.shared_tile
  %bounds = vc4tile.tile_bounds_mask %rows, %cols {shape = [4, 4], layout = #vc4tile.layout<row_major>} : i32, i32 -> vector<16xi1>
  %mask = vc4tile.mask_or %bounds, %bounds : vector<16xi1>, vector<16xi1> -> vector<16xi1>
  // expected-error@+1 {{planned_fragment_class = row_set_fragment}}
  "vc4tile.copy_tile"(%in, %shared, %zero, %mask) {shape = [4, 4], src_space = #vc4tile.memory_space<global>, dst_space = #vc4tile.memory_space<shared_vpm>, src_layout = #vc4tile.layout<row_major>, dst_layout = #vc4tile.layout<vpm_row>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, elem_bytes = 4 : i32, memory_pitch_bytes = 16 : i32} : (i32, !vc4tile.shared_tile, i32, vector<16xi1>) -> ()
  vc4tile.return
}

// -----

vc4tile.kernel @predicate_algebra_nested_commuted_bounds_tail(%in : i32, %rows : i32, %cols : i32, %n : i32) attributes {
  public_name = "predicate_algebra_nested_commuted_bounds_tail",
  schedule_mode = #vc4tile.schedule_mode<cooperative_block>,
  arg_attrs = [
    {abi_name = "in", kind = "buffer", direction = "in", elem_type = "u32", type = "u32"},
    {abi_name = "rows", kind = "scalar", direction = "by_value", type = "u32"},
    {abi_name = "cols", kind = "scalar", direction = "by_value", type = "u32"},
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
  %shared = "vc4tile.shared_tile_alloc"() {rows = 4 : i32, elem_bytes = 4 : i32, shape = [4, 16], element_type = i32, storage_type = i32, layout = #vc4tile.layout<vpm_row>, role = #vc4tile.role<scratch>, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, memory_space = #vc4tile.memory_space<shared_vpm>} : () -> !vc4tile.shared_tile
  %bounds = vc4tile.tile_bounds_mask %rows, %cols {shape = [4, 4], layout = #vc4tile.layout<row_major>} : i32, i32 -> vector<16xi1>
  %tail = vc4tile.tail_mask %zero, %n : i32, i32 -> vector<16xi1>
  %all = vc4tile.mask_all : vector<16xi1>
  %nested = vc4tile.mask_and %bounds, %all : vector<16xi1>, vector<16xi1> -> vector<16xi1>
  %mask = vc4tile.mask_and %tail, %nested : vector<16xi1>, vector<16xi1> -> vector<16xi1>
  // expected-error@+1 {{planned_fragment_class = row_set_fragment}}
  "vc4tile.copy_tile"(%in, %shared, %zero, %mask) {shape = [4, 4], src_space = #vc4tile.memory_space<global>, dst_space = #vc4tile.memory_space<shared_vpm>, src_layout = #vc4tile.layout<row_major>, dst_layout = #vc4tile.layout<vpm_row>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, elem_bytes = 4 : i32, memory_pitch_bytes = 16 : i32} : (i32, !vc4tile.shared_tile, i32, vector<16xi1>) -> ()
  vc4tile.return
}

// -----

vc4tile.kernel @predicate_algebra_bounds_or_tail(%in : i32, %rows : i32, %cols : i32, %n : i32) attributes {
  public_name = "predicate_algebra_bounds_or_tail",
  schedule_mode = #vc4tile.schedule_mode<cooperative_block>,
  arg_attrs = [
    {abi_name = "in", kind = "buffer", direction = "in", elem_type = "u32", type = "u32"},
    {abi_name = "rows", kind = "scalar", direction = "by_value", type = "u32"},
    {abi_name = "cols", kind = "scalar", direction = "by_value", type = "u32"},
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
  %shared = "vc4tile.shared_tile_alloc"() {rows = 4 : i32, elem_bytes = 4 : i32, shape = [4, 16], element_type = i32, storage_type = i32, layout = #vc4tile.layout<vpm_row>, role = #vc4tile.role<scratch>, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, memory_space = #vc4tile.memory_space<shared_vpm>} : () -> !vc4tile.shared_tile
  %bounds = vc4tile.tile_bounds_mask %rows, %cols {shape = [4, 4], layout = #vc4tile.layout<row_major>} : i32, i32 -> vector<16xi1>
  %tail = vc4tile.tail_mask %zero, %n : i32, i32 -> vector<16xi1>
  %mask = vc4tile.mask_or %tail, %bounds : vector<16xi1>, vector<16xi1> -> vector<16xi1>
  // expected-error@+1 {{planned_fragment_class = row_set_fragment}}
  "vc4tile.copy_tile"(%in, %shared, %zero, %mask) {shape = [4, 4], src_space = #vc4tile.memory_space<global>, dst_space = #vc4tile.memory_space<shared_vpm>, src_layout = #vc4tile.layout<row_major>, dst_layout = #vc4tile.layout<vpm_row>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, elem_bytes = 4 : i32, memory_pitch_bytes = 16 : i32} : (i32, !vc4tile.shared_tile, i32, vector<16xi1>) -> ()
  vc4tile.return
}

// -----

vc4tile.kernel @predicate_algebra_tail_and_not_bounds(%in : i32, %rows : i32, %cols : i32, %n : i32) attributes {
  public_name = "predicate_algebra_tail_and_not_bounds",
  schedule_mode = #vc4tile.schedule_mode<cooperative_block>,
  arg_attrs = [
    {abi_name = "in", kind = "buffer", direction = "in", elem_type = "u32", type = "u32"},
    {abi_name = "rows", kind = "scalar", direction = "by_value", type = "u32"},
    {abi_name = "cols", kind = "scalar", direction = "by_value", type = "u32"},
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
  %shared = "vc4tile.shared_tile_alloc"() {rows = 4 : i32, elem_bytes = 4 : i32, shape = [4, 16], element_type = i32, storage_type = i32, layout = #vc4tile.layout<vpm_row>, role = #vc4tile.role<scratch>, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, memory_space = #vc4tile.memory_space<shared_vpm>} : () -> !vc4tile.shared_tile
  %bounds = vc4tile.tile_bounds_mask %rows, %cols {shape = [4, 4], layout = #vc4tile.layout<row_major>} : i32, i32 -> vector<16xi1>
  %tail = vc4tile.tail_mask %zero, %n : i32, i32 -> vector<16xi1>
  %not_bounds = vc4tile.mask_not %bounds : vector<16xi1> -> vector<16xi1>
  %mask = vc4tile.mask_and %tail, %not_bounds : vector<16xi1>, vector<16xi1> -> vector<16xi1>
  // expected-error@+1 {{planned_fragment_class = row_set_fragment}}
  "vc4tile.copy_tile"(%in, %shared, %zero, %mask) {shape = [4, 4], src_space = #vc4tile.memory_space<global>, dst_space = #vc4tile.memory_space<shared_vpm>, src_layout = #vc4tile.layout<row_major>, dst_layout = #vc4tile.layout<vpm_row>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, elem_bytes = 4 : i32, memory_pitch_bytes = 16 : i32} : (i32, !vc4tile.shared_tile, i32, vector<16xi1>) -> ()
  vc4tile.return
}

// -----

vc4tile.kernel @predicate_algebra_rect_and_bounds(%in : i32, %rows : i32, %cols : i32) attributes {
  public_name = "predicate_algebra_rect_and_bounds",
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
  %shared = "vc4tile.shared_tile_alloc"() {rows = 4 : i32, elem_bytes = 4 : i32, shape = [4, 16], element_type = i32, storage_type = i32, layout = #vc4tile.layout<vpm_row>, role = #vc4tile.role<scratch>, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, memory_space = #vc4tile.memory_space<shared_vpm>} : () -> !vc4tile.shared_tile
  %rect = vc4tile.tile_rect_mask {active_rows = 3 : i32, active_cols = 2 : i32, shape = [4, 4], layout = #vc4tile.layout<row_major>} : vector<16xi1>
  %bounds = vc4tile.tile_bounds_mask %rows, %cols {shape = [4, 4], layout = #vc4tile.layout<row_major>} : i32, i32 -> vector<16xi1>
  %mask = vc4tile.mask_and %rect, %bounds : vector<16xi1>, vector<16xi1> -> vector<16xi1>
  // expected-error@+1 {{planned_fragment_class = row_set_fragment}}
  "vc4tile.copy_tile"(%in, %shared, %zero, %mask) {shape = [4, 4], src_space = #vc4tile.memory_space<global>, dst_space = #vc4tile.memory_space<shared_vpm>, src_layout = #vc4tile.layout<row_major>, dst_layout = #vc4tile.layout<vpm_row>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, elem_bytes = 4 : i32, memory_pitch_bytes = 16 : i32} : (i32, !vc4tile.shared_tile, i32, vector<16xi1>) -> ()
  vc4tile.return
}

// -----

vc4tile.kernel @predicate_algebra_rect_and_tail(%in : i32, %n : i32) attributes {
  public_name = "predicate_algebra_rect_and_tail",
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
  %shared = "vc4tile.shared_tile_alloc"() {rows = 4 : i32, elem_bytes = 4 : i32, shape = [4, 16], element_type = i32, storage_type = i32, layout = #vc4tile.layout<vpm_row>, role = #vc4tile.role<scratch>, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, memory_space = #vc4tile.memory_space<shared_vpm>} : () -> !vc4tile.shared_tile
  %rect = vc4tile.tile_rect_mask {active_rows = 3 : i32, active_cols = 2 : i32, shape = [4, 4], layout = #vc4tile.layout<row_major>} : vector<16xi1>
  %tail = vc4tile.tail_mask %zero, %n : i32, i32 -> vector<16xi1>
  %mask = vc4tile.mask_and %rect, %tail : vector<16xi1>, vector<16xi1> -> vector<16xi1>
  // expected-error@+1 {{planned_fragment_class = row_set_fragment}}
  "vc4tile.copy_tile"(%in, %shared, %zero, %mask) {shape = [4, 4], src_space = #vc4tile.memory_space<global>, dst_space = #vc4tile.memory_space<shared_vpm>, src_layout = #vc4tile.layout<row_major>, dst_layout = #vc4tile.layout<vpm_row>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, elem_bytes = 4 : i32, memory_pitch_bytes = 16 : i32} : (i32, !vc4tile.shared_tile, i32, vector<16xi1>) -> ()
  vc4tile.return
}

// -----

vc4tile.kernel @predicate_algebra_sparse_disabled(%in : i32, %n : i32) attributes {
  public_name = "predicate_algebra_sparse_disabled",
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
  %shared = "vc4tile.shared_tile_alloc"() {rows = 4 : i32, elem_bytes = 4 : i32, shape = [4, 16], element_type = i32, storage_type = i32, layout = #vc4tile.layout<vpm_row>, role = #vc4tile.role<scratch>, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, memory_space = #vc4tile.memory_space<shared_vpm>} : () -> !vc4tile.shared_tile
  %tail = vc4tile.tail_mask %zero, %n : i32, i32 -> vector<16xi1>
  %mask = vc4tile.mask_not %tail : vector<16xi1> -> vector<16xi1>
  // expected-error@+1 {{unsupported predicate fragment plan: cannot normalize predicate algebra into dense fragments and sparse fallback is disabled}}
  "vc4tile.copy_tile"(%in, %shared, %zero, %mask) {shape = [4, 4], src_space = #vc4tile.memory_space<global>, dst_space = #vc4tile.memory_space<shared_vpm>, src_layout = #vc4tile.layout<row_major>, dst_layout = #vc4tile.layout<vpm_row>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, elem_bytes = 4 : i32, memory_pitch_bytes = 16 : i32} : (i32, !vc4tile.shared_tile, i32, vector<16xi1>) -> ()
  vc4tile.return
}
