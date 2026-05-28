// RUN: vc4-opt %s --canonicalize-vc4tile-surface --plan-vc4tile-copies --legalize-vc4tile-core-cfg --verify-vc4tile-core -o - | FileCheck %s --check-prefix=CORE
// RUN: vc4-opt %s --canonicalize-vc4tile-surface --plan-vc4tile-copies --legalize-vc4tile-core-cfg --verify-vc4tile-core --convert-vc4tile-to-ssavc4 -o - | FileCheck %s --check-prefix=SSAVC4

// CORE-LABEL: vc4tile.kernel @plan_shared_transpose_view
// CORE-NOT: vc4tile.transpose_view
// CORE-NOT: vc4tile.copy_tile
// CORE: vc4tile.vdr_load_tile
// CORE: vc4tile.shared_store_global
// CORE-SAME: #vc4tile.vpm_layout<column_major>
// CORE-NOT: vc4tile.shared_load
// CORE-NOT: vc4tile.masked_store_global
// SSAVC4-LABEL: ssavc4.func @plan_shared_transpose_view
// SSAVC4: ssavc4.vdr.load
// SSAVC4: ssavc4.vdw.store_vpm
// SSAVC4-SAME: orientation = "vertical"
// SSAVC4-NOT: ssavc4.vpm.read
// SSAVC4-NOT: ssavc4.vdw.store
vc4tile.kernel @plan_shared_transpose_view(%out : i32, %in : i32) attributes {
  public_name = "plan_shared_transpose_view",
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
  %shared = "vc4tile.shared_tile_alloc"() {rows = 16 : i32, elem_bytes = 4 : i32, shape = [16, 16], element_type = i32, storage_type = i32, layout = #vc4tile.layout<vpm_row>, role = #vc4tile.role<scratch>, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, memory_space = #vc4tile.memory_space<shared_vpm>} : () -> !vc4tile.shared_tile
  "vc4tile.copy_tile"(%in, %shared, %zero) {shape = [16, 16], src_space = #vc4tile.memory_space<global>, dst_space = #vc4tile.memory_space<shared_vpm>, src_layout = #vc4tile.layout<row_major>, dst_layout = #vc4tile.layout<vpm_row>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, elem_bytes = 4 : i32, memory_pitch_bytes = 64 : i32} : (i32, !vc4tile.shared_tile, i32) -> ()
  %view = "vc4tile.transpose_view"(%shared) {permutation = [1, 0]} : (!vc4tile.shared_tile) -> !vc4tile.shared_tile
  "vc4tile.tile_store"(%view, %out, %zero, %mask) {shape = [1, 16], dst_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (!vc4tile.shared_tile, i32, i32, !vc4tile.predicate) -> ()
  vc4tile.return
}
