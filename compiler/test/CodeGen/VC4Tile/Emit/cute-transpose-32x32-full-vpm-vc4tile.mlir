// RUN: vc4-opt %s --canonicalize-vc4tile-surface --plan-vc4tile-copies --legalize-vc4tile-core-cfg --verify-vc4tile-core --convert-vc4tile-to-ssavc4 -o - | FileCheck %s --check-prefix=SSAVC4
// RUN: vc4-opt %s --canonicalize-vc4tile-surface --plan-vc4tile-copies --legalize-vc4tile-core-cfg --verify-vc4tile-core --convert-vc4tile-to-ssavc4 | vc4-opt --convert-ssavc4-to-vc4 -o - | FileCheck %s --check-prefix=VC4

// SSAVC4-LABEL: ssavc4.func @cute_transpose_32x32_full_vpm_vc4tile_emit
// SSAVC4: ssavc4.vdr.load
// SSAVC4-SAME: memory_pitch_bytes = 128 : i32
// SSAVC4: ssavc4.barrier
// SSAVC4: ssavc4.vdw.store_vpm
// SSAVC4-SAME: memory_pitch_bytes = 128 : i32
// SSAVC4-SAME: nrows = 16 : i32
// SSAVC4-SAME: orientation = "vertical"
// SSAVC4-NOT: ssavc4.vpm.read
// SSAVC4-NOT: ssavc4.vdw.store_vpm
// VC4-LABEL: vc4.func @cute_transpose_32x32_full_vpm_vc4tile_emit
// VC4: vc4.qpu.vpmvcd_setup {{.*}}side = #vc4.vpmvcd_side<read>
// VC4: vc4.qpu.sema
// VC4: vc4.qpu.vpmvcd_setup {{.*}}side = #vc4.vpmvcd_side<write>
vc4tile.kernel @cute_transpose_32x32_full_vpm_vc4tile_emit(%out : i32, %in : i32) attributes {
  public_name = "cute_transpose_32x32_full_vpm_vc4tile_emit",
  schedule_mode = #vc4tile.schedule_mode<cooperative_block>,
  arg_attrs = [
    {abi_name = "out", kind = "buffer", direction = "out", elem_type = "u32", type = "u32"},
    {abi_name = "in", kind = "buffer", direction = "in", elem_type = "u32", type = "u32"}
  ],
  warps_per_block_max = 1 : i32,
  uses_shared_vpm = true,
  uses_barrier = true,
  require_full_block_residency = true,
  semaphores_per_block = 4 : i32,
  vpm_rows_per_block = 64 : i32,
  vpm_bytes_per_block = 4096 : i32
} {
  %mask = vc4tile.mask_all
  %in_br = arith.constant 528 : i32
  %out_br = arith.constant 528 : i32
  %shared = "vc4tile.shared_tile_alloc"() {rows = 64 : i32, elem_bytes = 4 : i32, shape = [32, 32], element_type = i32, storage_type = i32, layout = #vc4tile.layout<vpm_row>, role = #vc4tile.role<scratch>, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, memory_space = #vc4tile.memory_space<shared_vpm>} : () -> !vc4tile.shared_tile
  "vc4tile.copy_tile"(%in, %shared, %in_br, %mask) {shape = [16, 16], src_space = #vc4tile.memory_space<global>, dst_space = #vc4tile.memory_space<shared_vpm>, src_layout = #vc4tile.layout<row_major>, dst_layout = #vc4tile.layout<vpm_row>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, elem_bytes = 4 : i32, memory_pitch_bytes = 128 : i32, vpm_base_row = 48 : i32} : (i32, !vc4tile.shared_tile, i32, !vc4tile.predicate) -> ()
  vc4tile.barrier {scope = #vc4tile.barrier_scope<block>}
  %view = "vc4tile.transpose_view"(%shared) {permutation = [1, 0]} : (!vc4tile.shared_tile) -> !vc4tile.shared_tile
  "vc4tile.tile_store"(%view, %out, %out_br, %mask) {shape = [16, 16], dst_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, shared_row = 48 : i32, memory_pitch_bytes = 128 : i32} : (!vc4tile.shared_tile, i32, i32, !vc4tile.predicate) -> ()
  vc4tile.return
}
