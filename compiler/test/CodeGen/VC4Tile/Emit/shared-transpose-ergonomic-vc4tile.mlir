// RUN: vc4-opt %s --canonicalize-vc4tile-surface --plan-vc4tile-copies --legalize-vc4tile-core-cfg --verify-vc4tile-core --convert-vc4tile-to-ssavc4 -o - | FileCheck %s --check-prefix=SSAVC4
// RUN: vc4-opt %s --canonicalize-vc4tile-surface --plan-vc4tile-copies --legalize-vc4tile-core-cfg --verify-vc4tile-core --convert-vc4tile-to-ssavc4 | vc4-opt --convert-ssavc4-to-vc4 -o - | FileCheck %s --check-prefix=VC4

// SSAVC4-LABEL: ssavc4.func @shared_transpose_ergonomic_vc4tile
// SSAVC4-NOT: vc4tile.copy_tile
// SSAVC4-NOT: vc4tile.transpose_view
// SSAVC4: ssavc4.vpm.write
// SSAVC4-SAME: orientation = "horizontal"
// SSAVC4: ssavc4.barrier
// SSAVC4: ssavc4.vdw.store_vpm
// SSAVC4-SAME: orientation = "vertical"
// SSAVC4-NOT: ssavc4.vpm.read
// SSAVC4-NOT: ssavc4.vdw.store
// VC4-LABEL: vc4.func @shared_transpose_ergonomic_vc4tile
// VC4: vc4.qpu.vpmvcd_setup {{.*}}side = #vc4.vpmvcd_side<write>
// VC4: vc4.qpu.sema
// VC4: vc4.qpu.vpmvcd_setup {{.*}}side = #vc4.vpmvcd_side<write>
vc4tile.kernel @shared_transpose_ergonomic_vc4tile(%out : i32, %in : i32) attributes {
  public_name = "shared_transpose_ergonomic_vc4tile",
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
  vpm_rows_per_block = 16 : i32,
  vpm_bytes_per_block = 1024 : i32
} {
  %zero = arith.constant 0 : i32
  %sixteen = arith.constant 16 : i32
  %mask = vc4tile.mask_all : vector<16xi1>
  %shared = "vc4tile.shared_tile_alloc"() {rows = 16 : i32, elem_bytes = 4 : i32, shape = [16, 16], element_type = i32, storage_type = i32, layout = #vc4tile.layout<vpm_row>, role = #vc4tile.role<scratch>, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, memory_space = #vc4tile.memory_space<shared_vpm>} : () -> !vc4tile.shared_tile
  %r0 = "vc4tile.tile_load"(%in, %zero, %mask) {shape = [1, 16], src_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
  "vc4tile.copy_tile"(%r0, %shared, %zero, %mask) {shape = [1, 16], src_space = #vc4tile.memory_space<register>, dst_space = #vc4tile.memory_space<shared_vpm>, src_layout = #vc4tile.layout<row_major>, dst_layout = #vc4tile.layout<vpm_row>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, elem_bytes = 4 : i32} : (vector<16xi32>, !vc4tile.shared_tile, i32, vector<16xi1>) -> ()
  vc4tile.barrier {scope = #vc4tile.barrier_scope<block>}
  %view = "vc4tile.transpose_view"(%shared) {permutation = [1, 0]} : (!vc4tile.shared_tile) -> !vc4tile.shared_tile
  "vc4tile.tile_store"(%view, %out, %zero, %mask) {shape = [1, 16], dst_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (!vc4tile.shared_tile, i32, i32, vector<16xi1>) -> ()
  vc4tile.return
}
