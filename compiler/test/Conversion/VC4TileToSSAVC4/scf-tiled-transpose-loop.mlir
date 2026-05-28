// RUN: vc4-opt %s --canonicalize-vc4tile-surface --plan-vc4tile-copies --legalize-vc4tile-core-cfg --verify-vc4tile-core -o - | FileCheck %s --check-prefix=CORE
// RUN: vc4-opt %s --canonicalize-vc4tile-surface --plan-vc4tile-copies --legalize-vc4tile-core-cfg --verify-vc4tile-core --convert-vc4tile-to-ssavc4 -o - | FileCheck %s --check-prefix=SSAVC4

// CORE-LABEL: vc4tile.kernel @scf_tiled_transpose_loop
// CORE-NOT: scf.
// CORE-NOT: : index
// CORE-NOT: vc4tile.copy_tile
// CORE-NOT: vc4tile.transpose_view
// CORE: cf.cond_br
// CORE: vc4tile.shared_store
// CORE: vc4tile.barrier
// CORE: vc4tile.shared_store_global
// CORE-SAME: #vc4tile.vpm_layout<column_major>
// CORE-NOT: vc4tile.shared_load
// CORE-NOT: vc4tile.masked_store_global
// CORE-NOT: scf.
// CORE-NOT: : index
// CORE-NOT: vc4tile.tile_store
// SSAVC4-LABEL: ssavc4.func @scf_tiled_transpose_loop
// SSAVC4-NOT: scf.
// SSAVC4-NOT: : index
// SSAVC4: ssavc4.vpm.write
// SSAVC4: ssavc4.barrier
// SSAVC4: ssavc4.vdw.store_vpm
// SSAVC4-SAME: orientation = "vertical"
// SSAVC4-NOT: ssavc4.vpm.read
// SSAVC4-NOT: ssavc4.vdw.store
vc4tile.kernel @scf_tiled_transpose_loop(%out : i32, %in : i32) attributes {
  public_name = "scf_tiled_transpose_loop",
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
  %idx0 = arith.constant 0 : index
  %idx16 = arith.constant 16 : index
  %idx1 = arith.constant 1 : index
  %zero = arith.constant 0 : i32
  %sixteen = arith.constant 16 : i32
  %mask = vc4tile.mask_all : vector<16xi1>
  %shared = "vc4tile.shared_tile_alloc"() {rows = 16 : i32, elem_bytes = 4 : i32, shape = [16, 16], element_type = i32, storage_type = i32, layout = #vc4tile.layout<vpm_row>, role = #vc4tile.role<scratch>, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, memory_space = #vc4tile.memory_space<shared_vpm>} : () -> !vc4tile.shared_tile
  scf.for %row = %idx0 to %idx16 step %idx1 {
    %row_i32 = arith.index_cast %row : index to i32
    %offset = arith.muli %row_i32, %sixteen : i32
    %tile = "vc4tile.tile_load"(%in, %offset, %mask) {shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
    "vc4tile.copy_tile"(%tile, %shared, %row_i32, %mask) {shape = [1, 16], src_space = #vc4tile.memory_space<register>, dst_space = #vc4tile.memory_space<shared_vpm>, src_layout = #vc4tile.layout<row_major>, dst_layout = #vc4tile.layout<vpm_row>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, elem_bytes = 4 : i32} : (vector<16xi32>, !vc4tile.shared_tile, i32, vector<16xi1>) -> ()
  }
  vc4tile.barrier {scope = #vc4tile.barrier_scope<block>}
  %view = "vc4tile.transpose_view"(%shared) {permutation = [1, 0]} : (!vc4tile.shared_tile) -> !vc4tile.shared_tile
  "vc4tile.tile_store"(%view, %out, %zero, %mask) {shape = [1, 16], layout = #vc4tile.layout<transposed_view>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, shared_row = 0 : i32} : (!vc4tile.shared_tile, i32, i32, vector<16xi1>) -> ()
  vc4tile.return
}
