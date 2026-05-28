// RUN: vc4-opt %s --canonicalize-vc4tile-surface --plan-vc4tile-copies --legalize-vc4tile-core-cfg --verify-vc4tile-core -o - | FileCheck %s --check-prefix=CORE
// RUN: vc4-opt %s --canonicalize-vc4tile-surface --plan-vc4tile-copies --legalize-vc4tile-core-cfg --verify-vc4tile-core --convert-vc4tile-to-ssavc4 -o - | FileCheck %s --check-prefix=SSAVC4

// CORE-LABEL: vc4tile.kernel @scf_tiled_copy_loop
// CORE-NOT: scf.
// CORE-NOT: : index
// CORE-NOT: vc4tile.tile_load
// CORE-NOT: vc4tile.tile_store
// CORE: cf.cond_br
// CORE: vc4tile.masked_load_global
// CORE: vc4tile.masked_store_global
// CORE-NOT: scf.
// CORE-NOT: : index
// CORE-NOT: vc4tile.tile_load
// CORE-NOT: vc4tile.tile_store
// SSAVC4-LABEL: ssavc4.func @scf_tiled_copy_loop
// SSAVC4-NOT: scf.
// SSAVC4-NOT: : index
// SSAVC4-NOT: vc4tile.tile_load
// SSAVC4: ssavc4.cond_br
// SSAVC4: ssavc4.tmu.request
// SSAVC4: ssavc4.vdw.store
vc4tile.kernel @scf_tiled_copy_loop(%out : i32, %in : i32) attributes {
  public_name = "scf_tiled_copy_loop",
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  arg_attrs = [
    {abi_name = "out", kind = "buffer", direction = "out", elem_type = "u32", type = "u32"},
    {abi_name = "in", kind = "buffer", direction = "in", elem_type = "u32", type = "u32"}
  ],
  warps_per_block_max = 1 : i32,
  uses_shared_vpm = false,
  uses_barrier = false,
  semaphores_per_block = 0 : i32,
  vpm_rows_per_block = 0 : i32,
  vpm_bytes_per_block = 0 : i32
} {
  %lb = arith.constant 0 : index
  %ub = arith.constant 64 : index
  %step = arith.constant 16 : index
  %limit = arith.constant 64 : i32
  %one = arith.constant 1 : i32
  %mask = vc4tile.mask_all : vector<16xi1>
  %acc0 = arith.constant 0 : i32
  %sum = scf.for %i = %lb to %ub step %step iter_args(%acc = %acc0) -> (i32) {
    %i32 = arith.index_cast %i : index to i32
    %active = arith.cmpi ult, %i32, %limit : i32
    scf.if %active {
      %tile = "vc4tile.tile_load"(%in, %i32, %mask) {shape = [1, 16], src_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
      "vc4tile.tile_store"(%tile, %out, %i32, %mask) {shape = [1, 16], dst_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (vector<16xi32>, i32, i32, vector<16xi1>) -> ()
    }
    %next = arith.addi %acc, %one : i32
    scf.yield %next : i32
  }
  vc4tile.return
}
