// RUN: not vc4-opt %s -o - 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: sed '/requires_core_legalize/d' %s | vc4-opt - --canonicalize-vc4tile-surface --plan-vc4tile-copies --legalize-vc4tile-core-cfg --verify-vc4tile-core -o - | FileCheck %s --check-prefix=CORE
// RUN: sed '/requires_core_legalize/d' %s | vc4-opt - --canonicalize-vc4tile-surface --plan-vc4tile-copies --legalize-vc4tile-core-cfg --verify-vc4tile-core --convert-vc4tile-to-ssavc4 -o - | FileCheck %s --check-prefix=SSAVC4

// INVALID: error
// INVALID: raw scf
// CORE-LABEL: vc4tile.kernel @scf_tiled_copy_nondiv_trip
// CORE-NOT: scf.
// CORE-NOT: : index
// CORE-NOT: vc4tile.tile_load
// CORE: cf.cond_br
// CORE: vc4tile.core_tail_mask
// CORE: vc4tile.masked_load_global
// CORE: vc4tile.masked_store_global
// CORE-NOT: scf.
// CORE-NOT: : index
// CORE-NOT: vc4tile.tile_store
// SSAVC4-LABEL: ssavc4.func @scf_tiled_copy_nondiv_trip
// SSAVC4-NOT: scf.
// SSAVC4-NOT: : index
// SSAVC4: ssavc4.cond_br
// SSAVC4: ssavc4.tmu.request
// SSAVC4: ssavc4.vdw.store
vc4tile.kernel @scf_tiled_copy_nondiv_trip(%out : i32, %in : i32, %n : i32) attributes {
  public_name = "scf_tiled_copy_nondiv_trip",
  requires_core_legalize = true,
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  arg_attrs = [
    {abi_name = "out", kind = "buffer", direction = "out", elem_type = "u32", type = "u32"},
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
  %lb = arith.constant 0 : index
  %n_idx = arith.index_cast %n : i32 to index
  %step = arith.constant 16 : index
  scf.for %i = %lb to %n_idx step %step {
    %i32 = arith.index_cast %i : index to i32
    %mask = vc4tile.tail_mask %i32, %n : i32, i32
    %tile = "vc4tile.tile_load"(%in, %i32, %mask) {shape = [1, 16], src_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, boundary = #vc4tile.boundary_policy<tail_predicated>, packing = #vc4tile.packing<none>} : (i32, i32, !vc4tile.predicate) -> vector<16xi32>
    "vc4tile.tile_store"(%tile, %out, %i32, %mask) {shape = [1, 16], dst_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, boundary = #vc4tile.boundary_policy<tail_predicated>, packing = #vc4tile.packing<none>} : (vector<16xi32>, i32, i32, !vc4tile.predicate) -> ()
  }
  vc4tile.return
}
