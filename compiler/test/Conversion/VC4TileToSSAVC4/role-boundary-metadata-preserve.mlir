// RUN: vc4-opt %s --canonicalize-vc4tile-surface --plan-vc4tile-copies --legalize-vc4tile-core-cfg --verify-vc4tile-core -o - | FileCheck --check-prefix=CORE %s
// RUN: vc4-opt %s --canonicalize-vc4tile-surface --plan-vc4tile-copies --legalize-vc4tile-core-cfg --verify-vc4tile-core --convert-vc4tile-to-ssavc4 -o - | FileCheck --check-prefix=SSAVC4 %s

// CORE-LABEL: vc4tile.kernel @role_boundary_metadata_preserve
// CORE: vc4tile.masked_load_global
// CORE-SAME: boundary = #vc4tile.boundary_policy<tail_predicated>
// CORE-SAME: copy_stage = #vc4tile.copy_stage<prologue>
// CORE-SAME: reuse_hint = #vc4tile.reuse_hint<register>
// CORE-SAME: role = #vc4tile.role<input>
// CORE: vc4tile.masked_store_global
// CORE-SAME: boundary = #vc4tile.boundary_policy<tail_predicated>
// CORE-SAME: copy_stage = #vc4tile.copy_stage<epilogue>
// CORE-SAME: reuse_hint = #vc4tile.reuse_hint<producer>
// CORE-SAME: role = #vc4tile.role<output>
// SSAVC4-LABEL: ssavc4.func @role_boundary_metadata_preserve
// SSAVC4: ssavc4.tmu.request
// SSAVC4-SAME: boundary = #vc4tile.boundary_policy<tail_predicated>
// SSAVC4-SAME: role = #vc4tile.role<input>
// SSAVC4: ssavc4.cond_select
// SSAVC4-SAME: boundary = #vc4tile.boundary_policy<tail_predicated>
// SSAVC4-SAME: role = #vc4tile.role<input>
// SSAVC4: ssavc4.vdw.store
// SSAVC4-SAME: boundary = #vc4tile.boundary_policy<tail_predicated>
// SSAVC4-SAME: role = #vc4tile.role<output>
vc4tile.kernel @role_boundary_metadata_preserve(%out : i32, %in : i32, %n : i32) attributes {
  public_name = "role_boundary_metadata_preserve",
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
  %zero = arith.constant 0 : i32
  %mask = vc4tile.tail_mask %zero, %n : i32, i32
  %tile = "vc4tile.tile_load"(%in, %zero, %mask) {
    shape = [1, 16], src_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>,
    element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>,
    boundary = #vc4tile.boundary_policy<tail_predicated>, role = #vc4tile.role<input>,
    copy_stage = #vc4tile.copy_stage<prologue>, reuse_hint = #vc4tile.reuse_hint<register>
  } : (i32, i32, !vc4tile.predicate) -> vector<16xi32>
  "vc4tile.tile_store"(%tile, %out, %zero, %mask) {
    shape = [1, 16], dst_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>,
    element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>,
    boundary = #vc4tile.boundary_policy<tail_predicated>, role = #vc4tile.role<output>,
    copy_stage = #vc4tile.copy_stage<epilogue>, reuse_hint = #vc4tile.reuse_hint<producer>
  } : (vector<16xi32>, i32, i32, !vc4tile.predicate) -> ()
  vc4tile.return
}
