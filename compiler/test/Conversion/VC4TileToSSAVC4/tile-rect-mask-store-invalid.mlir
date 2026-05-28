// RUN: not vc4-opt %s --canonicalize-vc4tile-surface --plan-vc4tile-copies -o /dev/null 2>&1 | FileCheck %s

// CHECK: semantic predicate shape mismatch: predicate shape [4, 4] does not match consumer shape [1, 16]
vc4tile.kernel @tile_rect_mask_store_invalid_shape(%out : i32) attributes {
  public_name = "tile_rect_mask_store_invalid_shape",
  arg_attrs = [
    {abi_name = "out", kind = "buffer", direction = "out", elem_type = "u32", type = "u32"}
  ],
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  warps_per_block_max = 1 : i32,
  uses_shared_vpm = false,
  uses_barrier = false,
  semaphores_per_block = 0 : i32,
  vpm_rows_per_block = 0 : i32,
  vpm_bytes_per_block = 0 : i32
} {
  %zero = arith.constant 0 : i32
  %values = vc4tile.lane_range : vector<16xi32>
  %mask = vc4tile.tile_rect_mask {active_rows = 3 : i32, active_cols = 2 : i32, shape = [4, 4], layout = #vc4tile.layout<row_major>} : vector<16xi1>
  "vc4tile.tile_store"(%values, %out, %zero, %mask) {
    shape = [1, 16], dst_layout = #vc4tile.layout<row_major>,
    memory_space = #vc4tile.memory_space<global>,
    element_type = i32, storage_type = i32,
    precision = #vc4tile.precision<exact_32>,
    boundary = #vc4tile.boundary_policy<exact>,
    packing = #vc4tile.packing<none>
  } : (vector<16xi32>, i32, i32, vector<16xi1>) -> ()
  vc4tile.return
}
