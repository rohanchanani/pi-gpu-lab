// RUN: not vc4-opt %s --convert-vc4tile-to-ssavc4 -o /dev/null 2>&1 | FileCheck %s

// CHECK: error
// CHECK: is a surface operation
// CHECK: --plan-vc4tile-copies
// CHECK: --convert-vc4tile-to-ssavc4
vc4tile.kernel @predicate_copy_before_ssavc4(%out : i32) attributes {
  public_name = "predicate_copy_before_ssavc4",
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  arg_attrs = [{abi_name = "out", kind = "buffer", direction = "out", elem_type = "u32", type = "u32"}],
  warps_per_block_max = 1 : i32,
  uses_shared_vpm = false,
  uses_barrier = false,
  semaphores_per_block = 0 : i32,
  vpm_rows_per_block = 0 : i32,
  vpm_bytes_per_block = 0 : i32
} {
  %zero = arith.constant 0 : i32
  %tile = arith.constant dense<0> : vector<16xi32>
  %mask = vc4tile.tile_rect_mask {active_rows = 3 : i32, active_cols = 2 : i32, shape = [4, 4], layout = #vc4tile.layout<row_major>}
  "vc4tile.copy_tile"(%tile, %out, %zero, %mask) {
    shape = [4, 4],
    src_space = #vc4tile.memory_space<register>,
    dst_space = #vc4tile.memory_space<global>,
    src_layout = #vc4tile.layout<row_major>,
    dst_layout = #vc4tile.layout<row_major>,
    element_type = i32, storage_type = i32,
    precision = #vc4tile.precision<exact_32>,
    packing = #vc4tile.packing<none>,
    elem_bytes = 4 : i32,
    memory_pitch_bytes = 16 : i32
  } : (vector<16xi32>, i32, i32, !vc4tile.predicate) -> ()
  vc4tile.return
}
