// RUN: vc4-opt %s --canonicalize-vc4tile-surface --plan-vc4tile-copies --legalize-vc4tile-core-cfg --verify-vc4tile-core -o - | FileCheck %s

// CHECK-LABEL: vc4tile.kernel @plan_copy_global_register
// CHECK-NOT: vc4tile.tile_load
// CHECK: vc4tile.lane_range
// CHECK: vc4tile.masked_load_global
vc4tile.kernel @plan_copy_global_register(%in : i32, %n : i32) attributes {
  public_name = "plan_copy_global_register",
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  arg_attrs = [
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
    element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>
  } : (i32, i32, !vc4tile.predicate) -> vector<16xi32>
  vc4tile.return
}
