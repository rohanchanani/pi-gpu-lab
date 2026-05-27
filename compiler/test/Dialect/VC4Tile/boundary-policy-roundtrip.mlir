// RUN: vc4-opt %s | FileCheck %s

// CHECK-LABEL: vc4tile.kernel @boundary_policy_metadata
vc4tile.kernel @boundary_policy_metadata(%out : i32, %in : i32, %n : i32) attributes {
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  arg_attrs = [
    {abi_name = "out", kind = "buffer", direction = "out", elem_type = "u32", type = "u32"},
    {abi_name = "in", kind = "buffer", direction = "in", elem_type = "u32", type = "u32"},
    {abi_name = "n", kind = "scalar", direction = "by_value", type = "u32"}
  ],
  warps_per_block_max = 1 : i32,
  uses_shared_vpm = false,
  uses_barrier = false,
  semaphores_per_block = 0 : i32
} {
  %zero = arith.constant 0 : i32
  %all = vc4tile.mask_all : vector<16xi1>
  %tail = vc4tile.tail_mask %zero, %n : i32, i32 -> vector<16xi1>
  // CHECK-DAG: boundary = #vc4tile.boundary_policy<exact>
  %exact = "vc4tile.tile_load"(%in, %zero, %all) {
    shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>,
    element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>,
    role = #vc4tile.role<input>, boundary = #vc4tile.boundary_policy<exact>
  } : (i32, i32, vector<16xi1>) -> vector<16xi32>
  // CHECK-DAG: boundary = #vc4tile.boundary_policy<tail_predicated>
  %tail_tile = "vc4tile.tile_load"(%in, %zero, %tail) {
    shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>,
    element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>,
    role = #vc4tile.role<input>, boundary = #vc4tile.boundary_policy<tail_predicated>
  } : (i32, i32, vector<16xi1>) -> vector<16xi32>
  "vc4tile.tile_store"(%tail_tile, %out, %zero, %tail) {
    shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>,
    element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>,
    role = #vc4tile.role<output>, boundary = #vc4tile.boundary_policy<tail_predicated>
  } : (vector<16xi32>, i32, i32, vector<16xi1>) -> ()
  vc4tile.return
}
