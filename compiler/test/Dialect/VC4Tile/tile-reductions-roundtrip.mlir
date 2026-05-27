// RUN: vc4-opt %s -o - | FileCheck %s

// CHECK-LABEL: vc4tile.kernel @tile_reductions_roundtrip
// CHECK: "vc4tile.tile_reduce"
// CHECK: "vc4tile.row_reduce"
// CHECK: "vc4tile.warp_reduce"
vc4tile.kernel @tile_reductions_roundtrip(%out : i32, %in : i32) attributes {
  public_name = "tile_reductions_roundtrip",
  arg_attrs = [
    {abi_name = "out", kind = "buffer", direction = "out", elem_type = "u32", type = "u32"},
    {abi_name = "in", kind = "buffer", direction = "in", elem_type = "u32", type = "u32"}
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
  %mask = vc4tile.mask_all : vector<16xi1>
  %tile = "vc4tile.tile_load"(%in, %zero, %mask) {shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, boundary = #vc4tile.boundary_policy<exact>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
  %tile_sum = "vc4tile.tile_reduce"(%tile, %mask) {shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<register>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, kind = #vc4tile.reduce_kind<add>, axis = 1 : i32, role = #vc4tile.role<accumulator>, algorithm_hint = "rotate_add_tree"} : (vector<16xi32>, vector<16xi1>) -> vector<16xi32>
  %row_sum = "vc4tile.row_reduce"(%tile_sum, %mask) {shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<register>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, kind = #vc4tile.reduce_kind<add>, axis = 1 : i32, role = #vc4tile.role<accumulator>, algorithm_hint = "rotate_add_tree"} : (vector<16xi32>, vector<16xi1>) -> vector<16xi32>
  %warp_sum = "vc4tile.warp_reduce"(%row_sum, %mask) {shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<register>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, kind = #vc4tile.reduce_kind<add>, axis = 1 : i32, algorithm_hint = "rotate_add_tree"} : (vector<16xi32>, vector<16xi1>) -> vector<16xi32>
  "vc4tile.tile_store"(%warp_sum, %out, %zero, %mask) {shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, boundary = #vc4tile.boundary_policy<exact>, packing = #vc4tile.packing<none>} : (vector<16xi32>, i32, i32, vector<16xi1>) -> ()
  vc4tile.return
}

// CHECK-LABEL: vc4tile.kernel @block_reductions_roundtrip
// CHECK: "vc4tile.block_reduce"
vc4tile.kernel @block_reductions_roundtrip(%out : i32, %in : i32) attributes {
  public_name = "block_reductions_roundtrip",
  arg_attrs = [
    {abi_name = "out", kind = "buffer", direction = "out", elem_type = "u32", type = "u32"},
    {abi_name = "in", kind = "buffer", direction = "in", elem_type = "u32", type = "u32"}
  ],
  schedule_mode = #vc4tile.schedule_mode<cooperative_block>,
  warps_per_block_max = 1 : i32,
  uses_shared_vpm = true,
  uses_barrier = true,
  require_full_block_residency = true,
  semaphores_per_block = 4 : i32,
  vpm_rows_per_block = 1 : i32,
  vpm_bytes_per_block = 64 : i32
} {
  %zero = arith.constant 0 : i32
  %mask = vc4tile.mask_all : vector<16xi1>
  %tile = "vc4tile.tile_load"(%in, %zero, %mask) {shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, boundary = #vc4tile.boundary_policy<exact>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
  %block_sum = "vc4tile.block_reduce"(%tile, %mask) {shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<register>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, kind = #vc4tile.reduce_kind<add>, axis = 1 : i32, algorithm_hint = "shared_vpm_barrier"} : (vector<16xi32>, vector<16xi1>) -> vector<16xi32>
  "vc4tile.tile_store"(%block_sum, %out, %zero, %mask) {shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, boundary = #vc4tile.boundary_policy<exact>, packing = #vc4tile.packing<none>} : (vector<16xi32>, i32, i32, vector<16xi1>) -> ()
  vc4tile.return
}
