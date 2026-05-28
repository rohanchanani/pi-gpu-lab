// RUN: not vc4-opt %s -o - 2>&1 | FileCheck %s --check-prefix=INVALID

// INVALID: error
// INVALID: tile compute operations require memory_space = #vc4tile.memory_space<register>
vc4tile.kernel @tile_elementwise_invalid(%out : i32) attributes {
  public_name = "tile_elementwise_invalid",
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  arg_attrs = [{abi_name = "out", kind = "buffer", direction = "out", elem_type = "u32", type = "u32"}],
  warps_per_block_max = 1 : i32,
  uses_shared_vpm = false,
  uses_barrier = false,
  semaphores_per_block = 0 : i32,
  vpm_rows_per_block = 0 : i32,
  vpm_bytes_per_block = 0 : i32
} {
  %a = "vc4tile.tile_fill"() {value = 1 : i32, shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<register>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : () -> vector<16xi32>
  %b = "vc4tile.tile_fill"() {value = 2 : i32, shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<register>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : () -> vector<16xi32>
  %mask = vc4tile.mask_all : vector<16xi1>
  %bad = "vc4tile.tile_add"(%a, %b) {shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
  "vc4tile.tile_store"(%bad, %out, %out, %mask) {shape = [1, 16], dst_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, boundary = #vc4tile.boundary_policy<exact>, packing = #vc4tile.packing<none>} : (vector<16xi32>, i32, i32, vector<16xi1>) -> ()
  vc4tile.return
}
