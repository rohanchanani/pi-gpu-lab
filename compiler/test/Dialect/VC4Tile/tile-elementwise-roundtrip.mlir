// RUN: vc4-opt %s -o - | FileCheck %s

// CHECK-LABEL: vc4tile.kernel @tile_elementwise_roundtrip
// CHECK: "vc4tile.tile_fill"
// CHECK: "vc4tile.tile_broadcast"
// CHECK: "vc4tile.tile_add"
// CHECK: "vc4tile.tile_sub"
// CHECK: "vc4tile.tile_mul"
// CHECK: "vc4tile.tile_select"
vc4tile.kernel @tile_elementwise_roundtrip(%out : i32, %scale : i32) attributes {
  public_name = "tile_elementwise_roundtrip",
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  arg_attrs = [
    {abi_name = "out", kind = "buffer", direction = "out", elem_type = "u32", type = "u32"},
    {abi_name = "scale", kind = "scalar", direction = "by_value", type = "u32"}
  ],
  warps_per_block_max = 1 : i32,
  uses_shared_vpm = false,
  uses_barrier = false,
  semaphores_per_block = 0 : i32,
  vpm_rows_per_block = 0 : i32,
  vpm_bytes_per_block = 0 : i32
} {
  %mask = vc4tile.mask_all : vector<16xi1>
  %fill = "vc4tile.tile_fill"() {value = 7 : i32, shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<register>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, role = #vc4tile.role<constant>, copy_stage = #vc4tile.copy_stage<steady_state>, reuse_hint = #vc4tile.reuse_hint<register>, packing = #vc4tile.packing<none>} : () -> vector<16xi32>
  %bcast = "vc4tile.tile_broadcast"(%scale) {shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<register>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (i32) -> vector<16xi32>
  %sum = "vc4tile.tile_add"(%fill, %bcast) {shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<register>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
  %diff = "vc4tile.tile_sub"(%sum, %fill) {shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<register>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
  %prod = "vc4tile.tile_mul"(%diff, %fill) {shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<register>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
  %sel = "vc4tile.tile_select"(%mask, %prod, %fill) {shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<register>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (vector<16xi1>, vector<16xi32>, vector<16xi32>) -> vector<16xi32>
  "vc4tile.tile_store"(%sel, %out, %scale, %mask) {shape = [1, 16], dst_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, boundary = #vc4tile.boundary_policy<exact>, packing = #vc4tile.packing<none>} : (vector<16xi32>, i32, i32, vector<16xi1>) -> ()
  vc4tile.return
}
