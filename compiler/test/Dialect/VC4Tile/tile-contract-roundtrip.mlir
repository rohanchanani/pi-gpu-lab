// RUN: vc4-opt %s -o - | FileCheck %s

// CHECK-LABEL: vc4tile.kernel @tile_contract_roundtrip
// CHECK: "vc4tile.tile_dot"
// CHECK: "vc4tile.tile_contract"
// CHECK: "vc4tile.tile_matmul"
vc4tile.kernel @tile_contract_roundtrip(%out : i32, %lhs_base : i32, %rhs_base : i32) attributes {
  public_name = "tile_contract_roundtrip",
  arg_attrs = [
    {abi_name = "out", kind = "buffer", direction = "out", elem_type = "u32", type = "u32"},
    {abi_name = "lhs", kind = "buffer", direction = "in", elem_type = "u32", type = "u32"},
    {abi_name = "rhs", kind = "buffer", direction = "in", elem_type = "u32", type = "u32"}
  ],
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  warps_per_block_max = 1 : i32,
  uses_shared_vpm = false,
  uses_barrier = false,
  semaphores_per_block = 0 : i32,
  vpm_rows_per_block = 0 : i32,
  vpm_bytes_per_block = 0 : i32
} {
  %mask = vc4tile.mask_all : vector<16xi1>
  %lhs = vc4tile.lane_range : vector<16xi32>
  %rhs = vc4tile.lane_range : vector<16xi32>
  %acc0 = "vc4tile.tile_fill"() {shape = [4, 4], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<register>, element_type = i32, storage_type = i32, accumulator_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, value = 13 : i32} : () -> vector<16xi32>
  %dot = "vc4tile.tile_dot"(%lhs, %rhs, %mask) {k = 16 : i32, shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<register>, element_type = i32, storage_type = i32, accumulator_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, lhs_role = #vc4tile.role<input>, rhs_role = #vc4tile.role<input>, acc_role = #vc4tile.role<accumulator>, algorithm_hint = "dot_1x16"} : (vector<16xi32>, vector<16xi32>, vector<16xi1>) -> vector<16xi32>
  %contract = "vc4tile.tile_contract"(%lhs, %rhs, %acc0, %mask) {m = 4 : i32, n = 4 : i32, k = 4 : i32, shape = [4, 4], lhs_layout = #vc4tile.layout<row_major>, rhs_layout = #vc4tile.layout<row_major>, acc_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<register>, element_type = i32, storage_type = i32, accumulator_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, contracting_dims = [[1], [0]], iterator_types = ["parallel", "parallel", "reduction"], lhs_role = #vc4tile.role<input>, rhs_role = #vc4tile.role<input>, acc_role = #vc4tile.role<accumulator>, algorithm_hint = "matmul_4x4x4"} : (vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi1>) -> vector<16xi32>
  %matmul = "vc4tile.tile_matmul"(%lhs, %rhs, %acc0, %mask) {m = 4 : i32, n = 4 : i32, k = 4 : i32, shape = [4, 4], lhs_layout = #vc4tile.layout<row_major>, rhs_layout = #vc4tile.layout<row_major>, acc_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<register>, element_type = i32, storage_type = i32, accumulator_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, contracting_dims = [[1], [0]], iterator_types = ["parallel", "parallel", "reduction"], lhs_role = #vc4tile.role<input>, rhs_role = #vc4tile.role<input>, acc_role = #vc4tile.role<accumulator>, algorithm_hint = "matmul_4x4x4"} : (vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi1>) -> vector<16xi32>
  vc4tile.return
}
