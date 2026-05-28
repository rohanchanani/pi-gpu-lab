// RUN: not vc4-opt %s -o - 2>&1 | FileCheck %s

// CHECK: error
// CHECK: tile_contract/tile_matmul active_k must be in range [1, 4]
vc4tile.kernel @tile_contract_active_k_invalid attributes {
  public_name = "tile_contract_active_k_invalid",
  arg_attrs = [],
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  warps_per_block_max = 1 : i32,
  uses_shared_vpm = false,
  uses_barrier = false,
  semaphores_per_block = 0 : i32,
  vpm_rows_per_block = 0 : i32,
  vpm_bytes_per_block = 0 : i32
} {
  %mask = vc4tile.mask_all
  %lhs = vc4tile.lane_range : vector<16xi32>
  %rhs = vc4tile.lane_range : vector<16xi32>
  %acc = "vc4tile.tile_fill"() {shape = [4, 4], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<register>, element_type = i32, storage_type = i32, accumulator_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, value = 0 : i32} : () -> vector<16xi32>
  %bad = "vc4tile.tile_matmul"(%lhs, %rhs, %acc, %mask) {m = 4 : i32, n = 4 : i32, k = 4 : i32, active_k = 0 : i32, shape = [4, 4], lhs_layout = #vc4tile.layout<row_major>, rhs_layout = #vc4tile.layout<row_major>, acc_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<register>, element_type = i32, storage_type = i32, accumulator_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, contracting_dims = [[1], [0]], iterator_types = ["parallel", "parallel", "reduction"]} : (vector<16xi32>, vector<16xi32>, vector<16xi32>, !vc4tile.predicate) -> vector<16xi32>
  vc4tile.return
}
