// RUN: vc4-opt %s | FileCheck %s

// CHECK-LABEL: vc4tile.kernel @rotate_reduce
vc4tile.kernel @rotate_reduce attributes {
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  warps_per_block_max = 1 : i32,
  semaphores_per_block = 0 : i32
} {
  %values = vc4tile.lane_range : vector<16xi32>
  %mask = vc4tile.core_mask_all : vector<16xi1>
  // CHECK: vc4tile.rotate
  %rot = vc4tile.rotate %values {amount = 3 : i32} : vector<16xi32> -> vector<16xi32>
  // CHECK: vc4tile.reduce
  %sum = vc4tile.reduce %rot, %mask {kind = #vc4tile.reduce_kind<add>} : vector<16xi32>, vector<16xi1> -> vector<16xi32>
  // CHECK: #vc4tile.reduce_kind<bit_xor>
  %xor = vc4tile.reduce %sum, %mask {kind = #vc4tile.reduce_kind<bit_xor>} : vector<16xi32>, vector<16xi1> -> vector<16xi32>
  vc4tile.return
}
