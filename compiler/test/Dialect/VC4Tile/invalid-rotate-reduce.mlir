// RUN: not vc4-opt %s 2>&1 | FileCheck %s

vc4tile.kernel @bad_rotate_reduce attributes {public_name = "bad_rotate_reduce"} {
  %values = vc4tile.lane_range : vector<16xi32>
  %mask = vc4tile.core_mask_all : vector<16xi1>
  // CHECK: amount must be in range [0, 15]
  %bad = vc4tile.rotate %values {amount = 16 : i32} : vector<16xi32> -> vector<16xi32>
  %sum = vc4tile.reduce %bad, %mask {kind = #vc4tile.reduce_kind<add>} : vector<16xi32>, vector<16xi1> -> vector<16xi32>
  vc4tile.return
}
