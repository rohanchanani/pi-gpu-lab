// RUN: vc4-opt --convert-vc4tile-to-ssavc4 %s | FileCheck %s

// CHECK-LABEL: ssavc4.func @rotate_reduce
// CHECK: ssavc4.element_number
// CHECK: ssavc4.rotate
// CHECK-SAME: amount = 3 : i32
// CHECK: ssavc4.rotate
// CHECK-SAME: amount = 8 : i32
// CHECK: ssavc4.alu.add
// CHECK: ssavc4.rotate
// CHECK-SAME: amount = 4 : i32
// CHECK: ssavc4.alu.add
// CHECK: ssavc4.rotate
// CHECK-SAME: amount = 2 : i32
// CHECK: ssavc4.alu.add
// CHECK: ssavc4.rotate
// CHECK-SAME: amount = 1 : i32
// CHECK: ssavc4.alu.add
// CHECK: ssavc4.thread_end
// CHECK-NOT: vc4tile.
vc4tile.kernel @rotate_reduce attributes {public_name = "rotate_reduce"} {
  %values = vc4tile.lane_range : vector<16xi32>
  %mask = vc4tile.core_mask_all : vector<16xi1>
  %rot = vc4tile.rotate %values {amount = 3 : i32} : vector<16xi32> -> vector<16xi32>
  %sum = vc4tile.reduce %rot, %mask {kind = #vc4tile.reduce_kind<add>} : vector<16xi32>, vector<16xi1> -> vector<16xi32>
  vc4tile.return
}
