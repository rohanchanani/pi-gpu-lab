// RUN: vc4-opt --convert-vc4tile-to-ssavc4 %s | FileCheck %s

// CHECK-LABEL: ssavc4.func @ids_arith_masks
// CHECK: ssavc4.uniform.read
// CHECK: ssavc4.element_number
// CHECK: ssavc4.load_imm
// CHECK: ssavc4.splat
// CHECK: ssavc4.alu.add
// CHECK: ssavc4.make_flags
// CHECK: ssavc4.thread_end
// CHECK-NOT: vc4tile.
vc4tile.kernel @ids_arith_masks attributes {public_name = "ids_arith_masks"} {
  %pid = vc4tile.program_id : i32
  %lanes = vc4tile.lane_range : vector<16xi32>
  %bias = arith.constant 4 : i32
  %bias_vec = vector.broadcast %bias : i32 to vector<16xi32>
  %sum = arith.addi %lanes, %bias_vec : vector<16xi32>
  %cmp = arith.cmpi ult, %lanes, %sum : vector<16xi32>
  vc4tile.return
}
