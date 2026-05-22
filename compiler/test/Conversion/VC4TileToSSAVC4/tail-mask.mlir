// RUN: vc4-opt --convert-vc4tile-to-ssavc4 %s | FileCheck %s

// CHECK-LABEL: ssavc4.func @tail_mask
// CHECK: ssavc4.element_number
// CHECK: ssavc4.make_flags
// CHECK: ssavc4.thread_end
// CHECK-NOT: vc4tile.
vc4tile.kernel @tail_mask attributes {public_name = "tail_mask"} {
  %base = arith.constant 0 : i32
  %limit = arith.constant 13 : i32
  %mask = vc4tile.tail_mask %base, %limit : i32, i32 -> vector<16xi1>
  vc4tile.return
}
