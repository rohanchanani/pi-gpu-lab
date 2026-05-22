// RUN: vc4-opt --convert-vc4tile-to-ssavc4 %s | FileCheck %s

// CHECK-LABEL: ssavc4.func @uniform_control_flow
// CHECK: ssavc4.make_flags
// CHECK: ssavc4.cond_br
// CHECK: ssavc4.br
// CHECK: ssavc4.thread_end
// CHECK-NOT: vc4tile.
vc4tile.kernel @uniform_control_flow attributes {public_name = "uniform_control_flow"} {
  %pid = vc4tile.program_id : i32
  %one = arith.constant 1 : i32
  %pred = arith.cmpi ult, %pid, %one : i32
  cf.cond_br %pred, ^then, ^done
^then:
  cf.br ^done
^done:
  vc4tile.return
}
