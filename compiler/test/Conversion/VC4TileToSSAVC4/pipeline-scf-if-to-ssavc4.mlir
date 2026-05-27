// RUN: vc4-opt --legalize-vc4tile-core-cfg --verify-vc4tile-core --convert-vc4tile-to-ssavc4 %s | FileCheck %s

// CHECK: ssavc4.func
// CHECK: ssavc4.cond_br
// CHECK-NOT: scf.
// CHECK-NOT: vc4tile.
vc4tile.kernel @pipeline_scf_if_to_ssavc4 attributes {public_name = "pipeline_scf_if_to_ssavc4"} {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %go = arith.cmpi ult, %zero, %one : i32
  %selected = scf.if %go -> (i32) {
    scf.yield %one : i32
  } else {
    scf.yield %zero : i32
  }
  %use = arith.addi %selected, %one : i32
  vc4tile.return
}
