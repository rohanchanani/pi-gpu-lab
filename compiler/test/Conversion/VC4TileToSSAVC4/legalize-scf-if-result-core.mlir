// RUN: vc4-opt --legalize-vc4tile-core-cfg %s | FileCheck %s

// CHECK-NOT: scf.
// CHECK-NOT: index
// CHECK: cf.cond_br
// CHECK: ^bb{{[0-9]+}}(%{{.*}}: i32)
// CHECK-NOT: scf.
// CHECK-NOT: index
vc4tile.kernel @legalize_scf_if_result_core attributes {public_name = "legalize_scf_if_result_core"} {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %go = arith.cmpi ult, %zero, %one : i32
  %selected = scf.if %go -> (i32) {
    %x = arith.addi %zero, %one : i32
    scf.yield %x : i32
  } else {
    scf.yield %zero : i32
  }
  %use = arith.addi %selected, %one : i32
  vc4tile.return
}
