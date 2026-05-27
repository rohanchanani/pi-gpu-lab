// RUN: vc4-opt --legalize-vc4tile-core-cfg %s | FileCheck %s

// CHECK-NOT: scf.
// CHECK-NOT: index
// CHECK: cf.cond_br
vc4tile.kernel @legalize_scf_nested_core attributes {public_name = "legalize_scf_nested_core"} {
  %lb = arith.constant 0 : index
  %ub = arith.constant 2 : index
  %step = arith.constant 1 : index
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  scf.for %i = %lb to %ub step %step {
    %ii = arith.index_cast %i : index to i32
    %go = arith.cmpi ult, %ii, %one : i32
    scf.if %go {
      %x = arith.addi %zero, %one : i32
    }
  }
  vc4tile.return
}
