// RUN: vc4-opt --legalize-vc4tile-core-cfg %s | FileCheck %s

// CHECK-NOT: scf.for
// CHECK-NOT: index
// CHECK: cf.cond_br
// CHECK: cf.br
vc4tile.kernel @legalize_scf_for_static_core attributes {public_name = "legalize_scf_for_static_core"} {
  %lb = arith.constant 0 : index
  %ub = arith.constant 4 : index
  %step = arith.constant 1 : index
  scf.for %i = %lb to %ub step %step {
    %ii = arith.index_cast %i : index to i32
    %one = arith.constant 1 : i32
    %x = arith.addi %ii, %one : i32
  }
  vc4tile.return
}
