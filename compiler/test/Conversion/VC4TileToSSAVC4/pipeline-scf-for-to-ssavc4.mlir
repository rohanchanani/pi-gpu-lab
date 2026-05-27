// RUN: vc4-opt --legalize-vc4tile-core-cfg --verify-vc4tile-core --convert-vc4tile-to-ssavc4 %s | FileCheck %s

// CHECK: ssavc4.func
// CHECK: ssavc4.cond_br
// CHECK-NOT: scf.
// CHECK-NOT: vc4tile.
vc4tile.kernel @pipeline_scf_for_to_ssavc4 attributes {public_name = "pipeline_scf_for_to_ssavc4"} {
  %lb = arith.constant 0 : index
  %ub = arith.constant 2 : index
  %step = arith.constant 1 : index
  %zero = arith.constant 0 : i32
  %sum = scf.for %i = %lb to %ub step %step iter_args(%acc = %zero) -> (i32) {
    %ii = arith.index_cast %i : index to i32
    %next = arith.addi %acc, %ii : i32
    scf.yield %next : i32
  }
  vc4tile.return
}
