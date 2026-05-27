// RUN: vc4-opt --legalize-vc4tile-core-cfg %s | FileCheck %s

// CHECK-NOT: scf.for
// CHECK-NOT: index
// CHECK: ^bb{{[0-9]+}}(%{{.*}}: i32, %{{.*}}: i32, %{{.*}}: vector<16xi32>)
vc4tile.kernel @legalize_scf_for_iter_args_core attributes {public_name = "legalize_scf_for_iter_args_core"} {
  %lb = arith.constant 0 : index
  %ub = arith.constant 2 : index
  %step = arith.constant 1 : index
  %zero = arith.constant 0 : i32
  %lanes = vc4tile.lane_range : vector<16xi32>
  %sum:2 = scf.for %i = %lb to %ub step %step iter_args(%acc = %zero, %vec = %lanes) -> (i32, vector<16xi32>) {
    %ii = arith.index_cast %i : index to i32
    %next = arith.addi %acc, %ii : i32
    scf.yield %next, %vec : i32, vector<16xi32>
  }
  vc4tile.return
}
