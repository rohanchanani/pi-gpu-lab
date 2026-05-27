// RUN: not vc4-opt --legalize-vc4tile-core-cfg %s 2>&1 | FileCheck %s

// CHECK: index
vc4tile.kernel @legalize_scf_index_leak_invalid attributes {public_name = "legalize_scf_index_leak_invalid"} {
  %lb = arith.constant 0 : index
  %ub = arith.constant 2 : index
  %step = arith.constant 1 : index
  scf.for %i = %lb to %ub step %step {
    %bad = arith.addi %i, %i : index
  }
  vc4tile.return
}
