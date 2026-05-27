// RUN: not vc4-opt --convert-vc4tile-to-ssavc4 %s 2>&1 | FileCheck %s

// CHECK: legalize-vc4tile-core-cfg
vc4tile.kernel @reject_raw_scf_before_core_legalize attributes {public_name = "reject_raw_scf_before_core_legalize"} {
  %lb = arith.constant 0 : index
  %ub = arith.constant 2 : index
  %step = arith.constant 1 : index
  scf.for %i = %lb to %ub step %step {
    %ii = arith.index_cast %i : index to i32
  }
  vc4tile.return
}
