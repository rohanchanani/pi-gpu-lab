// RUN: vc4-opt --legalize-vc4tile-core-cfg --verify-vc4tile-core --convert-vc4tile-to-ssavc4 %s | FileCheck --check-prefix=SSAVC4 %s
// RUN: vc4-opt --legalize-vc4tile-core-cfg --verify-vc4tile-core --convert-vc4tile-to-ssavc4 %s | vc4-opt --convert-ssavc4-to-vc4 | FileCheck --check-prefix=VC4 %s

// SSAVC4: ssavc4.func @scf_for_vc4tile
// SSAVC4: ssavc4.cond_br
// SSAVC4: ssavc4.br
// SSAVC4-NOT: scf.

// VC4: vc4.func @scf_for_vc4tile
// VC4: vc4.qpu.bundle
// VC4-NOT: scf.
vc4tile.kernel @scf_for_vc4tile attributes {public_name = "scf_for_vc4tile"} {
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
