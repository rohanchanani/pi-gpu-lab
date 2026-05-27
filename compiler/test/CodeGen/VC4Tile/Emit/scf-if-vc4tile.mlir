// RUN: vc4-opt --legalize-vc4tile-core-cfg --verify-vc4tile-core --convert-vc4tile-to-ssavc4 %s | FileCheck --check-prefix=SSAVC4 %s
// RUN: vc4-opt --legalize-vc4tile-core-cfg --verify-vc4tile-core --convert-vc4tile-to-ssavc4 %s | vc4-opt --convert-ssavc4-to-vc4 | FileCheck --check-prefix=VC4 %s

// SSAVC4: ssavc4.func @scf_if_vc4tile
// SSAVC4: ssavc4.cond_br
// SSAVC4: ssavc4.br
// SSAVC4-NOT: scf.

// VC4: vc4.func @scf_if_vc4tile
// VC4: vc4.qpu.bundle
// VC4-NOT: scf.
vc4tile.kernel @scf_if_vc4tile attributes {public_name = "scf_if_vc4tile"} {
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
