// RUN: vc4-opt --legalize-vc4tile-core-cfg --verify-vc4tile-core --convert-vc4tile-to-ssavc4 %s | FileCheck %s

// CHECK: ssavc4.func @pipeline_scf_if_semantics_to_ssavc4
// CHECK: ssavc4.cond_br
// CHECK-SAME: #vc4.branch_cond<any_z_clear>
// CHECK-NOT: scf.
// CHECK-NOT: vc4tile.
// CHECK-NOT: index

vc4tile.kernel @pipeline_scf_if_semantics_to_ssavc4 attributes {public_name = "pipeline_scf_if_semantics_to_ssavc4"} {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %cond = arith.cmpi ne, %one, %zero : i32
  %total = scf.if %cond -> (i32) {
    %a = arith.constant 41 : i32
    scf.yield %a : i32
  } else {
    %b = arith.constant 5 : i32
    scf.yield %b : i32
  }
  vc4tile.return
}
