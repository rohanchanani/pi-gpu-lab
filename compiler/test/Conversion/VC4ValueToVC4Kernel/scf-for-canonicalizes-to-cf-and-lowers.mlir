// RUN: vc4-opt %s --convert-scf-to-cf --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s

func.func @scf_for_canonicalizes_to_cf_and_lowers(%n: index {vc4value.arg_name = "n"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant dense<0.000000e+00> : vector<16xf32>
  %one = arith.constant dense<1.000000e+00> : vector<16xf32>
  %result = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %zero) -> (vector<16xf32>) {
    %next = arith.addf %acc, %one : vector<16xf32>
    scf.yield %next : vector<16xf32>
  }
  %sink = arith.addf %result, %zero : vector<16xf32>
  return
}

// CHECK-LABEL: vc4kernel.kernel @scf_for_canonicalizes_to_cf_and_lowers
// CHECK: cf.br
// CHECK: ^{{.*}}(%{{.*}}: i32, %{{.*}}: vector<16xf32>)
// CHECK: cf.cond_br
// CHECK: vc4kernel.fragment_alu.add
// CHECK: vc4kernel.return
// CHECK-NOT: scf.
// CHECK-NOT: func.func
// CHECK-NOT: vc4value.
// CHECK-NOT: vector.
// CHECK-NOT: memref.
