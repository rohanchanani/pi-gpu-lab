// RUN: vc4-opt %s --convert-scf-to-cf --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s

func.func @loop_carried_f32_state_lowers(
    %out: memref<?xf32, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out", vc4value.shape_args = ["n"]},
    %n: index {vc4value.arg_name = "n", vc4value.scalar_role = "extent"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                vc4value.math_policy = "approx_sfu",
                vc4value.fp_domain = "finite",
                vc4value.max_policy = "finite"} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.000000e+00 : f32
  %one = arith.constant 1.000000e+00 : f32
  %state = scf.for %i = %c0 to %n step %c1 iter_args(%s = %zero) -> (f32) {
    %e = math.exp %s : f32
    %next = arith.addf %e, %one : f32
    scf.yield %next : f32
  }
  memref.store %state, %out[%c0] : memref<?xf32, #vc4value.global>
  return
}

// CHECK-LABEL: vc4kernel.kernel @loop_carried_f32_state_lowers
// CHECK: cf.br
// CHECK: ^{{.*}}(%{{.*}}: i32, %{{.*}}: vector<16xf32>)
// CHECK: vc4kernel.fragment_sfu
// CHECK: vc4kernel.fragment_alu.add
// CHECK: vc4kernel.vdw_store_fragment
// CHECK-NOT: scf.
// CHECK-NOT: func.func
// CHECK-NOT: vc4value.
