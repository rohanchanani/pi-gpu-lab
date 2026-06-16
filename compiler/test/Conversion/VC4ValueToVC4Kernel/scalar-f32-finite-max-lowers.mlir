// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s

func.func @scalar_f32_finite_max_lowers(
    %out: memref<?xf32, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out", vc4value.shape_args = ["n"]},
    %n: index {vc4value.arg_name = "n", vc4value.scalar_role = "extent"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                vc4value.fp_domain = "finite",
                vc4value.max_policy = "finite"} {
  %c0 = arith.constant 0 : index
  %a = arith.constant 1.000000e+00 : f32
  %b = arith.constant 2.000000e+00 : f32
  %m = arith.maxnumf %a, %b : f32
  memref.store %m, %out[%c0] : memref<?xf32, #vc4value.global>
  return
}

// CHECK-LABEL: vc4kernel.kernel @scalar_f32_finite_max_lowers
// CHECK: vc4kernel.fragment_cmp
// CHECK-SAME: predicate = #vc4kernel.cmp<ogt>
// CHECK: vc4kernel.fragment_select
// CHECK: vc4kernel.vdw_store_fragment
// CHECK-NOT: arith.maxnumf
// CHECK-NOT: func.func
// CHECK-NOT: vc4value.
