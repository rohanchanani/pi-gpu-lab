// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s

func.func @gemv_row_dot_f32_lowers(
    %a: memref<16xf32, #vc4value.global> {vc4value.arg_name = "a", vc4value.direction = "in"},
    %x: memref<16xf32, #vc4value.global> {vc4value.arg_name = "x", vc4value.direction = "in"},
    %y: memref<16xf32, #vc4value.global> {vc4value.arg_name = "y", vc4value.direction = "out"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                vc4value.fp_domain = "finite",
                vc4value.reduction_policy = "finite_tree"} {
  %row = vc4value.program_id {axis = 0 : i32} : index
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.000000e+00 : f32
  %av = vector.transfer_read %a[%c0], %zero {in_bounds = [true]} : memref<16xf32, #vc4value.global>, vector<16xf32>
  %xv = vector.transfer_read %x[%c0], %zero {in_bounds = [true]} : memref<16xf32, #vc4value.global>, vector<16xf32>
  %prod = arith.mulf %av, %xv : vector<16xf32>
  %dot = vector.reduction <add>, %prod : vector<16xf32> into f32
  memref.store %dot, %y[%row] : memref<16xf32, #vc4value.global>
  return
}

// CHECK-LABEL: vc4kernel.kernel @gemv_row_dot_f32_lowers
// CHECK: vc4kernel.tmu_load_fragment
// CHECK: vc4kernel.tmu_load_fragment
// CHECK: vc4kernel.fragment_alu.mul
// CHECK-SAME: opcode = #vc4kernel.mul_alu_opcode<fmul>
// CHECK: vc4kernel.fragment_reduce
// CHECK-SAME: fp_policy = #vc4kernel.fp_reduce_policy<finite_tree>
// CHECK: vc4kernel.vdw_store_fragment
// CHECK-SAME: inactive_store = #vc4kernel.inactive_store<preserve>
// CHECK-NOT: func.func
// CHECK-NOT: vc4value.
// CHECK-NOT: vector.
// CHECK-NOT: memref.
