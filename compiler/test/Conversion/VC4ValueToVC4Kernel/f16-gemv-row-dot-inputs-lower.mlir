// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s

func.func @f16_gemv_row_dot_inputs_lower(
    %a: memref<?x?xf16, strided<[?, 1], offset: 0>, #vc4value.global> {vc4value.arg_name = "a", vc4value.direction = "in", vc4value.shape_args = ["rows", "cols"], vc4value.stride_args = ["lda"]},
    %x: memref<?xf16, #vc4value.global> {vc4value.arg_name = "x", vc4value.direction = "in", vc4value.shape_args = ["cols"]},
    %y: memref<?xf32, #vc4value.global> {vc4value.arg_name = "y", vc4value.direction = "out", vc4value.shape_args = ["rows"]},
    %rows: index {vc4value.arg_name = "rows", vc4value.scalar_role = "extent"},
    %cols: index {vc4value.arg_name = "cols", vc4value.scalar_role = "extent"},
    %lda: index {vc4value.arg_name = "lda", vc4value.scalar_role = "stride"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                vc4value.fp_domain = "finite",
                vc4value.reduction_policy = "finite_tree"} {
  %row = vc4value.program_id {axis = 0 : i32} : index
  %c0 = arith.constant 0 : index
  %mask = vector.create_mask %cols : vector<16xi1>
  %zero_h = arith.constant 0.000000e+00 : f16
  %av_h = vector.transfer_read %a[%row, %c0], %zero_h, %mask {in_bounds = [true]} : memref<?x?xf16, strided<[?, 1], offset: 0>, #vc4value.global>, vector<16xf16>
  %xv_h = vector.transfer_read %x[%c0], %zero_h, %mask {in_bounds = [true]} : memref<?xf16, #vc4value.global>, vector<16xf16>
  %av = arith.extf %av_h : vector<16xf16> to vector<16xf32>
  %xv = arith.extf %xv_h : vector<16xf16> to vector<16xf32>
  %prod = arith.mulf %av, %xv : vector<16xf32>
  %dot = vector.reduction <add>, %prod : vector<16xf32> into f32
  memref.store %dot, %y[%row] : memref<?xf32, #vc4value.global>
  return
}

// CHECK-LABEL: vc4kernel.kernel @f16_gemv_row_dot_inputs_lower
// CHECK: vc4kernel.vdr_load_rect_to_vpm
// CHECK-SAME: elem_bytes = 2
// CHECK: vc4kernel.vdr_load_rect_to_vpm
// CHECK-SAME: elem_bytes = 2
// CHECK: vc4kernel.fragment_unpack
// CHECK: vc4kernel.fragment_unpack
// CHECK: vc4kernel.fragment_alu.mul
// CHECK: vc4kernel.fragment_reduce
// CHECK-SAME: fp_policy = #vc4kernel.fp_reduce_policy<finite_tree>
// CHECK: vc4kernel.vdw_store_fragment
// CHECK-NOT: vector.transfer
// CHECK-NOT: memref.
