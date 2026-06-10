// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s

func.func @row_strided_reduction_sum_lowers(
    %in: memref<?x?xf32, strided<[?, 1], offset: 0>, #vc4value.global> {vc4value.arg_name = "in", vc4value.direction = "in", vc4value.shape_args = ["rows", "cols"], vc4value.stride_args = ["row_stride"]},
    %out: memref<?xf32, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out", vc4value.shape_args = ["rows"]},
    %rows: index {vc4value.arg_name = "rows", vc4value.scalar_role = "extent"},
    %cols: index {vc4value.arg_name = "cols", vc4value.scalar_role = "extent"},
    %row_stride: index {vc4value.arg_name = "row_stride", vc4value.scalar_role = "stride"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                vc4value.fp_domain = "finite",
                vc4value.reduction_policy = "finite_tree"} {
  %row = vc4value.program_id {axis = 0 : i32} : index
  %c0 = arith.constant 0 : index
  %mask = vector.create_mask %cols : vector<16xi1>
  %zero = arith.constant 0.000000e+00 : f32
  %v = vector.transfer_read %in[%row, %c0], %zero, %mask {in_bounds = [true]} : memref<?x?xf32, strided<[?, 1], offset: 0>, #vc4value.global>, vector<16xf32>
  %sum = vector.reduction <add>, %v : vector<16xf32> into f32
  memref.store %sum, %out[%row] : memref<?xf32, #vc4value.global>
  return
}

// CHECK-LABEL: vc4kernel.kernel @row_strided_reduction_sum_lowers
// CHECK-SAME: %[[IN:arg[0-9]+]] : i32
// CHECK-SAME: %[[OUT:arg[0-9]+]] : i32
// CHECK: arith.muli {{.*}}, {{.*}} : i32
// CHECK: vc4kernel.tmu_load_fragment %[[IN]]
// CHECK: vc4kernel.fragment_reduce
// CHECK-SAME: fp_policy = #vc4kernel.fp_reduce_policy<finite_tree>
// CHECK: vc4kernel.vdw_store_fragment %[[OUT]]
// CHECK-NOT: vector.
// CHECK-NOT: memref.
