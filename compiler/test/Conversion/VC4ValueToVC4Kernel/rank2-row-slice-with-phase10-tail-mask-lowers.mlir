// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s

func.func @rank2_row_slice_with_phase10_tail_mask_lowers(
    %in: memref<?x?xf32, #vc4value.global> {vc4value.arg_name = "in", vc4value.direction = "in", vc4value.shape_args = ["rows", "cols"]},
    %out: memref<?x?xf32, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out", vc4value.shape_args = ["rows", "cols"]},
    %rows: index {vc4value.arg_name = "rows", vc4value.scalar_role = "extent"},
    %cols: index {vc4value.arg_name = "cols", vc4value.scalar_role = "extent"},
    %active: index {vc4value.arg_name = "active", vc4value.scalar_role = "value"})
    attributes {vc4value.kernel, vc4value.grid_rank = 2 : i32} {
  %pid_col = vc4value.program_id {axis = 0 : i32} : index
  %row = vc4value.program_id {axis = 1 : i32} : index
  %c16 = arith.constant 16 : index
  %col = arith.muli %pid_col, %c16 : index
  %remaining = arith.subi %active, %col : index
  %mask = vector.create_mask %remaining : vector<16xi1>
  %zero = arith.constant 0.000000e+00 : f32
  %v = vector.transfer_read %in[%row, %col], %zero, %mask {in_bounds = [true]} : memref<?x?xf32, #vc4value.global>, vector<16xf32>
  vector.transfer_write %v, %out[%row, %col], %mask {in_bounds = [true]} : vector<16xf32>, memref<?x?xf32, #vc4value.global>
  return
}

// CHECK-LABEL: vc4kernel.kernel @rank2_row_slice_with_phase10_tail_mask_lowers
// CHECK: arith.cmpi slt
// CHECK: arith.select
// CHECK: arith.cmpi sgt
// CHECK: arith.select
// CHECK: vc4kernel.pred.tail
// CHECK: vc4kernel.tmu_load_fragment
// CHECK-SAME: inactive_load = #vc4kernel.inactive_load<zero>
// CHECK: vc4kernel.vdw_store_fragment
// CHECK-SAME: inactive_store = #vc4kernel.inactive_store<preserve>
// CHECK-NOT: func.func
// CHECK-NOT: vc4value.
// CHECK-NOT: vector.
// CHECK-NOT: memref.
