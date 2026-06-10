// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s

func.func @memref_dim_shape_arg_lowers(
    %in: memref<?x?xi32, strided<[?, 1], offset: 0>, #vc4value.global> {vc4value.arg_name = "in", vc4value.direction = "in", vc4value.shape_args = ["rows", "cols"], vc4value.stride_args = ["row_stride"]},
    %out: memref<?x?xi32, strided<[?, 1], offset: 0>, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out", vc4value.shape_args = ["rows", "cols"], vc4value.stride_args = ["row_stride"]},
    %rows: index {vc4value.arg_name = "rows", vc4value.scalar_role = "extent"},
    %cols: index {vc4value.arg_name = "cols", vc4value.scalar_role = "extent"},
    %row_stride: index {vc4value.arg_name = "row_stride", vc4value.scalar_role = "stride"})
    attributes {vc4value.kernel, vc4value.grid_rank = 2 : i32} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %pid_col = vc4value.program_id {axis = 0 : i32} : index
  %row = vc4value.program_id {axis = 1 : i32} : index
  %dim_rows = memref.dim %in, %c0 : memref<?x?xi32, strided<[?, 1], offset: 0>, #vc4value.global>
  %dim_cols = memref.dim %in, %c1 : memref<?x?xi32, strided<[?, 1], offset: 0>, #vc4value.global>
  %inside = arith.cmpi ult, %row, %dim_rows : index
  cf.cond_br %inside, ^copy, ^done
^copy:
  %c16 = arith.constant 16 : index
  %col = arith.muli %pid_col, %c16 : index
  %remaining = arith.subi %dim_cols, %col : index
  %mask = vector.create_mask %remaining : vector<16xi1>
  %zero = arith.constant 0 : i32
  %v = vector.transfer_read %in[%row, %col], %zero, %mask {in_bounds = [true]} : memref<?x?xi32, strided<[?, 1], offset: 0>, #vc4value.global>, vector<16xi32>
  vector.transfer_write %v, %out[%row, %col], %mask {in_bounds = [true]} : vector<16xi32>, memref<?x?xi32, strided<[?, 1], offset: 0>, #vc4value.global>
  cf.br ^done
^done:
  return
}

// CHECK-LABEL: vc4kernel.kernel @memref_dim_shape_arg_lowers
// CHECK-SAME: %[[IN:arg[0-9]+]] : i32
// CHECK-SAME: %[[OUT:arg[0-9]+]] : i32
// CHECK-SAME: %[[ROWS:arg[0-9]+]] : i32
// CHECK-SAME: %[[COLS:arg[0-9]+]] : i32
// CHECK-SAME: %[[STRIDE:arg[0-9]+]] : i32
// CHECK: arith.cmpi ult, {{.*}}, %[[ROWS]] : i32
// CHECK: arith.subi %[[COLS]]
// CHECK: arith.muli {{.*}}, %[[STRIDE]] : i32
// CHECK: vc4kernel.tmu_load_fragment %[[IN]]
// CHECK: vc4kernel.vdw_store_fragment %[[OUT]]
// CHECK-NOT: memref.dim
// CHECK-NOT: func.func
// CHECK-NOT: vc4value.
// CHECK-NOT: vector.
// CHECK-NOT: memref.
