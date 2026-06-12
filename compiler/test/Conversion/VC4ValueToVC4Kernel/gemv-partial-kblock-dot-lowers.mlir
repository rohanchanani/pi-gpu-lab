// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s

func.func @gemv_partial_kblock_dot_lowers(
    %a: memref<?x?xf32, strided<[?, 1], offset: 0>, #vc4value.global> {vc4value.arg_name = "a", vc4value.direction = "in", vc4value.shape_args = ["rows", "cols"], vc4value.stride_args = ["lda"]},
    %x: memref<?xf32, #vc4value.global> {vc4value.arg_name = "x", vc4value.direction = "in", vc4value.shape_args = ["cols"]},
    %partials: memref<?xf32, #vc4value.global> {vc4value.arg_name = "partials", vc4value.direction = "out", vc4value.shape_args = ["partial_count"]},
    %rows: index {vc4value.arg_name = "rows", vc4value.scalar_role = "extent"},
    %cols: index {vc4value.arg_name = "cols", vc4value.scalar_role = "extent"},
    %lda: index {vc4value.arg_name = "lda", vc4value.scalar_role = "stride"},
    %num_kblocks: index {vc4value.arg_name = "num_kblocks", vc4value.scalar_role = "extent"},
    %partial_count: index {vc4value.arg_name = "partial_count", vc4value.scalar_role = "extent"})
    attributes {vc4value.kernel, vc4value.grid_rank = 2 : i32,
                vc4value.fp_domain = "finite",
                vc4value.reduction_policy = "finite_tree"} {
  %kblock = vc4value.program_id {axis = 0 : i32} : index
  %row = vc4value.program_id {axis = 1 : i32} : index
  %c16 = arith.constant 16 : index
  %kbase = arith.muli %kblock, %c16 : index
  %remaining = arith.subi %cols, %kbase : index
  %mask = vector.create_mask %remaining : vector<16xi1>
  %zero = arith.constant 0.000000e+00 : f32
  %av = vector.transfer_read %a[%row, %kbase], %zero, %mask {in_bounds = [true]} : memref<?x?xf32, strided<[?, 1], offset: 0>, #vc4value.global>, vector<16xf32>
  %xv = vector.transfer_read %x[%kbase], %zero, %mask {in_bounds = [true]} : memref<?xf32, #vc4value.global>, vector<16xf32>
  %prod = arith.mulf %av, %xv : vector<16xf32>
  %dot = vector.reduction <add>, %prod : vector<16xf32> into f32
  %row_base = arith.muli %row, %num_kblocks : index
  %out = arith.addi %row_base, %kblock : index
  memref.store %dot, %partials[%out] : memref<?xf32, #vc4value.global>
  return
}

// CHECK-LABEL: vc4kernel.kernel @gemv_partial_kblock_dot_lowers
// CHECK: vc4kernel.program_id
// CHECK-SAME: axis = 0
// CHECK: vc4kernel.program_id
// CHECK-SAME: axis = 1
// CHECK: vc4kernel.fragment_alu.mul
// CHECK-SAME: opcode = #vc4kernel.mul_alu_opcode<fmul>
// CHECK: vc4kernel.fragment_reduce
// CHECK-SAME: fp_policy = #vc4kernel.fp_reduce_policy<finite_tree>
// CHECK: vc4kernel.vdw_store_fragment
// CHECK-NOT: func.func
// CHECK-NOT: vc4value.
// CHECK-NOT: vector.
// CHECK-NOT: memref.
