// RUN: rm -rf %t && mkdir -p %t
// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel -o %t/mixed_value_gemv_row_dot.vc4kernel.mlir
// RUN: FileCheck %s --check-prefix=VC4KERNEL --input-file=%t/mixed_value_gemv_row_dot.vc4kernel.mlir
// RUN: vc4-opt %t/mixed_value_gemv_row_dot.vc4kernel.mlir --verify-vc4kernel --convert-vc4kernel-to-ssavc4 -o %t/mixed_value_gemv_row_dot.ssavc4.mlir
// RUN: FileCheck %s --check-prefix=SSAVC4 --input-file=%t/mixed_value_gemv_row_dot.ssavc4.mlir
// RUN: vc4-opt %t/mixed_value_gemv_row_dot.ssavc4.mlir --convert-ssavc4-to-vc4 --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o %t/mixed_value_gemv_row_dot.vc4.mlir
// RUN: FileCheck %s --check-prefix=VC4 --input-file=%t/mixed_value_gemv_row_dot.vc4.mlir

func.func @mixed_value_gemv_row_dot_axes_mask_cf_strided_reduction_vc4value(
    %a: memref<?x?xf32, strided<[?, 1], offset: 0>, #vc4value.global> {vc4value.arg_name = "a", vc4value.direction = "in", vc4value.shape_args = ["rows", "cols"], vc4value.stride_args = ["lda"]},
    %x: memref<?xf32, #vc4value.global> {vc4value.arg_name = "x", vc4value.direction = "in", vc4value.shape_args = ["cols"]},
    %y: memref<?xf32, #vc4value.global> {vc4value.arg_name = "y", vc4value.direction = "out", vc4value.shape_args = ["rows"]},
    %rows: index {vc4value.arg_name = "rows", vc4value.scalar_role = "extent"},
    %cols: index {vc4value.arg_name = "cols", vc4value.scalar_role = "extent"},
    %lda: index {vc4value.arg_name = "lda", vc4value.scalar_role = "stride"})
    attributes {vc4value.kernel, vc4value.grid_rank = 2 : i32,
                vc4value.fp_domain = "finite",
                vc4value.reduction_policy = "finite_tree"} {
  %pid0 = vc4value.program_id {axis = 0 : i32} : index
  %row = vc4value.program_id {axis = 1 : i32} : index
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %col = arith.muli %pid0, %c16 : index
  %remaining = arith.subi %cols, %col : index
  %mask = vector.create_mask %remaining : vector<16xi1>
  %zero = arith.constant 0.000000e+00 : f32
  %av = vector.transfer_read %a[%row, %col], %zero, %mask {in_bounds = [true]} : memref<?x?xf32, strided<[?, 1], offset: 0>, #vc4value.global>, vector<16xf32>
  %xv = vector.transfer_read %x[%col], %zero, %mask {in_bounds = [true]} : memref<?xf32, #vc4value.global>, vector<16xf32>
  %prod = arith.mulf %av, %xv : vector<16xf32>
  %dot = vector.reduction <add>, %prod : vector<16xf32> into f32
  %is_first_block = arith.cmpi eq, %pid0, %c0 : index
  cf.cond_br %is_first_block, ^store, ^done
^store:
  memref.store %dot, %y[%row] : memref<?xf32, #vc4value.global>
  cf.br ^done
^done:
  return
}

// VC4KERNEL-LABEL: vc4kernel.kernel @mixed_value_gemv_row_dot_axes_mask_cf_strided_reduction_vc4value
// VC4KERNEL: vc4kernel.program_id
// VC4KERNEL-SAME: axis = 0
// VC4KERNEL: vc4kernel.program_id
// VC4KERNEL-SAME: axis = 1
// VC4KERNEL: vc4kernel.pred.tail
// VC4KERNEL: vc4kernel.fragment_alu.mul
// VC4KERNEL: vc4kernel.fragment_reduce
// VC4KERNEL: cf.cond_br
// VC4KERNEL: vc4kernel.vdw_store_fragment
// VC4KERNEL-NOT: vector.
// VC4KERNEL-NOT: memref.

// SSAVC4-LABEL: ssavc4.func @mixed_value_gemv_row_dot_axes_mask_cf_strided_reduction_vc4value
// SSAVC4: program_id_x
// SSAVC4: program_id_y
// SSAVC4: ssavc4.tmu.request
// SSAVC4: ssavc4.vdw.store

// VC4: vc4.module
// VC4: vc4.qpu.
// VC4-NOT: ssavc4.
// VC4-NOT: vc4kernel.
