// RUN: rm -rf %t && mkdir -p %t
// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel -o %t/value_gemv_empty_repeat.vc4kernel.mlir
// RUN: FileCheck %s --check-prefix=VC4KERNEL --input-file=%t/value_gemv_empty_repeat.vc4kernel.mlir
// RUN: vc4-opt %t/value_gemv_empty_repeat.vc4kernel.mlir --verify-vc4kernel --convert-vc4kernel-to-ssavc4 -o %t/value_gemv_empty_repeat.ssavc4.mlir
// RUN: FileCheck %s --check-prefix=SSAVC4 --input-file=%t/value_gemv_empty_repeat.ssavc4.mlir
// RUN: vc4-opt %t/value_gemv_empty_repeat.ssavc4.mlir --convert-ssavc4-to-vc4 --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o %t/value_gemv_empty_repeat.vc4.mlir
// RUN: FileCheck %s --check-prefix=VC4 --input-file=%t/value_gemv_empty_repeat.vc4.mlir

func.func @value_gemv_empty_repeat_vc4value(
    %a: memref<?xf32, #vc4value.global> {vc4value.arg_name = "a", vc4value.direction = "in", vc4value.shape_args = ["n"]},
    %x: memref<?xf32, #vc4value.global> {vc4value.arg_name = "x", vc4value.direction = "in", vc4value.shape_args = ["n"]},
    %y: memref<?xf32, #vc4value.global> {vc4value.arg_name = "y", vc4value.direction = "out", vc4value.shape_args = ["rows"]},
    %n: index {vc4value.arg_name = "n", vc4value.scalar_role = "extent"},
    %rows: index {vc4value.arg_name = "rows", vc4value.scalar_role = "extent"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                vc4value.fp_domain = "finite",
                vc4value.reduction_policy = "finite_tree"} {
  %row = vc4value.program_id {axis = 0 : i32} : index
  %c0 = arith.constant 0 : index
  %empty = vector.create_mask %c0 : vector<16xi1>
  %zero = arith.constant 0.000000e+00 : f32
  %av = vector.transfer_read %a[%c0], %zero, %empty {in_bounds = [true]} : memref<?xf32, #vc4value.global>, vector<16xf32>
  %xv = vector.transfer_read %x[%c0], %zero, %empty {in_bounds = [true]} : memref<?xf32, #vc4value.global>, vector<16xf32>
  %prod0 = arith.mulf %av, %xv : vector<16xf32>
  %dot0 = vector.reduction <add>, %prod0 : vector<16xf32> into f32
  memref.store %dot0, %y[%row] : memref<?xf32, #vc4value.global>
  return
}

// VC4KERNEL-LABEL: vc4kernel.kernel @value_gemv_empty_repeat_vc4value
// VC4KERNEL: vc4kernel.pred.empty
// VC4KERNEL: vc4kernel.fragment_alu.mul
// VC4KERNEL: vc4kernel.fragment_reduce
// VC4KERNEL: vc4kernel.vdw_store_fragment
// VC4KERNEL-NOT: vector.
// VC4KERNEL-NOT: memref.

// SSAVC4-LABEL: ssavc4.func @value_gemv_empty_repeat_vc4value
// SSAVC4: ssavc4.vdw.store

// VC4: vc4.module
// VC4: vc4.qpu.
// VC4-NOT: ssavc4.
// VC4-NOT: vc4kernel.
