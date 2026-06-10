// RUN: rm -rf %t && mkdir -p %t
// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel -o %t/value_mask_compute_select_tail.vc4kernel.mlir
// RUN: FileCheck %s --check-prefix=VC4KERNEL --input-file=%t/value_mask_compute_select_tail.vc4kernel.mlir
// RUN: vc4-opt %t/value_mask_compute_select_tail.vc4kernel.mlir --verify-vc4kernel --convert-vc4kernel-to-ssavc4 -o %t/value_mask_compute_select_tail.ssavc4.mlir
// RUN: FileCheck %s --check-prefix=SSAVC4 --input-file=%t/value_mask_compute_select_tail.ssavc4.mlir
// RUN: vc4-opt %t/value_mask_compute_select_tail.ssavc4.mlir --convert-ssavc4-to-vc4 --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o %t/value_mask_compute_select_tail.vc4.mlir
// RUN: FileCheck %s --check-prefix=VC4 --input-file=%t/value_mask_compute_select_tail.vc4.mlir

func.func @value_mask_compute_select_tail_vc4value(
    %in: memref<64xf32, #vc4value.global> {vc4value.arg_name = "in", vc4value.direction = "in"},
    %out: memref<64xf32, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out"},
    %base: index {vc4value.arg_name = "base"},
    %n: index {vc4value.arg_name = "n"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                vc4value.fp_domain = "finite"} {
  %zero = arith.constant 0.000000e+00 : f32
  %zero_v = arith.constant dense<0.000000e+00> : vector<16xf32>
  %tail = vector.create_mask %n : vector<16xi1>
  %x = vector.transfer_read %in[%base], %zero, %tail {in_bounds = [true]} : memref<64xf32, #vc4value.global>, vector<16xf32>
  %m = arith.cmpf ogt, %x, %zero_v : vector<16xf32>
  %sel = arith.select %m, %x, %zero_v : vector<16xi1>, vector<16xf32>
  vector.transfer_write %sel, %out[%base], %tail {in_bounds = [true]} : vector<16xf32>, memref<64xf32, #vc4value.global>
  return
}

// VC4KERNEL-LABEL: vc4kernel.kernel @value_mask_compute_select_tail_vc4value
// VC4KERNEL: vc4kernel.pred.tail
// VC4KERNEL: vc4kernel.tmu_load_fragment
// VC4KERNEL: vc4kernel.fragment_cmp
// VC4KERNEL: vc4kernel.fragment_select
// VC4KERNEL: vc4kernel.vdw_store_fragment
// VC4KERNEL-NOT: sparse or unknown transfer
// VC4KERNEL-NOT: func.func
// VC4KERNEL-NOT: vc4value.
// VC4KERNEL-NOT: vector.
// VC4KERNEL-NOT: memref.

// SSAVC4-LABEL: ssavc4.func @value_mask_compute_select_tail_vc4value
// SSAVC4: ssavc4.tmu.request
// SSAVC4: ssavc4.vdw.store
// SSAVC4-NOT: vc4kernel.

// VC4: vc4.module
// VC4: vc4.qpu.
// VC4-NOT: ssavc4.
// VC4-NOT: vc4kernel.
