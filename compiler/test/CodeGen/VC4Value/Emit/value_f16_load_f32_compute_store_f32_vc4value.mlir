// RUN: rm -rf %t && mkdir -p %t
// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel -o %t/value_f16_load.vc4kernel.mlir
// RUN: FileCheck %s --check-prefix=VC4KERNEL --input-file=%t/value_f16_load.vc4kernel.mlir
// RUN: vc4-opt %t/value_f16_load.vc4kernel.mlir --verify-vc4kernel --convert-vc4kernel-to-ssavc4 -o %t/value_f16_load.ssavc4.mlir
// RUN: FileCheck %s --check-prefix=SSAVC4 --input-file=%t/value_f16_load.ssavc4.mlir
// RUN: vc4-opt %t/value_f16_load.ssavc4.mlir --convert-ssavc4-to-vc4 --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o %t/value_f16_load.vc4.mlir
// RUN: FileCheck %s --check-prefix=VC4 --input-file=%t/value_f16_load.vc4.mlir

func.func @value_f16_load_f32_compute_store_f32_vc4value(
    %in: memref<?xf16, #vc4value.global> {vc4value.arg_name = "in", vc4value.direction = "in", vc4value.shape_args = ["n"]},
    %out: memref<?xf32, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out", vc4value.shape_args = ["n"]},
    %n: index {vc4value.arg_name = "n", vc4value.scalar_role = "extent"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %pid = vc4value.program_id {axis = 0 : i32} : index
  %c16 = arith.constant 16 : index
  %base = arith.muli %pid, %c16 : index
  %remaining = arith.subi %n, %base : index
  %mask = vector.create_mask %remaining : vector<16xi1>
  %zero_h = arith.constant 0.000000e+00 : f16
  %loaded = vector.transfer_read %in[%base], %zero_h, %mask {in_bounds = [true]} : memref<?xf16, #vc4value.global>, vector<16xf16>
  %wide = arith.extf %loaded : vector<16xf16> to vector<16xf32>
  %one = arith.constant dense<1.000000e+00> : vector<16xf32>
  %sum = arith.addf %wide, %one : vector<16xf32>
  vector.transfer_write %sum, %out[%base], %mask {in_bounds = [true]} : vector<16xf32>, memref<?xf32, #vc4value.global>
  return
}

// VC4KERNEL-LABEL: vc4kernel.kernel @value_f16_load_f32_compute_store_f32_vc4value
// VC4KERNEL: vc4kernel.vdr_load_rect_to_vpm
// VC4KERNEL: vc4kernel.fragment_unpack
// VC4KERNEL: vc4kernel.fragment_alu.add
// VC4KERNEL: vc4kernel.vdw_store_fragment
// VC4KERNEL-NOT: vector.
// VC4KERNEL-NOT: memref.

// SSAVC4-LABEL: ssavc4.func @value_f16_load_f32_compute_store_f32_vc4value
// SSAVC4: ssavc4.vdr.load
// SSAVC4: f16_storage_conversion
// SSAVC4: ssavc4.vdw.store

// VC4: vc4.module
// VC4: vc4.qpu.
// VC4-NOT: ssavc4.
// VC4-NOT: vc4kernel.
