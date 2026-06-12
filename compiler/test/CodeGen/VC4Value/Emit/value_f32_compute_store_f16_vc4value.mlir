// RUN: rm -rf %t && mkdir -p %t
// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel -o %t/value_f16_store.vc4kernel.mlir
// RUN: FileCheck %s --check-prefix=VC4KERNEL --input-file=%t/value_f16_store.vc4kernel.mlir
// RUN: vc4-opt %t/value_f16_store.vc4kernel.mlir --verify-vc4kernel --convert-vc4kernel-to-ssavc4 -o %t/value_f16_store.ssavc4.mlir
// RUN: FileCheck %s --check-prefix=SSAVC4 --input-file=%t/value_f16_store.ssavc4.mlir
// RUN: vc4-opt %t/value_f16_store.ssavc4.mlir --convert-ssavc4-to-vc4 --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o %t/value_f16_store.vc4.mlir
// RUN: FileCheck %s --check-prefix=VC4 --input-file=%t/value_f16_store.vc4.mlir

func.func @value_f32_compute_store_f16_vc4value(
    %in: memref<?xf32, #vc4value.global> {vc4value.arg_name = "in", vc4value.direction = "in", vc4value.shape_args = ["n"]},
    %out: memref<?xf16, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out", vc4value.shape_args = ["n"]},
    %n: index {vc4value.arg_name = "n", vc4value.scalar_role = "extent"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                vc4value.f16_storage_policy = "finite"} {
  %pid = vc4value.program_id {axis = 0 : i32} : index
  %c16 = arith.constant 16 : index
  %base = arith.muli %pid, %c16 : index
  %remaining = arith.subi %n, %base : index
  %mask = vector.create_mask %remaining : vector<16xi1>
  %zero = arith.constant 0.000000e+00 : f32
  %loaded = vector.transfer_read %in[%base], %zero, %mask {in_bounds = [true]} : memref<?xf32, #vc4value.global>, vector<16xf32>
  %bias = arith.constant dense<2.500000e-01> : vector<16xf32>
  %sum = arith.addf %loaded, %bias : vector<16xf32>
  %narrow = arith.truncf %sum : vector<16xf32> to vector<16xf16>
  vector.transfer_write %narrow, %out[%base], %mask {in_bounds = [true]} : vector<16xf16>, memref<?xf16, #vc4value.global>
  return
}

// VC4KERNEL-LABEL: vc4kernel.kernel @value_f32_compute_store_f16_vc4value
// VC4KERNEL: vc4kernel.fragment_pack
// VC4KERNEL: vc4kernel.vpm_write_fragment
// VC4KERNEL: vc4kernel.vdw_store_rect_from_vpm
// VC4KERNEL-NOT: vector.
// VC4KERNEL-NOT: memref.

// SSAVC4-LABEL: ssavc4.func @value_f32_compute_store_f16_vc4value
// SSAVC4: f16_storage_conversion
// SSAVC4: ssavc4.vdw.store

// VC4: vc4.module
// VC4: vc4.qpu.
// VC4-NOT: ssavc4.
// VC4-NOT: vc4kernel.
