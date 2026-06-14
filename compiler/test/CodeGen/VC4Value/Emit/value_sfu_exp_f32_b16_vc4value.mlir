// RUN: rm -rf %t && mkdir -p %t
// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel -o %t/value_sfu_exp.vc4kernel.mlir
// RUN: FileCheck %s --check-prefix=VC4KERNEL --input-file=%t/value_sfu_exp.vc4kernel.mlir
// RUN: vc4-opt %t/value_sfu_exp.vc4kernel.mlir --verify-vc4kernel --convert-vc4kernel-to-ssavc4 -o %t/value_sfu_exp.ssavc4.mlir
// RUN: FileCheck %s --check-prefix=SSAVC4 --input-file=%t/value_sfu_exp.ssavc4.mlir
// RUN: vc4-opt %t/value_sfu_exp.ssavc4.mlir --convert-ssavc4-to-vc4 --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o %t/value_sfu_exp.vc4.mlir
// RUN: FileCheck %s --check-prefix=VC4 --input-file=%t/value_sfu_exp.vc4.mlir
// RUN: vc4-codegen %t/value_sfu_exp.vc4.mlir --emit-bundle %t/value_sfu_exp.bundle
// RUN: test -f %t/value_sfu_exp.bundle/manifest.json

func.func @value_sfu_exp_f32_b16_vc4value(
    %x: memref<?xf32, #vc4value.global> {vc4value.arg_name = "x", vc4value.direction = "in", vc4value.shape_args = ["n"]},
    %y: memref<?xf32, #vc4value.global> {vc4value.arg_name = "y", vc4value.direction = "out", vc4value.shape_args = ["n"]},
    %n: index {vc4value.arg_name = "n", vc4value.scalar_role = "extent"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                vc4value.math_policy = "approx_sfu",
                vc4value.fp_domain = "finite"} {
  %pid = vc4value.program_id {axis = 0 : i32} : index
  %c16 = arith.constant 16 : index
  %base = arith.muli %pid, %c16 : index
  %remaining = arith.subi %n, %base : index
  %mask = vector.create_mask %remaining : vector<16xi1>
  %zero = arith.constant 0.000000e+00 : f32
  %v = vector.transfer_read %x[%base], %zero, %mask {in_bounds = [true]} : memref<?xf32, #vc4value.global>, vector<16xf32>
  %e = math.exp %v : vector<16xf32>
  vector.transfer_write %e, %y[%base], %mask {in_bounds = [true]} : vector<16xf32>, memref<?xf32, #vc4value.global>
  return
}

// VC4KERNEL-LABEL: vc4kernel.kernel @value_sfu_exp_f32_b16_vc4value
// VC4KERNEL: vc4kernel.fragment_sfu
// VC4KERNEL-SAME: kind = #vc4kernel.sfu_kind<exp>
// SSAVC4-LABEL: ssavc4.func @value_sfu_exp_f32_b16_vc4value
// SSAVC4: ssavc4.sfu
// SSAVC4-SAME: kind = #ssavc4.sfu_kind<exp>
// VC4: vc4.module
// VC4-NOT: ssavc4.
// VC4-NOT: vc4kernel.
