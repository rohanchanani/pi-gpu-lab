// REQUIRES: vc4-has-triton-cpp-frontend
// RUN: %vc4_triton_opt "%vc4_repo_root/examples/triton/phase15_sfu_softmax/generated/ttir_sfu_sqrt_f32_b16.ttir.mlir" --convert-triton-to-vc4-value -o - | FileCheck %s --check-prefix=VALUE --implicit-check-not='tt.' --implicit-check-not='ttg.' --implicit-check-not='vc4kernel.' --implicit-check-not='ssavc4.'
// RUN: %vc4_triton_opt "%vc4_repo_root/examples/triton/phase15_sfu_softmax/generated/ttir_sfu_sqrt_f32_b16.ttir.mlir" --convert-triton-to-vc4-value -o %t.value.mlir
// RUN: %vc4_opt_triton %t.value.mlir --vc4-verify-value-surface --convert-scf-to-cf --convert-vc4-value-to-vc4kernel --verify-vc4kernel -o - | FileCheck %s --check-prefix=VC4KERNEL --implicit-check-not='math.sqrt'
// RUN: %vc4_opt_triton %t.value.mlir --vc4-verify-value-surface --convert-scf-to-cf --convert-vc4-value-to-vc4kernel --verify-vc4kernel -o %t.vc4kernel.mlir
// RUN: %vc4_opt_triton %t.vc4kernel.mlir --convert-vc4kernel-to-ssavc4 -o %t.ssavc4.mlir
// RUN: %vc4_opt_triton %t.ssavc4.mlir --convert-ssavc4-to-vc4 --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o %t.vc4.mlir

// VALUE-LABEL: func.func @ttir_sfu_sqrt_f32_b16
// VALUE-SAME: vc4value.fp_domain = "finite_positive"
// VALUE-SAME: vc4value.math_policy = "approx_sfu"
// VALUE: math.sqrt
// VALUE-SAME: vc4value.fp_domain = "finite_positive"
// VALUE-SAME: vc4value.math_policy = "approx_sfu"

// VC4KERNEL-LABEL: vc4kernel.kernel @ttir_sfu_sqrt_f32_b16
// VC4KERNEL: vc4kernel.fragment_sfu
// VC4KERNEL-SAME: kind = #vc4kernel.sfu_kind<rsqrt>
// VC4KERNEL-NEXT: vc4kernel.fragment_alu.mul
