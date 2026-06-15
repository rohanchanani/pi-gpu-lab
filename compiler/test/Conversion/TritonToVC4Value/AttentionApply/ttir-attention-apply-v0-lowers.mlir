// REQUIRES: vc4-triton-cpp-frontend

// RUN: %vc4_triton_opt "%vc4_repo_root/examples/triton/phase16_attention_apply_v0/generated/ttir_attention_apply_v0_f32_b16.ttir.mlir" --convert-triton-to-vc4-value -o - | FileCheck %s --check-prefix=BASE
// RUN: %vc4_triton_opt "%vc4_repo_root/examples/triton/phase16_attention_apply_v0/generated/ttir_attention_apply_v0_scaled_f32_b16.ttir.mlir" --convert-triton-to-vc4-value -o - | FileCheck %s --check-prefix=SCALED
// RUN: %vc4_triton_opt "%vc4_repo_root/examples/triton/phase16_attention_apply_v0/generated/mixed_ttir_attention_apply_v0_axes_mask_cf_f16_storage_b16.ttir.mlir" --convert-triton-to-vc4-value -o - | FileCheck %s --check-prefix=MIXED

// BASE-LABEL: func.func @ttir_attention_apply_v0_f32_b16_kernel
// BASE-SAME: vc4value.kernel
// BASE: vector.transfer_read
// BASE: vector.reduction <maxnumf>
// BASE: math.exp
// BASE: vector.reduction <add>
// BASE: vector.transfer_read
// BASE: arith.mulf
// BASE: memref.store
// BASE-NOT: tt.
// BASE-NOT: vc4kernel.

// SCALED-LABEL: func.func @ttir_attention_apply_v0_scaled_f32_b16_kernel
// SCALED-SAME: %{{.*}}: f32
// SCALED: vector.broadcast
// SCALED: arith.mulf
// SCALED: math.exp
// SCALED: memref.store
// SCALED-NOT: tt.

// MIXED-LABEL: func.func @mixed_ttir_attention_apply_v0_axes_mask_cf_f16_storage_b16_kernel
// MIXED-SAME: memref<?xf16, #vc4value.global>
// MIXED: vc4value.program_id
// MIXED: scf.if
// MIXED: math.exp
// MIXED: arith.extf
// MIXED: memref.store
// MIXED-NOT: tt.
