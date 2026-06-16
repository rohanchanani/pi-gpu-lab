// REQUIRES: vc4-triton-cpp-frontend

// RUN: not %vc4_triton_opt "%vc4_repo_root/examples/triton/phase16_attention_apply_v0/generated/ttir_attention_apply_nontransposed_v_reject_b16.ttir.mlir" --convert-triton-to-vc4-value -o %t.nontransposed.mlir 2>&1 | FileCheck %s --check-prefix=NONTRANSPOSED
// RUN: not %vc4_triton_opt "%vc4_repo_root/examples/triton/phase16_attention_apply_v0/generated/ttir_attention_apply_scalar_scale_load_reject_b16.ttir.mlir" --convert-triton-to-vc4-value -o %t.scalar_load.mlir 2>&1 | FileCheck %s --check-prefix=SCALARLOAD
// RUN: not %vc4_triton_opt "%vc4_repo_root/examples/triton/phase16_attention_apply_v0/generated/ttir_attention_apply_k_zero_reject_b16.ttir.mlir" --convert-triton-to-vc4-value -o %t.kzero.mlir 2>&1 | FileCheck %s --check-prefix=KZERO
// RUN: not %vc4_triton_opt "%vc4_repo_root/examples/triton/phase16_attention_apply_v0/generated/ttir_attention_apply_multiblock_reject_b16.ttir.mlir" --convert-triton-to-vc4-value -o %t.multiblock.mlir 2>&1 | FileCheck %s --check-prefix=MULTIBLOCK
// RUN: not %vc4_triton_opt "%vc4_repo_root/examples/triton/phase16_attention_apply_v0/generated/ttir_attention_apply_tl_dot_reject_b16.ttir.mlir" --convert-triton-to-vc4-value -o %t.dot.mlir 2>&1 | FileCheck %s --check-prefix=DOT

// NONTRANSPOSED: lane-varying stride/gather pointer expression staged
// NONTRANSPOSED: READY_FOR_TRITON remains NO

// SCALARLOAD: scalar tt.load
// SCALARLOAD: READY_FOR_TRITON remains NO

// KZERO: sparse or unknown tt.load memory mask
// KZERO: READY_FOR_TRITON remains NO

// MULTIBLOCK: multi-block K accumulation is staged
// MULTIBLOCK: READY_FOR_TRITON remains NO

// DOT: tt.dot / contract
// DOT: READY_FOR_TRITON remains NO
