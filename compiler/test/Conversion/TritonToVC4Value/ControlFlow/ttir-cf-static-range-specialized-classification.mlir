# REQUIRES: vc4-has-triton-cpp-frontend

# RUN: %vc4_triton_opt "%vc4_repo_root/examples/triton/phase8_5_control_flow/generated/ttir_cf_static_range_unrolled_b16.ttir.mlir" --convert-triton-to-vc4-value -o - | FileCheck %s

# CHECK-LABEL: func.func @ttir_cf_static_range_unrolled_b16_kernel
# CHECK: vc4value.kernel
# CHECK: vector.transfer_read
# CHECK: vector.transfer_write
# CHECK-NOT: scf.for
# CHECK-NOT: scf.while
# CHECK-NOT: tt.
# CHECK-NOT: arith.sitofp
# CHECK-NOT: vector<16xindex>
