# REQUIRES: vc4-has-triton-cpp-frontend

# RUN: %vc4_triton_opt "%vc4_repo_root/examples/triton/phase8_5_control_flow/generated/mixed_ttir_cf_loop_if_tail_b16.ttir.mlir" --convert-triton-to-vc4-value -o - | FileCheck %s

# CHECK-LABEL: func.func @mixed_ttir_cf_loop_if_tail_b16_kernel
# CHECK: vc4value.kernel
# CHECK: vector.transfer_read
# CHECK: scf.while
# CHECK: scf.if
# CHECK: vector.transfer_write
# CHECK-NOT: tt.
# CHECK-NOT: triton_gpu
# CHECK-NOT: nvgpu
# CHECK-NOT: nvvm
# CHECK-NOT: gpu.
# CHECK-NOT: vc4kernel
# CHECK-NOT: ssavc4
# CHECK-NOT: vc4.qpu
# CHECK-NOT: arith.sitofp
# CHECK-NOT: vector<16xindex>
