// RUN: vc4-opt --lower-gpu-to-vc4 %s | vc4-translate --mlir-to-vc4-launcher-h | FileCheck %s --check-prefix=H
// RUN: vc4-opt --lower-gpu-to-vc4 %s | vc4-translate --mlir-to-vc4-launcher-c | FileCheck %s --check-prefix=C

gpu.module @kernels {
  gpu.func @blend_linear(%lhs: memref<?xf32>, %rhs: memref<?xf32>,
                         %out: memref<?xf32>, %alpha: f32,
                         %beta: f32, %extent: index) kernel {
    %lane = gpu.global_id x
    %inBounds = arith.cmpi ult, %lane, %extent : index
    scf.if %inBounds {
      %lhsv = memref.load %lhs[%lane] : memref<?xf32>
      %rhsv = memref.load %rhs[%lane] : memref<?xf32>
      %lhsScaled = arith.mulf %alpha, %lhsv : f32
      %rhsScaled = arith.mulf %beta, %rhsv : f32
      %sum = arith.addf %lhsScaled, %rhsScaled : f32
      memref.store %sum, %out[%lane] : memref<?xf32>
    }
    gpu.return
  }
}

// H: struct vc4_runtime;
// H: int blend_linear_launch(struct vc4_runtime *rt, float * arg0, float * arg1, float * arg2, float arg3, float arg4, uint32_t arg5);
// H-NOT: qpu_id
// H-NOT: num_qpus
// H-NOT: uniform

// C: struct blend_linear_launch_state {
// C: memcpy((void *)blend_linear_code_ptr(state), blend_linearshader, blend_linear_code_bytes());
// C-DAG: float *gpuArg0 = blend_linear_arg0_ptr(state, arg5);
// C-DAG: memcpy(gpuArg0, arg0, (size_t)arg5 * sizeof(float));
// C-DAG: uint32_t gpuArg0Addr = GPU_BASE + (uint32_t)gpuArg0;
// C-DAG: float *gpuArg1 = blend_linear_arg1_ptr(state, arg5);
// C-DAG: memcpy(gpuArg1, arg1, (size_t)arg5 * sizeof(float));
// C-DAG: uint32_t gpuArg1Addr = GPU_BASE + (uint32_t)gpuArg1;
// C-DAG: float *gpuArg2 = blend_linear_arg2_ptr(state, arg5);
// C-DAG: memcpy(gpuArg2, arg2, (size_t)arg5 * sizeof(float));
// C-DAG: uint32_t gpuArg2Addr = GPU_BASE + (uint32_t)gpuArg2;
// C: state->unif[qpu][0] = gpuArg0Addr;
// C: state->unif[qpu][1] = gpuArg1Addr;
// C: state->unif[qpu][2] = gpuArg2Addr;
// C: state->unif[qpu][3] = vc4_pack_f32(arg3);
// C: state->unif[qpu][4] = vc4_pack_f32(arg4);
// C: state->unif[qpu][5] = arg5;
// C: state->unif[qpu][6] = qpu;
// C: state->unif[qpu][7] = activeQpus;
// C: memcpy(arg2, gpuArg2, (size_t)arg5 * sizeof(float));
