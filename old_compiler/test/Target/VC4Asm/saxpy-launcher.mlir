// RUN: vc4-opt --lower-gpu-to-vc4 %s | vc4-translate --mlir-to-vc4-launcher-h | FileCheck %s --check-prefix=H
// RUN: vc4-opt --lower-gpu-to-vc4 %s | vc4-translate --mlir-to-vc4-launcher-c | FileCheck %s --check-prefix=C

gpu.module @kernels {
  gpu.func @saxpy(%x: memref<?xf32>, %y: memref<?xf32>, %a: f32, %n: index) kernel {
    %gid = gpu.global_id x
    %inBounds = arith.cmpi ult, %gid, %n : index
    scf.if %inBounds {
      %xval = memref.load %x[%gid] : memref<?xf32>
      %yval = memref.load %y[%gid] : memref<?xf32>
      %mul = arith.mulf %a, %xval : f32
      %sum = arith.addf %mul, %yval : f32
      memref.store %sum, %y[%gid] : memref<?xf32>
    }
    gpu.return
  }
}

// H: int saxpy_launch(struct vc4_runtime *rt, float * arg0, float * arg1, float arg2, uint32_t arg3);
// H-NOT: qpu_id
// H-NOT: num_qpus

// C: struct saxpy_launch_state {
// C: uint32_t unif[VC4_RUNTIME_MAX_QPUS][NUM_UNIFS];
// C: uint32_t unif_ptr[VC4_RUNTIME_MAX_QPUS];
// C: uint32_t handle;
// C: uint32_t payload[];
// C: size_t allocSize = saxpy_launch_state_size(arg3);
// C: uint32_t handle = mem_alloc((uint32_t)allocSize, 4096, GPU_MEM_FLG);
// C: uint32_t vc = mem_lock(handle);
// C: memcpy((void *)saxpy_code_ptr(state), saxpyshader, saxpy_code_bytes());
// C-DAG: float *gpuArg0 = saxpy_arg0_ptr(state, arg3);
// C-DAG: memcpy(gpuArg0, arg0, (size_t)arg3 * sizeof(float));
// C-DAG: uint32_t gpuArg0Addr = GPU_BASE + (uint32_t)gpuArg0;
// C-DAG: float *gpuArg1 = saxpy_arg1_ptr(state, arg3);
// C-DAG: memcpy(gpuArg1, arg1, (size_t)arg3 * sizeof(float));
// C-DAG: uint32_t gpuArg1Addr = GPU_BASE + (uint32_t)gpuArg1;
// C: state->unif[qpu][0] = gpuArg0Addr;
// C: state->unif[qpu][1] = gpuArg1Addr;
// C: state->unif[qpu][2] = vc4_pack_f32(arg2);
// C: state->unif[qpu][3] = arg3;
// C: state->unif[qpu][4] = qpu;
// C: state->unif[qpu][5] = activeQpus;
// C: state->unif_ptr[qpu] = GPU_BASE + (uint32_t)&state->unif[qpu];
// C: gpu_fft_base_exec_direct((uint32_t)saxpy_code_ptr(state), (uint32_t *)state->unif_ptr, activeQpus);
// C: memcpy(arg1, gpuArg1, (size_t)arg3 * sizeof(float));
// C: mem_unlock(handle);
// C: mem_free(handle);
