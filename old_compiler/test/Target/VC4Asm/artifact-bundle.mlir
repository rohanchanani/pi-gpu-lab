// RUN: rm -rf %t && mkdir -p %t
// RUN: vc4-opt --lower-gpu-to-vc4 %s | vc4-translate --emit-vc4-artifacts=%t
// RUN: ls %t | FileCheck %s --check-prefix=FILES
// RUN: test -f %t/share/vc4inc/vc4.qinc
// RUN: FileCheck %s --check-prefix=HDR < %t/kernel_launch.h
// RUN: FileCheck %s --check-prefix=SRC < %t/kernel_launch.c
// RUN: FileCheck %s --check-prefix=QASM < %t/kernel.qasm

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

// FILES: kernel.qasm
// FILES: kernel_launch.c
// FILES: kernel_launch.h

// HDR: int saxpy_launch(struct vc4_runtime *rt, float * arg0, float * arg1, float arg2, uint32_t arg3);
// SRC: struct saxpy_launch_state {
// SRC: state->unif[qpu][4] = qpu;
// SRC: state->unif[qpu][5] = activeQpus;
// SRC: memcpy(arg1, gpuArg1, (size_t)arg3 * sizeof(float));
// QASM: .include "share/vc4inc/vc4.qinc"
// QASM: mov ra0, unif
