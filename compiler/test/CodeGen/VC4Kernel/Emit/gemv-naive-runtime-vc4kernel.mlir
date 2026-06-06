// RUN: rm -rf %t && mkdir -p %t
// RUN: vc4-opt %S/../Hardware/Run/gemv_naive_vc4kernel/input.mlir --verify-vc4kernel -o %t/gemv.verified.vc4kernel.mlir
// RUN: FileCheck %s --check-prefix=VERIFY --input-file=%t/gemv.verified.vc4kernel.mlir
// RUN: vc4-opt %t/gemv.verified.vc4kernel.mlir --convert-vc4kernel-to-ssavc4 -o %t/gemv.ssavc4.mlir
// RUN: FileCheck %s --check-prefix=SSAVC4 --input-file=%t/gemv.ssavc4.mlir
// RUN: vc4-opt %t/gemv.ssavc4.mlir --convert-ssavc4-to-vc4 --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o %t/gemv.vc4.mlir
// RUN: FileCheck %s --check-prefix=VC4 --input-file=%t/gemv.vc4.mlir
// RUN: vc4-codegen %t/gemv.vc4.mlir --emit-bundle %t/gemv.bundle
// RUN: test -f %t/gemv.bundle/kernel_launch.c
// RUN: test -f %t/gemv.bundle/manifest.json
// RUN: test -f %t/gemv.bundle/kernels/gemv_naive_vc4kernel.qasm
// RUN: FileCheck %s --check-prefix=SOURCE --input-file=%t/gemv.bundle/kernel_launch.c
// RUN: FileCheck %s --check-prefix=MANIFEST --input-file=%t/gemv.bundle/manifest.json

// VERIFY: vc4kernel.kernel @gemv_naive_vc4kernel
// VERIFY-SAME: {direction = "by_value", kind = "scalar", name = "lda", type = "i32"}
// VERIFY: vc4kernel.fragment_alu.mul
// VERIFY: vc4kernel.tmu_load_fragment
// VERIFY: vc4kernel.vdw_store_fragment
// VERIFY-NOT: ssavc4.

// SSAVC4-LABEL: ssavc4.func @gemv_naive_vc4kernel
// SSAVC4-SAME: {direction = "in", elem_type = "f32", kind = "buffer", name = "a", uniform_index = 0 : i32}
// SSAVC4-SAME: {direction = "in", elem_type = "f32", kind = "buffer", name = "x", uniform_index = 1 : i32}
// SSAVC4-SAME: {direction = "inout", elem_type = "f32", kind = "buffer", name = "y", uniform_index = 2 : i32}
// SSAVC4-SAME: {direction = "by_value", kind = "scalar", name = "m", type = "i32", uniform_index = 3 : i32}
// SSAVC4-SAME: {direction = "by_value", kind = "scalar", name = "n", type = "i32", uniform_index = 4 : i32}
// SSAVC4-SAME: {direction = "by_value", kind = "scalar", name = "lda", type = "i32", uniform_index = 5 : i32}
// SSAVC4: ssavc4.uniform.read 5 : i32
// SSAVC4-DAG: ssavc4.tmu.request
// SSAVC4-DAG: ssavc4.alu.mul
// SSAVC4-DAG: ssavc4.vdw.store
// SSAVC4-NOT: vc4kernel.

// VC4: vc4.module
// VC4-NOT: ssavc4.
// VC4-NOT: vc4kernel.

// SOURCE: .name = "gemv_naive_vc4kernel"
// SOURCE: .uses_tmu = 1u
// SOURCE: .uses_vdw = 1u
// SOURCE: [5] arg lda

// MANIFEST: "public_name": "gemv_naive_vc4kernel"
// MANIFEST: "uniform_words_per_request":
// MANIFEST: "uses_tmu": true
// MANIFEST: "uses_vdw": true
