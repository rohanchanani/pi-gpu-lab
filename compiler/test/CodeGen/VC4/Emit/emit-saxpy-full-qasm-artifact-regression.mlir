// RUN: rm -rf %t.bundle
// RUN: vc4-codegen %S/../Hardware/Run/saxpy_full/input.mlir --emit-bundle %t.bundle
// RUN: test -f %t.bundle/kernels/saxpy_full.qasm
// RUN: test ! -f %t.bundle/kernel.qasm
// RUN: test -f %t.bundle/manifest.json
// RUN: FileCheck %s --check-prefix=QASM --input-file=%t.bundle/kernels/saxpy_full.qasm --implicit-check-not='ra32' --implicit-check-not='rb32' --implicit-check-not='ra38' --implicit-check-not='rb38' --implicit-check-not='ra48' --implicit-check-not='rb48' --implicit-check-not='ra49' --implicit-check-not='rb49' --implicit-check-not='ra50' --implicit-check-not='rb50' --implicit-check-not='ra56' --implicit-check-not='rb56'
// RUN: FileCheck %s --check-prefix=HEADER --input-file=%t.bundle/kernel_launch.h
// RUN: FileCheck %s --check-prefix=SOURCE --input-file=%t.bundle/kernel_launch.c
// RUN: FileCheck %s --check-prefix=MANIFEST --input-file=%t.bundle/manifest.json

// MANIFEST: "schema_version": 2
// MANIFEST: "program_name": "saxpy_full"
// MANIFEST: "kernels": [
// MANIFEST: "kernel_id": 0
// MANIFEST: "symbol_name": "saxpy_full_kernel"
// MANIFEST: "public_name": "saxpy_full"
// MANIFEST: "qasm_path": "kernels/saxpy_full.qasm"
// MANIFEST: "code_symbol": "saxpy_full_shader"
// MANIFEST-DAG: {"name": "x", "kind": "buffer", "direction": "in", "elem_type": "f32", "c_type": "vc4_deviceptr_t", "uniform_index": 0}
// MANIFEST-DAG: {"name": "y", "kind": "buffer", "direction": "inout", "elem_type": "f32", "c_type": "vc4_deviceptr_t", "uniform_index": 1}
// MANIFEST-DAG: {"name": "alpha", "kind": "scalar", "direction": "by_value", "type": "f32", "c_type": "float", "uniform_index": 2}
// MANIFEST-DAG: {"name": "n", "kind": "scalar", "direction": "by_value", "type": "u32", "c_type": "uint32_t", "uniform_index": 3}

// This is a non-hardware regression for the QASM spelling surface exercised by
// the saxpy_full hardware fixture.  The scheduled MLIR keeps raw QPU addresses;
// the artifact emitter must lower peripheral addresses to vc4asm symbolic
// names, emit side-effect-only waits, and use labels for resolvable branches.

// QASM-DAG: mov ra0, unif
// QASM-DAG: shl r1, elem_num, 2
// QASM-DAG: add t0s, ra10, r1
// QASM-DAG: add t0s, ra11, r1
// QASM-DAG: add vw_setup, r3, ra12
// QASM-DAG: mov vpm, r2
// QASM-DAG: add vw_setup, r2, r1
// QASM-DAG: mov vw_addr, ra11
// QASM-DAG: read vw_wait
// QASM-DAG: brr.anync -, :vc4_qpu_slot_66
// QASM-DAG: brr.anync -, :vc4_qpu_slot_35
// QASM-DAG: brr -, :vc4_qpu_slot_37
// QASM-DAG: brr.anyc -, :vc4_qpu_slot_21
// QASM-DAG: target_label=vc4_qpu_slot_66
// QASM-DAG: target_label=vc4_qpu_slot_35
// QASM-DAG: target_label=vc4_qpu_slot_37
// QASM-DAG: target_label=vc4_qpu_slot_21

// HEADER-DAG: typedef uint32_t vc4_deviceptr_t;
// HEADER-DAG: typedef struct vc4_dim3
// HEADER-DAG: struct vc4_program;
// HEADER-DAG: int vc4_program_create(struct vc4_program **out, uint32_t requested_bytes);
// HEADER-DAG: int vc4Malloc(struct vc4_program *program, vc4_deviceptr_t *out, uint32_t bytes);
// HEADER-DAG: int vc4MemcpyHtoD(struct vc4_program *program, vc4_deviceptr_t dst, const void *src, uint32_t bytes);
// HEADER-DAG: int vc4MemcpyDtoH(struct vc4_program *program, void *dst, vc4_deviceptr_t src, uint32_t bytes);
// HEADER-DAG: int vc4Free(struct vc4_program *program, vc4_deviceptr_t ptr);
// HEADER-DAG: int saxpy_full_launch(struct vc4_program *program, vc4_dim3 grid, vc4_dim3 block, vc4_deviceptr_t x, vc4_deviceptr_t y, float alpha, uint32_t n);
// HEADER-DAG: uint32_t saxpy_full_runtime_allocations(void);
// HEADER-DAG: uint32_t saxpy_full_runtime_launches(void);
// HEADER-DAG: uint32_t saxpy_full_runtime_capacity(void);
// HEADER-NOT: struct vc4_runtime
// HEADER-NOT: saxpy_full_prepare
// HEADER-NOT: const float *x
// HEADER-NOT: float *y
// HEADER-NOT: uint32_t *uniform

// SOURCE: #include "kernel_launch.h"
// SOURCE-LABEL: int saxpy_full_launch(struct vc4_program *program, vc4_dim3 grid, vc4_dim3 block, vc4_deviceptr_t x, vc4_deviceptr_t y, float alpha, uint32_t n) {
// SOURCE-NOT: vc4Malloc(
// SOURCE-NOT: vc4MemcpyHtoD(
// SOURCE-NOT: vc4MemcpyDtoH(
// SOURCE-NOT: vc4Free(
// SOURCE: [qpu][0] = (uint32_t)x; /* arg x */
// SOURCE: [qpu][1] = (uint32_t)y; /* arg y */
// SOURCE: [qpu][2] = vc4_codegen_pack_f32(alpha); /* arg alpha */
// SOURCE: [qpu][3] = (uint32_t)n; /* arg n */
// SOURCE: [qpu][4] = logicalRequest; /* builtin qpu_id */
// SOURCE: [qpu][5] = totalRequests; /* builtin num_qpus */
// SOURCE: PUT32(V3D_SRQUA,
// SOURCE: PUT32(V3D_SRQPC,
// SOURCE: return 0;
