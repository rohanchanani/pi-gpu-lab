// RUN: rm -rf %t.bundle
// RUN: vc4-codegen %S/../Hardware/Run/saxpy_full/input.mlir --emit-bundle %t.bundle
// RUN: test -f %t.bundle/kernels/saxpy_full_launch.qasm
// RUN: test ! -f %t.bundle/kernel.qasm
// RUN: test -f %t.bundle/manifest.json
// RUN: FileCheck %s --check-prefix=QASM --input-file=%t.bundle/kernels/saxpy_full_launch.qasm --implicit-check-not='ra32' --implicit-check-not='rb32' --implicit-check-not='ra38' --implicit-check-not='rb38' --implicit-check-not='ra48' --implicit-check-not='rb48' --implicit-check-not='ra49' --implicit-check-not='rb49' --implicit-check-not='ra50' --implicit-check-not='rb50' --implicit-check-not='ra56' --implicit-check-not='rb56'
// RUN: FileCheck %s --check-prefix=HEADER --input-file=%t.bundle/kernel_launch.h
// RUN: FileCheck %s --check-prefix=SOURCE --input-file=%t.bundle/kernel_launch.c
// RUN: FileCheck %s --check-prefix=MANIFEST --input-file=%t.bundle/manifest.json

// MANIFEST: "schema_version": 2
// MANIFEST: "program_name": "saxpy_full"
// MANIFEST: "kernels": [
// MANIFEST: "kernel_id": 0
// MANIFEST: "symbol_name": "saxpy_full_kernel"
// MANIFEST: "public_name": "saxpy_full_launch"
// MANIFEST: "qasm_path": "kernels/saxpy_full_launch.qasm"
// MANIFEST: "code_symbol": "saxpy_full_launch_shader"

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

// HEADER-DAG: int saxpy_full_prepare(struct vc4_runtime *rt, uint32_t max_n);
// HEADER-DAG: int saxpy_full_launch(struct vc4_runtime *rt, const float *x, float *y, float alpha, uint32_t n);
// HEADER-DAG: uint32_t saxpy_full_runtime_allocations(void);
// HEADER-DAG: uint32_t saxpy_full_runtime_launches(void);
// HEADER-DAG: uint32_t saxpy_full_runtime_capacity(void);

// SOURCE-DAG: uint32_t code[sizeof(kernelshader) / sizeof(uint32_t)];
// SOURCE-DAG: int saxpy_full_prepare(struct vc4_runtime *rt, uint32_t max_n)
// SOURCE-DAG: int saxpy_full_launch(struct vc4_runtime *rt, const float *x, float *y, float alpha, uint32_t n)
