// RUN: rm -rf %t.bundle
// RUN: vc4-codegen %S/../Hardware/Run/saxpy_full/input.mlir --emit-bundle %t.bundle
// RUN: test -f %t.bundle/kernels/saxpy_full.qasm
// RUN: test ! -f %t.bundle/kernel.qasm
// RUN: test -f %t.bundle/manifest.json
// RUN: FileCheck %s --check-prefix=QASM --input-file=%t.bundle/kernels/saxpy_full.qasm --implicit-check-not='ra32' --implicit-check-not='rb32' --implicit-check-not='ra38' --implicit-check-not='rb38' --implicit-check-not='ra48' --implicit-check-not='rb48' --implicit-check-not='ra49' --implicit-check-not='rb49' --implicit-check-not='ra50' --implicit-check-not='rb50' --implicit-check-not='ra56' --implicit-check-not='rb56'
// RUN: FileCheck %s --check-prefix=HEADER --input-file=%t.bundle/kernel_launch.h --implicit-check-not='"mailbox.h"' --implicit-check-not='typedef uint32_t vc4_deviceptr_t'
// RUN: FileCheck %s --check-prefix=SOURCE --input-file=%t.bundle/kernel_launch.c --implicit-check-not='struct vc4_program {' --implicit-check-not='struct vc4_codegen_heap_block' --implicit-check-not='int vc4Malloc' --implicit-check-not='int vc4Memcpy' --implicit-check-not='qpu_enable(' --implicit-check-not='mem_alloc(' --implicit-check-not='mem_lock(' --implicit-check-not='mem_free(' --implicit-check-not='gpu_fft_base_exec_direct' --implicit-check-not='V3D_' --implicit-check-not='PUT32(' --implicit-check-not='GET32(' --implicit-check-not='VC4_HEAP_STATS' --implicit-check-not='VC4_SCHEDULE_COOPERATIVE_BLOCK VC4_SCHEDULE_INDEPENDENT_VECTOR'
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

// HEADER-DAG: #include "vc4_runtime.h"
// HEADER-DAG: int vc4_program_create(struct vc4_program **out, uint32_t requested_bytes);
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
// SOURCE: static const struct vc4_kernel_image vc4_codegen_kernels[] = {
// SOURCE: .name = "saxpy_full"
// SOURCE: .code = saxpy_full_shader
// SOURCE: .schedule_mode = VC4_SCHEDULE_INDEPENDENT_VECTOR
// SOURCE: static const struct vc4_module_image vc4_codegen_module = {
// SOURCE: return vc4ProgramCreateFromImage(out, &vc4_codegen_module, requested_bytes);
// SOURCE-LABEL: static int saxpy_full_pack_uniforms(void *opaque, const struct vc4_launch_request_info *requestInfo, uint32_t *uniformWords, uint32_t uniformWordsPerRequest) {
// SOURCE: uniformWords[0] = (uint32_t)ctx->x; /* arg x */
// SOURCE: uniformWords[1] = (uint32_t)ctx->y; /* arg y */
// SOURCE: uniformWords[2] = vc4_codegen_pack_f32(ctx->alpha); /* arg alpha */
// SOURCE: uniformWords[3] = (uint32_t)ctx->n; /* arg n */
// SOURCE: uniformWords[4] = requestInfo->logical_request; /* builtin logical_request */
// SOURCE: uniformWords[5] = requestInfo->total_requests; /* builtin total_requests */
// SOURCE-LABEL: int saxpy_full_launch(struct vc4_program *program, vc4_dim3 grid, vc4_dim3 block, vc4_deviceptr_t x, vc4_deviceptr_t y, float alpha, uint32_t n) {
// SOURCE: vc4DeviceRangeIsAllocated(program, x,
// SOURCE: vc4DeviceRangeIsAllocated(program, y,
// SOURCE: return vc4LaunchKernel(program, 0u, grid, block, totalRequests, warpsPerBlock, saxpy_full_pack_uniforms, &ctx);
