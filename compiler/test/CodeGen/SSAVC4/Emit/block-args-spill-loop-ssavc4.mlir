// RUN: rm -rf %t.lowered.mlir %t.bundle
// RUN: vc4-opt %S/Inputs/block_args_spill_smoke_ssavc4_input.mlir.inc --convert-ssavc4-to-vc4 -o %t.lowered.mlir
// RUN: FileCheck %s --check-prefix=LOWERED --input-file=%t.lowered.mlir
// RUN: vc4-opt %t.lowered.mlir --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o /dev/null
// RUN: vc4-codegen %t.lowered.mlir --emit-bundle %t.bundle
// RUN: test -f %t.bundle/kernels/block_args_spill_smoke_ssavc4.qasm
// RUN: FileCheck %s --check-prefix=QASM --input-file=%t.bundle/kernels/block_args_spill_smoke_ssavc4.qasm
// RUN: FileCheck %s --check-prefix=SOURCE --input-file=%t.bundle/kernel_launch.c
// RUN: FileCheck %s --check-prefix=MANIFEST --input-file=%t.bundle/manifest.json

// LOWERED-LABEL: vc4.func @block_args_spill_smoke_ssavc4_kernel
// LOWERED-SAME: spill_frame_bytes = {{[1-9][0-9]*}} : i32
// LOWERED-SAME: spill_frame_base
// LOWERED-SAME: uses_vdr = true
// LOWERED-SAME: uses_vpm_qpu_read = true
// LOWERED-NOT: ssavc4.phi
// LOWERED-DAG: vc4.qpu.branch attributes {{.*}}immediate = -{{[0-9]+}} : i32
// LOWERED-DAG: vc4.qpu.vpmvcd_addr
// LOWERED-DAG: vc4.qpu.vpmvcd_setup
// LOWERED-DAG: vc4.qpu.vpm
// LOWERED-NOT: sig = #vc4.qpu_signal<ldtmu0>
// QASM: brr.any
// QASM-SAME: delay_slots=3
// QASM: brr
// QASM-SAME: delay_slots=3
// QASM: vr_setup
// QASM: mov {{r[a-b][0-9]+}}, vpm
// QASM-NOT: ldtmu0
// SOURCE: uniformWords[0] = requestInfo->spill_frame_base; /* builtin spill_frame_base */
// SOURCE: uniformWords[1] = requestInfo->vpm_base_row + KERNEL_0_USER_VPM_ROWS_PER_BLOCK + KERNEL_0_COMPILER_VPM_STAGING_ROWS_PER_BLOCK + KERNEL_0_WARPS_PER_BLOCK * KERNEL_0_COMPILER_VPM_STAGING_ROWS_PER_WARP; /* builtin spill_vpm_row */
// SOURCE-LABEL: int block_args_spill_smoke_ssavc4_launch(struct vc4_program *program, vc4_dim3 grid, vc4_dim3 block
// MANIFEST: "spill_frame_bytes": {{[1-9][0-9]*}}
// MANIFEST: "spill_frame_base"
