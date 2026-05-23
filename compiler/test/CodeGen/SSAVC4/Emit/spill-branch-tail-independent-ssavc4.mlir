// RUN: rm -rf %t.lowered.mlir %t.bundle
// RUN: vc4-opt %S/../../../Conversion/SSAVC4ToVC4/spill-branch-tail-independent.mlir --convert-ssavc4-to-vc4 -o %t.lowered.mlir
// RUN: FileCheck %s --check-prefix=LOWERED --input-file=%t.lowered.mlir
// RUN: vc4-opt %t.lowered.mlir --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o /dev/null
// RUN: vc4-codegen %t.lowered.mlir --emit-bundle %t.bundle
// RUN: test -f %t.bundle/kernels/spill_branch_tail_independent.qasm
// RUN: FileCheck %s --check-prefix=QASM --input-file=%t.bundle/kernels/spill_branch_tail_independent.qasm
// RUN: FileCheck %s --check-prefix=SOURCE --input-file=%t.bundle/kernel_launch.c
// RUN: FileCheck %s --check-prefix=MANIFEST --input-file=%t.bundle/manifest.json

// LOWERED-LABEL: vc4.func @spill_branch_tail_independent_kernel
// LOWERED-SAME: spill_frame_bytes = {{[1-9][0-9]*}} : i32
// LOWERED-SAME: spill_frame_base
// LOWERED: vc4.qpu.vpmvcd_addr
// LOWERED: vc4.qpu.branch
// LOWERED: sig = #vc4.qpu_signal<ldtmu0>
// QASM: brr.anyc
// QASM-SAME: delay_slots=3
// QASM: brr
// QASM-SAME: delay_slots=3
// QASM: ldtmu0
// QASM: thrend
// SOURCE: uniformWords[2] = requestInfo->spill_frame_base; /* builtin spill_frame_base */
// SOURCE-LABEL: int spill_branch_tail_independent_launch(struct vc4_program *program, vc4_dim3 grid, vc4_dim3 block, vc4_deviceptr_t out)
// MANIFEST: "spill_frame_bytes": {{[1-9][0-9]*}}
// MANIFEST: "spill_frame_base"
