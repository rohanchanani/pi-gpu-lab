// RUN: rm -rf %t && mkdir -p %t
// RUN: vc4-opt %S/../Hardware/Run/vector_store_smoke_vc4kernel/input.mlir --verify-vc4kernel -o %t/verified.vc4kernel.mlir
// RUN: FileCheck %s --check-prefix=VERIFY --input-file=%t/verified.vc4kernel.mlir
// RUN: vc4-opt %t/verified.vc4kernel.mlir --convert-vc4kernel-to-ssavc4 -o %t/lowered.ssavc4.mlir
// RUN: FileCheck %s --check-prefix=SSAVC4 --input-file=%t/lowered.ssavc4.mlir
// RUN: vc4-opt %t/lowered.ssavc4.mlir --convert-ssavc4-to-vc4 --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o %t/scheduled.vc4.mlir
// RUN: FileCheck %s --check-prefix=VC4 --input-file=%t/scheduled.vc4.mlir
// RUN: vc4-codegen %t/scheduled.vc4.mlir --emit-bundle %t/bundle
// RUN: test -f %t/bundle/kernel_launch.c
// RUN: test -f %t/bundle/kernel_launch.h
// RUN: test -f %t/bundle/manifest.json
// RUN: test -f %t/bundle/kernels/vector_store_smoke_vc4kernel.qasm

// VERIFY: vc4kernel.kernel @vector_store_smoke_vc4kernel
// VERIFY-NOT: ssavc4.

// SSAVC4-LABEL: ssavc4.func @vector_store_smoke_vc4kernel
// SSAVC4-SAME: vc4.launch_abi
// SSAVC4-SAME: vc4.resource
// SSAVC4-SAME: compiler_vpm_staging_rows_per_warp = 1 : i32
// SSAVC4-SAME: requires_vpm_base_row_builtin = true
// SSAVC4: ssavc4.vdw.store
// SSAVC4-NOT: vc4kernel.
// SSAVC4-NOT: vc4.qpu

// VC4: vc4.module
// VC4: vc4.qpu.vpmvcd_setup
// VC4: vc4.qpu.vpmvcd_wait
// VC4-NOT: ssavc4.
// VC4-NOT: vc4kernel.
