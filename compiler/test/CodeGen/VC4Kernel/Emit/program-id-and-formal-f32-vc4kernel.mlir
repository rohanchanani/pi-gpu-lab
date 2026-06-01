// RUN: rm -rf %t && mkdir -p %t
// RUN: vc4-opt %S/../Hardware/Run/program_id_writeback_vc4kernel/input.mlir --verify-vc4kernel -o %t/program_id.verified.vc4kernel.mlir
// RUN: FileCheck %s --check-prefix=PID-VERIFY --input-file=%t/program_id.verified.vc4kernel.mlir
// RUN: vc4-opt %t/program_id.verified.vc4kernel.mlir --convert-vc4kernel-to-ssavc4 -o %t/program_id.ssavc4.mlir
// RUN: FileCheck %s --check-prefix=PID-SSAVC4 --input-file=%t/program_id.ssavc4.mlir
// RUN: vc4-opt %t/program_id.ssavc4.mlir --convert-ssavc4-to-vc4 --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o %t/program_id.vc4.mlir
// RUN: FileCheck %s --check-prefix=PID-VC4 --input-file=%t/program_id.vc4.mlir
// RUN: vc4-codegen %t/program_id.vc4.mlir --emit-bundle %t/program_id.bundle
// RUN: test -f %t/program_id.bundle/kernel_launch.c
// RUN: test -f %t/program_id.bundle/kernels/program_id_writeback_vc4kernel.qasm
// RUN: vc4-opt %S/../Hardware/Run/formal_f32_splat_vc4kernel/input.mlir --verify-vc4kernel -o %t/formal_f32.verified.vc4kernel.mlir
// RUN: FileCheck %s --check-prefix=F32-VERIFY --input-file=%t/formal_f32.verified.vc4kernel.mlir
// RUN: vc4-opt %t/formal_f32.verified.vc4kernel.mlir --convert-vc4kernel-to-ssavc4 -o %t/formal_f32.ssavc4.mlir
// RUN: FileCheck %s --check-prefix=F32-SSAVC4 --input-file=%t/formal_f32.ssavc4.mlir
// RUN: vc4-opt %t/formal_f32.ssavc4.mlir --convert-ssavc4-to-vc4 --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o %t/formal_f32.vc4.mlir
// RUN: FileCheck %s --check-prefix=F32-VC4 --input-file=%t/formal_f32.vc4.mlir
// RUN: vc4-codegen %t/formal_f32.vc4.mlir --emit-bundle %t/formal_f32.bundle
// RUN: test -f %t/formal_f32.bundle/kernel_launch.c
// RUN: test -f %t/formal_f32.bundle/kernels/formal_f32_splat_vc4kernel.qasm

// PID-VERIFY: vc4kernel.kernel @program_id_writeback_vc4kernel
// PID-VERIFY-NOT: ssavc4.

// PID-SSAVC4-LABEL: ssavc4.func @program_id_writeback_vc4kernel
// PID-SSAVC4-SAME: {direction = "out", elem_type = "i32", kind = "buffer", name = "out", uniform_index = 0 : i32}
// PID-SSAVC4-SAME: {kind = #vc4.builtin_kind<logical_request>, materialization = "uniform_suffix", name = "logical_request", uniform_index = 1 : i32}
// PID-SSAVC4-SAME: {kind = #vc4.builtin_kind<total_requests>, materialization = "uniform_suffix", name = "total_requests", uniform_index = 2 : i32}
// PID-SSAVC4-SAME: {kind = #vc4.builtin_kind<vpm_base_row>, materialization = "uniform_suffix", name = "vpm_base_row", uniform_index = 3 : i32}
// PID-SSAVC4-SAME: uniform_words_per_qpu = 4 : i32
// PID-SSAVC4: ssavc4.uniform.read 1 : i32
// PID-SSAVC4: ssavc4.element_number : vector<16xi32>
// PID-SSAVC4: ssavc4.vdw.store
// PID-SSAVC4-NOT: QPU_NUMBER
// PID-SSAVC4-NOT: vc4kernel.

// PID-VC4: vc4.module
// PID-VC4-NOT: ssavc4.
// PID-VC4-NOT: vc4kernel.

// F32-VERIFY: vc4kernel.kernel @formal_f32_splat_vc4kernel
// F32-VERIFY-NOT: ssavc4.

// F32-SSAVC4-LABEL: ssavc4.func @formal_f32_splat_vc4kernel
// F32-SSAVC4-SAME: {direction = "out", elem_type = "f32", kind = "buffer", name = "out", uniform_index = 0 : i32}
// F32-SSAVC4-SAME: {direction = "by_value", kind = "scalar", name = "alpha", type = "f32", uniform_index = 1 : i32}
// F32-SSAVC4-SAME: {direction = "by_value", kind = "scalar", name = "offset_elems", type = "i32", uniform_index = 2 : i32}
// F32-SSAVC4-SAME: {kind = #vc4.builtin_kind<vpm_base_row>, materialization = "uniform_suffix", name = "vpm_base_row", uniform_index = 3 : i32}
// F32-SSAVC4-SAME: uniform_words_per_qpu = 4 : i32
// F32-SSAVC4: ssavc4.uniform.read 1 : f32
// F32-SSAVC4: ssavc4.splat {{.*}} : f32 -> vector<16xf32>
// F32-SSAVC4: ssavc4.vdw.store
// F32-SSAVC4-NOT: vc4kernel.

// F32-VC4: vc4.module
// F32-VC4-NOT: ssavc4.
// F32-VC4-NOT: vc4kernel.
