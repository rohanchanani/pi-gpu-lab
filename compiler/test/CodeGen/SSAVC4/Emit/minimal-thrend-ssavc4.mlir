// RUN: rm -rf %t.lowered.mlir %t.bundle
// RUN: vc4-opt %s --convert-ssavc4-to-vc4 -o %t.lowered.mlir
// RUN: vc4-opt %t.lowered.mlir --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o /dev/null
// RUN: vc4-codegen %t.lowered.mlir --emit-bundle %t.bundle
// RUN: test -f %t.bundle/kernels/minimal_thrend_ssavc4.qasm
// RUN: test -f %t.bundle/kernel_launch.c
// RUN: test -f %t.bundle/kernel_launch.h
// RUN: test -f %t.bundle/manifest.json
// RUN: FileCheck %s --check-prefix=QASM --input-file=%t.bundle/kernels/minimal_thrend_ssavc4.qasm
// RUN: FileCheck %s --check-prefix=MANIFEST --input-file=%t.bundle/manifest.json

// QASM: thrend
// QASM-NEXT: nop
// QASM-NEXT: nop

// MANIFEST: "schema_version": 2
// MANIFEST: "public_name": "minimal_thrend_ssavc4"
// MANIFEST: "code_symbol": "minimal_thrend_ssavc4_shader"
// MANIFEST: "scheduled_sink_ops": 3
ssavc4.module @minimal_thrend_ssavc4_codegen {
  ssavc4.func @minimal_thrend_ssavc4_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "minimal_thrend_ssavc4",
      code_symbol = "minimal_thrend_ssavc4_shader",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 2 : i32,
      args = [],
      builtins = [
        {name = "logical_request", kind = #vc4.builtin_kind<logical_request>, materialization = "uniform_suffix", uniform_index = 0 : i32},
        {name = "total_requests", kind = #vc4.builtin_kind<total_requests>, materialization = "uniform_suffix", uniform_index = 1 : i32}
      ]
    }
  } {
    ssavc4.thread_end
  }
}
