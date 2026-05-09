// RUN: rm -rf %t.bundle
// RUN: vc4-codegen %s --emit-bundle %t.bundle
// RUN: test -f %t.bundle/kernel_launch.c
// RUN: test -f %t.bundle/kernel_launch.h
// RUN: test -f %t.bundle/manifest.json
// RUN: test ! -f %t.bundle/kernel.qasm
// RUN: test -f %t.bundle/kernels/single_launch.qasm
// RUN: FileCheck %s --input-file=%t.bundle/manifest.json

// CHECK: "schema_version": 2
// CHECK: "kind": "vc4-codegen-artifact-bundle"
// CHECK: "program_name": "manifest_v2_single"
// CHECK: "target": {
// CHECK: "max_active_qpus": 12
// CHECK: "kernels": [
// CHECK: "kernel_id": 0
// CHECK: "symbol_name": "single_kernel"
// CHECK: "public_name": "single_launch"
// CHECK: "qasm_path": "kernels/single_launch.qasm"
// CHECK: "code_symbol": "single_launch_shader"
// CHECK: "uniform_words_per_request": 2
// CHECK: "tail_policy": "exact_multiple"
// CHECK: "schedule_mode": "independent_vector"
// CHECK: "uses_barrier": false

vc4.module @manifest_v2_single {
  vc4.func @single_kernel() attributes {
    domain = #vc4.execution_domain<qpu>,
    form = #vc4.function_form<scheduled>,
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "single_launch",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 2 : i32,
      args = [],
      builtins = [
        {name = "qpu_id", kind = #vc4.builtin_kind<qpu_num>, materialization = "uniform_suffix", uniform_index = 0 : i32},
        {name = "num_qpus", kind = #vc4.builtin_kind<num_qpus>, materialization = "uniform_suffix", uniform_index = 1 : i32}
      ]
    }
  } {
    vc4.qpu.bundle {
      sig = #vc4.qpu_signal<thrend>,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      waddr_add = 32 : i32,
      waddr_mul = 33 : i32,
      op_add = #vc4.add_opcode<nop>,
      op_mul = #vc4.mul_opcode<nop>,
      raddr_a = 0 : i32,
      raddr_b = 1 : i32,
      add_a = #vc4.qpu_mux<a>,
      add_b = #vc4.qpu_mux<b>,
      mul_a = #vc4.qpu_mux<r0>,
      mul_b = #vc4.qpu_mux<r1>
    }

    vc4.qpu.bundle {
      sig = #vc4.qpu_signal<none>,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      waddr_add = 32 : i32,
      waddr_mul = 33 : i32,
      op_add = #vc4.add_opcode<nop>,
      op_mul = #vc4.mul_opcode<nop>,
      raddr_a = 0 : i32,
      raddr_b = 1 : i32,
      add_a = #vc4.qpu_mux<a>,
      add_b = #vc4.qpu_mux<b>,
      mul_a = #vc4.qpu_mux<r0>,
      mul_b = #vc4.qpu_mux<r1>
    }

    vc4.qpu.bundle {
      sig = #vc4.qpu_signal<none>,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      waddr_add = 32 : i32,
      waddr_mul = 33 : i32,
      op_add = #vc4.add_opcode<nop>,
      op_mul = #vc4.mul_opcode<nop>,
      raddr_a = 0 : i32,
      raddr_b = 1 : i32,
      add_a = #vc4.qpu_mux<a>,
      add_b = #vc4.qpu_mux<b>,
      mul_a = #vc4.qpu_mux<r0>,
      mul_b = #vc4.qpu_mux<r1>
    }
  }
}
