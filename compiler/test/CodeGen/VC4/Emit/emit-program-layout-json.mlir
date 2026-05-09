// RUN: rm -rf %t.bundle
// RUN: vc4-codegen %s --emit-bundle %t.bundle
// RUN: test -f %t.bundle/layout.json
// RUN: test ! -f %t.bundle/kernel.qasm
// RUN: python3 %S/../Support/check_m2_layout.py %t.bundle/layout.json --alignment 8 --require-heap --min-heap-bytes 4096
// RUN: FileCheck %s --check-prefix=LAYOUT --input-file=%t.bundle/layout.json
// RUN: FileCheck %s --check-prefix=SOURCE --input-file=%t.bundle/kernel_launch.c

// LAYOUT: "schema_version": 1
// LAYOUT: "kind": "vc4-program-layout"
// LAYOUT: "program_name": "program_layout_json"
// LAYOUT: "kernel_count": 2
// LAYOUT-DAG: "name": "kernel_descriptor_table"
// LAYOUT-DAG: "name": "first_layout_launch.code"
// LAYOUT-DAG: "name": "first_layout_launch.uniforms"
// LAYOUT-DAG: "name": "first_layout_launch.unif_ptrs"
// LAYOUT-DAG: "name": "second_layout_launch.code"
// LAYOUT-DAG: "name": "second_layout_launch.uniforms"
// LAYOUT-DAG: "name": "second_layout_launch.unif_ptrs"
// LAYOUT-DAG: "name": "heap"
// LAYOUT-DAG: "descriptor": {"offset":
// LAYOUT-DAG: "uniform_stream": {"offset":
// LAYOUT-DAG: "uniform_pointer_array": {"offset":

// SOURCE: VC4_RUNTIME_LAYOUT
// SOURCE-DAG: struct vc4_codegen_kernel_desc
// SOURCE-DAG: kernel_descs[VC4_CODEGEN_PROGRAM_KERNELS]
// SOURCE-DAG: memcpy((void *)state->kernel_0_code, first_layout_shader
// SOURCE-DAG: memcpy((void *)state->kernel_1_code, second_layout_shader
// SOURCE-DAG: kernel_0_unif_ptr
// SOURCE-DAG: kernel_1_unif_ptr

vc4.module @program_layout_json {
  vc4.func @first_layout_kernel() attributes {
    domain = #vc4.execution_domain<qpu>,
    form = #vc4.function_form<scheduled>,
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.resource" = {schedule_mode = "independent_vector", uses_barrier = false, uses_shared_vpm = false},
    "vc4.launch_abi" = {
      public_name = "first_layout_launch",
      code_symbol = "first_layout_shader",
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

  vc4.func @second_layout_kernel() attributes {
    domain = #vc4.execution_domain<qpu>,
    form = #vc4.function_form<scheduled>,
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.resource" = {schedule_mode = "independent_vector", uses_barrier = false, uses_shared_vpm = false},
    "vc4.launch_abi" = {
      public_name = "second_layout_launch",
      code_symbol = "second_layout_shader",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 3 : i32,
      args = [
        {name = "scale", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 0 : i32}
      ],
      builtins = [
        {name = "qpu_id", kind = #vc4.builtin_kind<qpu_num>, materialization = "uniform_suffix", uniform_index = 1 : i32},
        {name = "num_qpus", kind = #vc4.builtin_kind<num_qpus>, materialization = "uniform_suffix", uniform_index = 2 : i32}
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
