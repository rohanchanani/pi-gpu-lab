// RUN: vc4-opt %s --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o /dev/null

// Diagnostic GEMV fixture: exact saxpy_full scheduled body with GEMV public
// ABI.  Physical uniform packing is reordered so the QPU consumes the same
// first five semantic uniforms as saxpy needs:
//   x, y, logical_count(m), qpu_id, num_qpus
// The public C ABI remains gemv_naive_tail(program, grid, block, a, x, y, m, n).
// Unused a/n are packed after the consumed uniforms to satisfy the dense ABI
// contract without perturbing the QPU stream.

module attributes {
  "vc4.hardware_run_test.name" = "gemv_naive_tail",
  "vc4.hardware_run_test.kind" = "hardware-run-reference",
  "vc4.hardware_run_test.reference_kernel" = "reference/gemv_naive_tail.qasm",
  "vc4.hardware_run_test.launch_abi" = {
    public_name = "gemv_naive_tail",
    tail_policy = "tail_safe",
    uniform_words_per_qpu = 7 : i32,
    args = [
      {name = "a", kind = "buffer", direction = "in", elem_type = "f32", uniform_index = 5 : i32},
      {name = "x", kind = "buffer", direction = "in", elem_type = "f32", uniform_index = 0 : i32},
      {name = "y", kind = "buffer", direction = "inout", elem_type = "f32", uniform_index = 1 : i32},
      {name = "m", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 2 : i32},
      {name = "n", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 6 : i32}
    ],
    builtins = [
      {name = "qpu_id", kind = "qpu_num", materialization = "uniform_suffix", uniform_index = 3 : i32},
      {name = "num_qpus", kind = "num_qpus", materialization = "uniform_suffix", uniform_index = 4 : i32}
    ],
    work_distribution = {
      base_element = "qpu_id * 16",
      stride_elements = "num_qpus * 16",
      tail_store = "dynamic_vdw_depth"
    }
  }
} {
  vc4.module @gemv_naive_tail {
    vc4.func @gemv_naive_tail_kernel() attributes {
      domain = #vc4.execution_domain<qpu>,
      form = #vc4.function_form<scheduled>,
      kernel,
      threading = #vc4.threading_mode<single>,
      "vc4.launch_abi" = {
        public_name = "gemv_naive_tail",
        tail_policy = "tail_safe",
        uniform_words_per_qpu = 7 : i32,
        args = [
          {name = "a", kind = "buffer", direction = "in", elem_type = "f32", uniform_index = 5 : i32},
          {name = "x", kind = "buffer", direction = "in", elem_type = "f32", uniform_index = 0 : i32},
          {name = "y", kind = "buffer", direction = "inout", elem_type = "f32", uniform_index = 1 : i32},
          {name = "m", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 2 : i32},
          {name = "n", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 6 : i32}
        ],
        builtins = [
          {name = "qpu_id", kind = #vc4.builtin_kind<qpu_num>, materialization = "uniform_suffix", uniform_index = 3 : i32},
          {name = "num_qpus", kind = #vc4.builtin_kind<num_qpus>, materialization = "uniform_suffix", uniform_index = 4 : i32}
        ]
      }
    } {
      // uniform 0: x base -> exact saxpy input base ra0.
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 0 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 32 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // uniform 1: y base -> exact saxpy output base ra1.
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 1 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 32 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // saxpy alpha = 1.0f as a constant; this occupies the same slot as
      // saxpy_full's alpha uniform read.
      vc4.qpu.ldi <splat32> {value = 1065353216 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 2 : i32, waddr_mul = 39 : i32}

      // uniform 2: m -> exact saxpy logical element count ra3.
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 3 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 32 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // uniform 3: qpu_id -> exact saxpy qpu_id ra4.
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 4 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 32 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // uniform 4: num_qpus -> exact saxpy num_qpus ra5.
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 5 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 32 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
      // qasm:32
      vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<shl>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 4 : i32, small_imm = 6 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // qasm:33
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 8 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<r0>, add_b = #vc4.qpu_mux<r0>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // qasm:36
      vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 35 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<shl>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 5 : i32, small_imm = 6 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // qasm:37
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, write_swap, waddr_add = 20 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<r3>, add_b = #vc4.qpu_mux<r3>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // qasm:40
      vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 33 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<shl>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 3 : i32, small_imm = 2 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // qasm:41
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 9 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<r1>, add_b = #vc4.qpu_mux<r1>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // qasm:44
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 10 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<r0>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // qasm:45
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 11 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 1 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<r0>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // qasm:48
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 12 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 4 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // qasm:51
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 33 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 9 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // qasm:52
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, set_flags, waddr_add = 33 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<sub>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 8 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<r1>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // qasm:53
      vc4.qpu.branch attributes {cond = #vc4.branch_cond<any_c_clear>, relative = true, use_reg = false, raddr_a = 0 : i32, immediate = 392 : i32, waddr_add = 39 : i32, waddr_mul = 39 : i32} {
            vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
            vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
            vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
          }

      // qasm:61
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 33 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 8 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // qasm:62
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 35 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<sub>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 9 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<r1>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // qasm:66
      vc4.qpu.ldi <splat32> {value = 64 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 33 : i32, waddr_mul = 39 : i32}

      // qasm:67
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, set_flags, waddr_add = 39 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<sub>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<r3>, add_b = #vc4.qpu_mux<r1>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // qasm:68
      vc4.qpu.branch attributes {cond = #vc4.branch_cond<any_c_clear>, relative = true, use_reg = false, raddr_a = 0 : i32, immediate = 80 : i32, waddr_add = 39 : i32, waddr_mul = 39 : i32} {
            vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
            vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
            vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
          }

      // qasm:74
      vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 33 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<shr>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, small_imm = 2 : i32, add_a = #vc4.qpu_mux<r3>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // qasm:75
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 13 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<r1>, add_b = #vc4.qpu_mux<r1>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // qasm:76
      vc4.qpu.branch attributes {cond = #vc4.branch_cond<always>, relative = true, use_reg = false, raddr_a = 0 : i32, immediate = 48 : i32, waddr_add = 39 : i32, waddr_mul = 39 : i32} {
            vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
            vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
            vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
          }

      // qasm:82
      vc4.qpu.ldi <splat32> {value = 16 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 33 : i32, waddr_mul = 39 : i32}

      // qasm:83
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 13 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<r1>, add_b = #vc4.qpu_mux<r1>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // qasm:89
      vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 33 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<shl>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 38 : i32, small_imm = 2 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // qasm:95
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 56 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 10 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<r1>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // qasm:96
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 56 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 11 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<r1>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // qasm:97
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // qasm:100
      vc4.qpu.bundle {sig = #vc4.qpu_signal<ldtmu0>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // qasm:104
      vc4.qpu.bundle {sig = #vc4.qpu_signal<ldtmu0>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<always>, waddr_add = 39 : i32, waddr_mul = 34 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<fmul>, raddr_a = 2 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r4>, mul_b = #vc4.qpu_mux<a>}

      // qasm:107
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 34 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<fadd>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<r2>, add_b = #vc4.qpu_mux<r4>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // qasm:110
      vc4.qpu.ldi <splat32> {value = 1055232 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 35 : i32, waddr_mul = 39 : i32}

      // qasm:111
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 49 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 12 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<r3>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // qasm:112
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 48 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<r2>, add_b = #vc4.qpu_mux<r2>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // qasm:113
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 39 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 50 : i32, add_a = #vc4.qpu_mux<r0>, add_b = #vc4.qpu_mux<r1>, mul_a = #vc4.qpu_mux<r2>, mul_b = #vc4.qpu_mux<r3>}

      // qasm:120
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 33 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 13 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // qasm:121
      vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 33 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<shl>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, small_imm = 8 : i32, add_a = #vc4.qpu_mux<r1>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // qasm:122
      vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 33 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<shl>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, small_imm = 8 : i32, add_a = #vc4.qpu_mux<r1>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // qasm:123
      vc4.qpu.ldi <splat32> {value = -2139078656 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 35 : i32, waddr_mul = 39 : i32}

      // qasm:124
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 34 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<r3>, add_b = #vc4.qpu_mux<r1>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // qasm:125
      vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 33 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<shl>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 12 : i32, small_imm = 7 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // qasm:126
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 49 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<r2>, add_b = #vc4.qpu_mux<r1>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // qasm:127
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 50 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 11 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // qasm:128
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 39 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 50 : i32, add_a = #vc4.qpu_mux<r0>, add_b = #vc4.qpu_mux<r1>, mul_a = #vc4.qpu_mux<r2>, mul_b = #vc4.qpu_mux<r3>}

      // qasm:131
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 8 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 8 : i32, raddr_b = 20 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // qasm:132
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 10 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 10 : i32, raddr_b = 20 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // qasm:133
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 11 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 11 : i32, raddr_b = 20 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // qasm:136
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 33 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 9 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // qasm:137
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, set_flags, waddr_add = 33 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<sub>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 8 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<r1>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // qasm:138
      vc4.qpu.branch attributes {cond = #vc4.branch_cond<any_c_set>, relative = true, use_reg = false, raddr_a = 0 : i32, immediate = -328 : i32, waddr_add = 39 : i32, waddr_mul = 39 : i32} {
            vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
            vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
            vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
          }

      // qasm:146
      vc4.qpu.bundle {sig = #vc4.qpu_signal<thrend>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // qasm:147
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // qasm:148
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

    }
  }
}
