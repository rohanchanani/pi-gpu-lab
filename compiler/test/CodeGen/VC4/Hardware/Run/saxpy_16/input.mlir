// RUN: vc4-opt %s --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o /dev/null

// Final-stage VC4 scheduled-sink input for the saxpy_16 hardware golden.
//
// Reference-side semantics:
//   for i in 0..n:
//     y[i] = alpha * x[i] + y[i]
//
// Test policy:
//   n is a runtime kernel argument.
//   The kernel/reference expects n to be a multiple of 16 * active_qpus.
//   Tail masking is intentionally not part of this test.
//   The generated launcher must not guard on n.
//
// Physical uniform stream per QPU:
//   [0] x base address
//   [1] y base address
//   [2] alpha as one f32/u32 word
//   [3] n element count
//   [4] qpu_id
//   [5] num_qpus
//
// This fixture intentionally uses the current dialect spelling for qpu.branch:
//   cond, relative, use_reg, immediate, raddr_a, waddr_add, waddr_mul.
// False write_swap is represented by omission, not by `write_swap = false`.

vc4.module @saxpy_16 {
  vc4.func @saxpy_16_kernel() attributes {
    domain = #vc4.execution_domain<qpu>,
    form = #vc4.function_form<scheduled>,
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "saxpy_16_launch",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 6 : i32,
      args = [
        {name = "x", kind = "buffer", direction = "in", elem_type = "f32", uniform_index = 0 : i32},
        {name = "y", kind = "buffer", direction = "inout", elem_type = "f32", uniform_index = 1 : i32},
        {name = "alpha", kind = "scalar", direction = "by_value", type = "f32", uniform_index = 2 : i32},
        {name = "n", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 3 : i32}
      ],
      builtins = [
        {name = "qpu_id", kind = #vc4.builtin_kind<qpu_num>, materialization = "uniform_suffix", uniform_index = 4 : i32},
        {name = "num_qpus", kind = #vc4.builtin_kind<num_qpus>, materialization = "uniform_suffix", uniform_index = 5 : i32}
      ]
    }
  } {
    // Scheduled sink body skeleton for the current verifier-facing fixture.
    // The trusted reference semantics are established by reference/saxpy_16.qasm
    // and the hardware result oracle. Candidate-side codegen should eventually
    // replace this schematic body with the exact scheduled QPU sink for that qasm.

    // Prologue / setup slots.
    "vc4.qpu.bundle"() <{
      sig = #vc4.qpu_signal<none>, pm = false,
      op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>,
      cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>,
      add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>,
      mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>,
      raddr_a = 0 : i32, raddr_b = 1 : i32,
      waddr_add = 39 : i32, waddr_mul = 39 : i32
    }> : () -> ()

    "vc4.qpu.bundle"() <{
      sig = #vc4.qpu_signal<none>, pm = false,
      op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>,
      cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>,
      add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>,
      mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>,
      raddr_a = 0 : i32, raddr_b = 1 : i32,
      waddr_add = 39 : i32, waddr_mul = 39 : i32
    }> : () -> ()

    "vc4.qpu.bundle"() <{
      sig = #vc4.qpu_signal<none>, pm = false,
      op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>,
      cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>,
      add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>,
      mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>,
      raddr_a = 0 : i32, raddr_b = 1 : i32,
      waddr_add = 39 : i32, waddr_mul = 39 : i32
    }> : () -> ()

    // Loop-back branch shape. The true emitted immediate is resolved by codegen.
    // The hardware branch fields are represented in dialect properties with the
    // long names below; do not use rel/reg/ws in this dialect syntax.
    "vc4.qpu.branch"() <{
      cond = #vc4.branch_cond<any_c_clear>,
      relative = true,
      use_reg = false,
      raddr_a = 0 : i32,
      waddr_add = 39 : i32,
      waddr_mul = 39 : i32,
      immediate = 0 : i32
    }> ({
      "vc4.qpu.bundle"() <{
        sig = #vc4.qpu_signal<none>, pm = false,
        op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>,
        cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>,
        add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>,
        mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>,
        raddr_a = 0 : i32, raddr_b = 1 : i32,
        waddr_add = 39 : i32, waddr_mul = 39 : i32
      }> : () -> ()

      "vc4.qpu.bundle"() <{
        sig = #vc4.qpu_signal<none>, pm = false,
        op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>,
        cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>,
        add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>,
        mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>,
        raddr_a = 0 : i32, raddr_b = 1 : i32,
        waddr_add = 39 : i32, waddr_mul = 39 : i32
      }> : () -> ()

      "vc4.qpu.bundle"() <{
        sig = #vc4.qpu_signal<none>, pm = false,
        op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>,
        cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>,
        add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>,
        mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>,
        raddr_a = 0 : i32, raddr_b = 1 : i32,
        waddr_add = 39 : i32, waddr_mul = 39 : i32
      }> : () -> ()
    }) {vc4.codegen.target = "loop"} : () -> ()

    // Thread end plus two delay slots. The final three instructions do not touch
    // uniforms, VPM/VDR/VDW, or regfile address 14.
    "vc4.qpu.bundle"() <{
      sig = #vc4.qpu_signal<thrend>, pm = false,
      op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>,
      cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>,
      add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>,
      mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>,
      raddr_a = 0 : i32, raddr_b = 1 : i32,
      waddr_add = 39 : i32, waddr_mul = 39 : i32
    }> : () -> ()

    "vc4.qpu.bundle"() <{
      sig = #vc4.qpu_signal<none>, pm = false,
      op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>,
      cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>,
      add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>,
      mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>,
      raddr_a = 0 : i32, raddr_b = 1 : i32,
      waddr_add = 39 : i32, waddr_mul = 39 : i32
    }> : () -> ()

    "vc4.qpu.bundle"() <{
      sig = #vc4.qpu_signal<none>, pm = false,
      op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>,
      cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>,
      add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>,
      mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>,
      raddr_a = 0 : i32, raddr_b = 1 : i32,
      waddr_add = 39 : i32, waddr_mul = 39 : i32
    }> : () -> ()
  }
}
