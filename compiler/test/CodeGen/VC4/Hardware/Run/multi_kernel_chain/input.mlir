// RUN: vc4-opt %s --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o /dev/null

// M2 two-kernel hardware fixture. Kernel A writes the memory_output
// pattern into a heap buffer; kernel B copies that buffer to the result.
// The CPU oracle checks the final result against the memory_output pattern.

vc4.module @multi_kernel_chain {

  vc4.func @memory_output_kernel() attributes {
    domain = #vc4.execution_domain<qpu>,
    form = #vc4.function_form<scheduled>,
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "memory_output",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 3 : i32,
      args = [
        {name = "out", kind = "buffer", direction = "out", elem_type = "u32", uniform_index = 0 : i32}
      ],
      builtins = [
        {name = "qpu_id", kind = #vc4.builtin_kind<qpu_num>, materialization = "uniform_suffix", uniform_index = 1 : i32},
        {name = "num_qpus", kind = #vc4.builtin_kind<num_qpus>, materialization = "uniform_suffix", uniform_index = 2 : i32}
      ]
    }
  } {
    // mov ra0, unif      ; output base address
    vc4.qpu.bundle {
      sig = #vc4.qpu_signal<none>,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      waddr_add = 0 : i32,
      waddr_mul = 39 : i32,
      op_add = #vc4.add_opcode<or>,
      op_mul = #vc4.mul_opcode<nop>,
      raddr_a = 32 : i32,
      raddr_b = 0 : i32,
      add_a = #vc4.qpu_mux<a>,
      add_b = #vc4.qpu_mux<a>,
      mul_a = #vc4.qpu_mux<r0>,
      mul_b = #vc4.qpu_mux<r1>
    }

    // mov ra1, unif      ; qpu_id
    vc4.qpu.bundle {
      sig = #vc4.qpu_signal<none>,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      waddr_add = 1 : i32,
      waddr_mul = 39 : i32,
      op_add = #vc4.add_opcode<or>,
      op_mul = #vc4.mul_opcode<nop>,
      raddr_a = 32 : i32,
      raddr_b = 0 : i32,
      add_a = #vc4.qpu_mux<a>,
      add_b = #vc4.qpu_mux<a>,
      mul_a = #vc4.qpu_mux<r0>,
      mul_b = #vc4.qpu_mux<r1>
    }

    // mov ra2, unif      ; num_qpus, carried for ABI coverage
    vc4.qpu.bundle {
      sig = #vc4.qpu_signal<none>,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      waddr_add = 2 : i32,
      waddr_mul = 39 : i32,
      op_add = #vc4.add_opcode<or>,
      op_mul = #vc4.mul_opcode<nop>,
      raddr_a = 32 : i32,
      raddr_b = 0 : i32,
      add_a = #vc4.qpu_mux<a>,
      add_b = #vc4.qpu_mux<a>,
      mul_a = #vc4.qpu_mux<r0>,
      mul_b = #vc4.qpu_mux<r1>
    }

    // shl r0, ra1, 6     ; qpu byte offset = qpu_id * 16 lanes * 4 bytes
    vc4.qpu.bundle {
      sig = #vc4.qpu_signal<small_imm>,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      waddr_add = 32 : i32,
      waddr_mul = 39 : i32,
      op_add = #vc4.add_opcode<shl>,
      op_mul = #vc4.mul_opcode<nop>,
      raddr_a = 1 : i32,
      small_imm = 6 : i32,
      add_a = #vc4.qpu_mux<a>,
      add_b = #vc4.qpu_mux<b>,
      mul_a = #vc4.qpu_mux<r0>,
      mul_b = #vc4.qpu_mux<r1>
    }

    // add ra3, ra0, r0   ; per-QPU output address
    vc4.qpu.bundle {
      sig = #vc4.qpu_signal<none>,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      waddr_add = 3 : i32,
      waddr_mul = 39 : i32,
      op_add = #vc4.add_opcode<add>,
      op_mul = #vc4.mul_opcode<nop>,
      raddr_a = 0 : i32,
      raddr_b = 0 : i32,
      add_a = #vc4.qpu_mux<a>,
      add_b = #vc4.qpu_mux<r0>,
      mul_a = #vc4.qpu_mux<r0>,
      mul_b = #vc4.qpu_mux<r1>
    }

    // mov r0, ra1        ; output data vector, all lanes = qpu_id
    vc4.qpu.bundle {
      sig = #vc4.qpu_signal<none>,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      waddr_add = 32 : i32,
      waddr_mul = 39 : i32,
      op_add = #vc4.add_opcode<or>,
      op_mul = #vc4.mul_opcode<nop>,
      raddr_a = 1 : i32,
      raddr_b = 0 : i32,
      add_a = #vc4.qpu_mux<a>,
      add_b = #vc4.qpu_mux<a>,
      mul_a = #vc4.qpu_mux<r0>,
      mul_b = #vc4.qpu_mux<r1>
    }

    // mov r3, vpm_setup(1, 1, h32(0))
    vc4.qpu.ldi <splat32> {
      value = 1055232 : i32,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      waddr_add = 35 : i32,
      waddr_mul = 39 : i32
    }

    // add vw_setup, r3, ra1
    vc4.qpu.vpmvcd_setup {
      side = #vc4.vpmvcd_side<write>,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      write_swap,
      op_add = #vc4.add_opcode<add>,
      op_mul = #vc4.mul_opcode<nop>,
      raddr_a = 1 : i32,
      raddr_b = 0 : i32,
      add_a = #vc4.qpu_mux<r3>,
      add_b = #vc4.qpu_mux<a>,
      mul_a = #vc4.qpu_mux<r0>,
      mul_b = #vc4.qpu_mux<r1>
    }

    // mov vpm, r0
    vc4.qpu.bundle {
      sig = #vc4.qpu_signal<none>,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      waddr_add = 48 : i32,
      waddr_mul = 39 : i32,
      op_add = #vc4.add_opcode<or>,
      op_mul = #vc4.mul_opcode<nop>,
      raddr_a = 0 : i32,
      raddr_b = 0 : i32,
      add_a = #vc4.qpu_mux<r0>,
      add_b = #vc4.qpu_mux<r0>,
      mul_a = #vc4.qpu_mux<r0>,
      mul_b = #vc4.qpu_mux<r1>
    }

    // mov -, vw_wait
    vc4.qpu.vpmvcd_wait {side = #vc4.vpmvcd_side<write>}

    // shl r1, ra1, 7     ; VDW setup row offset
    vc4.qpu.bundle {
      sig = #vc4.qpu_signal<small_imm>,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      waddr_add = 33 : i32,
      waddr_mul = 39 : i32,
      op_add = #vc4.add_opcode<shl>,
      op_mul = #vc4.mul_opcode<nop>,
      raddr_a = 1 : i32,
      small_imm = 7 : i32,
      add_a = #vc4.qpu_mux<a>,
      add_b = #vc4.qpu_mux<b>,
      mul_a = #vc4.qpu_mux<r0>,
      mul_b = #vc4.qpu_mux<r1>
    }

    // mov r2, vdw_setup_0(1, 16, dma_h32(0, 0))
    vc4.qpu.ldi <splat32> {
      value = -2138030080 : i32,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      waddr_add = 34 : i32,
      waddr_mul = 39 : i32
    }

    // add vw_setup, r2, r1
    vc4.qpu.vpmvcd_setup {
      side = #vc4.vpmvcd_side<write>,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      write_swap,
      op_add = #vc4.add_opcode<add>,
      op_mul = #vc4.mul_opcode<nop>,
      raddr_a = 0 : i32,
      raddr_b = 0 : i32,
      add_a = #vc4.qpu_mux<r2>,
      add_b = #vc4.qpu_mux<r1>,
      mul_a = #vc4.qpu_mux<r0>,
      mul_b = #vc4.qpu_mux<r1>
    }

    // mov vw_addr, ra3
    vc4.qpu.vpmvcd_addr {
      side = #vc4.vpmvcd_side<write>,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      write_swap,
      op_add = #vc4.add_opcode<or>,
      op_mul = #vc4.mul_opcode<nop>,
      raddr_a = 3 : i32,
      raddr_b = 0 : i32,
      add_a = #vc4.qpu_mux<a>,
      add_b = #vc4.qpu_mux<a>,
      mul_a = #vc4.qpu_mux<r0>,
      mul_b = #vc4.qpu_mux<r1>
    }

    // mov -, vw_wait
    vc4.qpu.vpmvcd_wait {side = #vc4.vpmvcd_side<write>}

    // thrend
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

    // nop
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

    // nop
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

  vc4.func @read_nop_write_kernel() attributes {
    domain = #vc4.execution_domain<qpu>,
    form = #vc4.function_form<scheduled>,
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "read_nop_write",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 5 : i32,
      args = [
        {name = "input", kind = "buffer", direction = "in", elem_type = "u32", uniform_index = 0 : i32},
        {name = "result", kind = "buffer", direction = "out", elem_type = "u32", uniform_index = 1 : i32},
        {name = "n", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 2 : i32}
      ],
      builtins = [
        {name = "qpu_id", kind = #vc4.builtin_kind<qpu_num>, materialization = "uniform_suffix", uniform_index = 3 : i32},
        {name = "num_qpus", kind = #vc4.builtin_kind<num_qpus>, materialization = "uniform_suffix", uniform_index = 4 : i32}
      ]
    }
  } {
    // ra0 = input base
    vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 0 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 32 : i32, small_imm = 0 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
    // ra1 = result base
    vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 1 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 32 : i32, small_imm = 0 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
    // ra2 = n
    vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 2 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 32 : i32, small_imm = 0 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
    // ra3 = qpu_id
    vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 3 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 32 : i32, small_imm = 0 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
    // ra4 = num_qpus
    vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 4 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 32 : i32, small_imm = 0 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

    // r0 = qpu_id << 6, byte offset for one 16-lane u32 vector.
    vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<shl>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 3 : i32, small_imm = 6 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
    // ra5 = input + byte_offset
    vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 5 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 0 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<r0>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
    // ra6 = result + byte_offset
    vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 6 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 1 : i32, raddr_b = 0 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<r0>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
    // ra7 = qpu_id, used as the VPM row.
    vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 7 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 3 : i32, small_imm = 0 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

    // VDR DMA load one vector from input into VPM row qpu_id.
    vc4.qpu.ldi <splat32> {value = -2147414016 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 34 : i32, waddr_mul = 32 : i32}
    vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 33 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<shl>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 7 : i32, small_imm = 4 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
    vc4.qpu.vpmvcd_setup {
      side = #vc4.vpmvcd_side<read>,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      op_add = #vc4.add_opcode<add>,
      op_mul = #vc4.mul_opcode<nop>,
      raddr_a = 0 : i32,
      raddr_b = 0 : i32,
      add_a = #vc4.qpu_mux<r2>,
      add_b = #vc4.qpu_mux<r1>,
      mul_a = #vc4.qpu_mux<r0>,
      mul_b = #vc4.qpu_mux<r1>
    }
    vc4.qpu.vpmvcd_addr {
      side = #vc4.vpmvcd_side<read>,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      op_add = #vc4.add_opcode<add>,
      op_mul = #vc4.mul_opcode<nop>,
      raddr_a = 5 : i32,
      small_imm = 0 : i32,
      add_a = #vc4.qpu_mux<a>,
      add_b = #vc4.qpu_mux<b>,
      mul_a = #vc4.qpu_mux<r0>,
      mul_b = #vc4.qpu_mux<r1>
    }
    vc4.qpu.vpmvcd_wait {side = #vc4.vpmvcd_side<read>}

    // VPM read setup, three latency nops, then read VPM into r0.
    vc4.qpu.ldi <splat32> {value = 1055232 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 34 : i32, waddr_mul = 32 : i32}
    vc4.qpu.vpmvcd_setup {
      side = #vc4.vpmvcd_side<read>,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      op_add = #vc4.add_opcode<add>,
      op_mul = #vc4.mul_opcode<nop>,
      raddr_a = 7 : i32,
      raddr_b = 0 : i32,
      add_a = #vc4.qpu_mux<r2>,
      add_b = #vc4.qpu_mux<a>,
      mul_a = #vc4.qpu_mux<r0>,
      mul_b = #vc4.qpu_mux<r1>
    }
    vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
    vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
    vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
    vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 48 : i32, small_imm = 0 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
    vc4.qpu.vpmvcd_wait {side = #vc4.vpmvcd_side<write>}

    // VPM write unchanged r0 into the same row.
    vc4.qpu.ldi <splat32> {value = 1055232 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 35 : i32, waddr_mul = 32 : i32}
    vc4.qpu.vpmvcd_setup {
      side = #vc4.vpmvcd_side<write>,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      op_add = #vc4.add_opcode<add>,
      op_mul = #vc4.mul_opcode<nop>,
      raddr_a = 7 : i32,
      raddr_b = 0 : i32,
      add_a = #vc4.qpu_mux<r3>,
      add_b = #vc4.qpu_mux<a>,
      mul_a = #vc4.qpu_mux<r0>,
      mul_b = #vc4.qpu_mux<r1>
    }
    vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 48 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, small_imm = 0 : i32, add_a = #vc4.qpu_mux<r0>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
    vc4.qpu.vpmvcd_wait {side = #vc4.vpmvcd_side<write>}

    // VDW DMA store VPM row to result.
    vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 33 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<shl>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 7 : i32, small_imm = 7 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
    vc4.qpu.ldi <splat32> {value = -2138030080 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 34 : i32, waddr_mul = 32 : i32}
    vc4.qpu.vpmvcd_setup {
      side = #vc4.vpmvcd_side<write>,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      op_add = #vc4.add_opcode<add>,
      op_mul = #vc4.mul_opcode<nop>,
      raddr_a = 0 : i32,
      raddr_b = 0 : i32,
      add_a = #vc4.qpu_mux<r2>,
      add_b = #vc4.qpu_mux<r1>,
      mul_a = #vc4.qpu_mux<r0>,
      mul_b = #vc4.qpu_mux<r1>
    }
    vc4.qpu.vpmvcd_addr {
      side = #vc4.vpmvcd_side<write>,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      op_add = #vc4.add_opcode<add>,
      op_mul = #vc4.mul_opcode<nop>,
      raddr_a = 6 : i32,
      small_imm = 0 : i32,
      add_a = #vc4.qpu_mux<a>,
      add_b = #vc4.qpu_mux<b>,
      mul_a = #vc4.qpu_mux<r0>,
      mul_b = #vc4.qpu_mux<r1>
    }
    vc4.qpu.vpmvcd_wait {side = #vc4.vpmvcd_side<write>}

    // Thread-end epilogue: Program End plus two safe delay-slot instructions.
    vc4.qpu.bundle {sig = #vc4.qpu_signal<thrend>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
    vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
    vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
  }

}
