// RUN: vc4-opt %s --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o /dev/null

// Final-stage scheduled-sink input for the saxpy_tmu hardware golden.
//
// Semantics:
//   For one 16-lane vector per active QPU:
//     y[i] = alpha * x[i] + y[i]
//
// Physical uniform stream per QPU:
//   [0] x base address
//   [1] y base address/result address
//   [2] alpha scalar, carried as one f32/u32 uniform word
//   [3] n element count, carried for ABI coverage
//   [4] qpu_id
//   [5] num_qpus
//
// This differs from saxpy_basic only in the input path: x and y come from TMU0
// direct memory lookups instead of VDR DMA loads. The arithmetic stays as
// fmul/fadd, and the output path remains VPM -> VDW DMA store.

vc4.module @saxpy_tmu {
  vc4.func @saxpy_tmu_kernel() attributes {
    domain = #vc4.execution_domain<qpu>,
    form = #vc4.function_form<scheduled>,
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "saxpy_tmu",
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
    // ra0 = x base
    vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 0 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 32 : i32, raddr_b = 39 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

    // ra1 = y base/result base
    vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 1 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 32 : i32, raddr_b = 39 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

    // ra2 = alpha
    vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 2 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 32 : i32, raddr_b = 39 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

    // ra3 = n, consumed for ABI coverage
    vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 3 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 32 : i32, raddr_b = 39 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

    // ra4 = qpu_id
    vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 4 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 32 : i32, raddr_b = 39 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

    // ra5 = num_qpus
    vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 5 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 32 : i32, raddr_b = 39 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

    // r0 = qpu_id * 64 bytes
    vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<shl>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 4 : i32, small_imm = 6 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

    // ra6 = x + offset
    vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 6 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 39 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<r0>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

    // ra7 = y + offset
    vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 7 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 1 : i32, raddr_b = 39 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<r0>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

    // ra8 = VPM row = qpu_id
    vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 8 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 4 : i32, raddr_b = 39 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

    // r1 = elem_num * 4 byte lane offset
    vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 33 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<shl>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 38 : i32, small_imm = 2 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

    // t0s = x + offset + lane*4, issuing direct TMU0 memory request for x
    vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 56 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 6 : i32, raddr_b = 39 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<r1>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

    // Spacer while TMU request is in flight.
    vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 39 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<r0>, add_b = #vc4.qpu_mux<r1>, mul_a = #vc4.qpu_mux<r2>, mul_b = #vc4.qpu_mux<r3>}

    // Signal TMU0 receive; x arrives in r4 for the following instruction.
    vc4.qpu.bundle {sig = #vc4.qpu_signal<ldtmu0>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 39 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<r0>, add_b = #vc4.qpu_mux<r1>, mul_a = #vc4.qpu_mux<r2>, mul_b = #vc4.qpu_mux<r3>}

    // r0 = x vector from r4
    vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<r4>, add_b = #vc4.qpu_mux<r4>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

    // t0s = y + offset + lane*4, issuing direct TMU0 memory request for y
    vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 56 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 7 : i32, raddr_b = 39 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<r1>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

    // Spacer while TMU request is in flight.
    vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 39 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<r0>, add_b = #vc4.qpu_mux<r1>, mul_a = #vc4.qpu_mux<r2>, mul_b = #vc4.qpu_mux<r3>}

    // Signal TMU0 receive; y arrives in r4 for the following instruction.
    vc4.qpu.bundle {sig = #vc4.qpu_signal<ldtmu0>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 39 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<r0>, add_b = #vc4.qpu_mux<r1>, mul_a = #vc4.qpu_mux<r2>, mul_b = #vc4.qpu_mux<r3>}

    // r1 = y vector from r4
    vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 33 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<r4>, add_b = #vc4.qpu_mux<r4>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

    // Canonical SAXPY arithmetic:
    //   r2 = alpha * x
    //   r2 = r2 + y
    vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<always>, waddr_add = 39 : i32, waddr_mul = 34 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<fmul>, raddr_a = 2 : i32, raddr_b = 0 : i32, add_a = #vc4.qpu_mux<r0>, add_b = #vc4.qpu_mux<r1>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<a>}
    vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 34 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<fadd>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<r2>, add_b = #vc4.qpu_mux<r1>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

    // r3 = vpm_setup(1, 1, h32(0))
    vc4.qpu.ldi <splat32> {value = 1055232 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 35 : i32, waddr_mul = 39 : i32}

    // vw_setup = r3 + row
    vc4.qpu.vpmvcd_setup {
      side = #vc4.vpmvcd_side<write>,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      op_add = #vc4.add_opcode<add>,
      op_mul = #vc4.mul_opcode<nop>,
      raddr_a = 8 : i32,
      raddr_b = 39 : i32,
      add_a = #vc4.qpu_mux<r3>,
      add_b = #vc4.qpu_mux<a>,
      mul_a = #vc4.qpu_mux<r0>,
      mul_b = #vc4.qpu_mux<r1>
    }

    // vpm = r2
    vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 48 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<r2>, add_b = #vc4.qpu_mux<r2>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

    // read vw_wait
    vc4.qpu.vpmvcd_wait {side = #vc4.vpmvcd_side<write>}

    // r1 = row << 7, matching vdw_setup_0 row offset encoding.
    vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 33 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<shl>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 8 : i32, small_imm = 7 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

    // r2 = vdw_setup_0(1, 16, dma_h32(0, 0))
    vc4.qpu.ldi <splat32> {value = -2138030080 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 34 : i32, waddr_mul = 39 : i32}

    // vw_setup = r2 + row_offset
    vc4.qpu.vpmvcd_setup {
      side = #vc4.vpmvcd_side<write>,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      op_add = #vc4.add_opcode<add>,
      op_mul = #vc4.mul_opcode<nop>,
      raddr_a = 0 : i32,
      raddr_b = 1 : i32,
      add_a = #vc4.qpu_mux<r2>,
      add_b = #vc4.qpu_mux<r1>,
      mul_a = #vc4.qpu_mux<r0>,
      mul_b = #vc4.qpu_mux<r1>
    }

    // vw_addr = y + offset
    vc4.qpu.vpmvcd_addr {
      side = #vc4.vpmvcd_side<write>,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      op_add = #vc4.add_opcode<or>,
      op_mul = #vc4.mul_opcode<nop>,
      raddr_a = 7 : i32,
      raddr_b = 39 : i32,
      add_a = #vc4.qpu_mux<a>,
      add_b = #vc4.qpu_mux<a>,
      mul_a = #vc4.qpu_mux<r0>,
      mul_b = #vc4.qpu_mux<r1>
    }

    // read vw_wait
    vc4.qpu.vpmvcd_wait {side = #vc4.vpmvcd_side<write>}

    // Thread-end epilogue.
    vc4.qpu.bundle {sig = #vc4.qpu_signal<thrend>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<r0>, add_b = #vc4.qpu_mux<r1>, mul_a = #vc4.qpu_mux<r2>, mul_b = #vc4.qpu_mux<r3>}
    vc4.qpu.ldi <splat32> {value = 0 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 34 : i32, waddr_mul = 35 : i32}
    vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 39 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<r0>, add_b = #vc4.qpu_mux<r1>, mul_a = #vc4.qpu_mux<r2>, mul_b = #vc4.qpu_mux<r3>}
  }
}
