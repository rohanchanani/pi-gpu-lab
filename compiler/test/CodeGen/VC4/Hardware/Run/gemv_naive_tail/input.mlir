// RUN: vc4-opt %s --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o /dev/null

// Scheduled VC4 GEMV tail fixture.
//
// CUDA-like mapping:
//   one QPU request = one 16-lane warp;
//   ELEMENT_NUMBER = lane / output-row offset inside a 16-row tile;
//   qpu_id = logical 16-row tile id supplied by the M2 launcher;
//   num_qpus = total logical requests for the launch;
//   row = qpu_id * 16 + ELEMENT_NUMBER;
//   each lane computes one output y[row] by looping over N columns.
//
// Semantics:
//   y[row] = sum_{col=0..N-1} A[row*N + col] * x[col]
// for row in [0, M). The final partial 16-row tile uses dynamic VDW DEPTH
// to avoid overwriting sentinel guard rows.

module attributes {
  "vc4.hardware_run_test.name" = "gemv_naive_tail",
  "vc4.hardware_run_test.kind" = "hardware-run-reference",
  "vc4.hardware_run_test.reference_kernel" = "reference/gemv_naive_tail.qasm",
  "vc4.hardware_run_test.launch_abi" = {
    public_name = "gemv_naive_tail",
    tail_policy = "tail_safe",
    uniform_words_per_qpu = 7 : i32,
    args = [
      {name = "a", kind = "buffer", direction = "in", elem_type = "f32", uniform_index = 0 : i32},
      {name = "x", kind = "buffer", direction = "in", elem_type = "f32", uniform_index = 1 : i32},
      {name = "y", kind = "buffer", direction = "out", elem_type = "f32", uniform_index = 2 : i32},
      {name = "m", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 3 : i32},
      {name = "n", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 4 : i32}
    ],
    builtins = [
      {name = "qpu_id", kind = "qpu_num", materialization = "uniform_suffix", uniform_index = 5 : i32},
      {name = "num_qpus", kind = "num_qpus", materialization = "uniform_suffix", uniform_index = 6 : i32}
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
          {name = "a", kind = "buffer", direction = "in", elem_type = "f32", uniform_index = 0 : i32},
          {name = "x", kind = "buffer", direction = "in", elem_type = "f32", uniform_index = 1 : i32},
          {name = "y", kind = "buffer", direction = "out", elem_type = "f32", uniform_index = 2 : i32},
          {name = "m", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 3 : i32},
          {name = "n", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 4 : i32}
        ],
        builtins = [
          {name = "qpu_id", kind = #vc4.builtin_kind<qpu_num>, materialization = "uniform_suffix", uniform_index = 5 : i32},
          {name = "num_qpus", kind = #vc4.builtin_kind<num_qpus>, materialization = "uniform_suffix", uniform_index = 6 : i32}
        ]
      }
    } {
      // uniform 0: A base -> ra0
      // slot 0
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 0 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 32 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // uniform 1: x base -> ra1
      // slot 1
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 1 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 32 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // uniform 2: y base -> ra2
      // slot 2
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 2 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 32 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // uniform 3: M -> ra3
      // slot 3
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 3 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 32 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // uniform 4: N -> ra4
      // slot 4
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 4 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 32 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // uniform 5: qpu_id -> ra5
      // slot 5
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 5 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 32 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // uniform 6: num_qpus -> ra6
      // slot 6
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 6 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 32 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // r3 = num_qpus * 16 rows
      // slot 7
      vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 35 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<shl>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 6 : i32, small_imm = 4 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // rb20 = row-block stride in rows
      // slot 8
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, write_swap, waddr_add = 20 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<r3>, add_b = #vc4.qpu_mux<r3>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // r3 = num_qpus * 64 bytes
      // slot 9
      vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 35 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<shl>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 6 : i32, small_imm = 6 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // rb24 = row-block stride in bytes
      // slot 10
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, write_swap, waddr_add = 24 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<r3>, add_b = #vc4.qpu_mux<r3>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // rb21 = N
      // slot 11
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, write_swap, waddr_add = 21 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 4 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // r3 = 16
      // slot 12
      vc4.qpu.ldi <splat32> {value = 16 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 35 : i32, waddr_mul = 39 : i32}

      // rb23 = 16
      // slot 13
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, write_swap, waddr_add = 23 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<r3>, add_b = #vc4.qpu_mux<r3>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // r0 = qpu_id * 16 = row_base
      // slot 14
      vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<shl>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 5 : i32, small_imm = 4 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // ra8 = row_base
      // slot 15
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 8 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<r0>, add_b = #vc4.qpu_mux<r0>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // ra12 = VPM staging row
      // slot 16
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 12 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 5 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // r1 = row_base * 4
      // slot 17
      vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 33 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<shl>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 8 : i32, small_imm = 2 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // ra11 = y + row_base*4
      // slot 18
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 11 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 2 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<r1>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // label row_loop
      // r1 = M
      // slot 19
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 33 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 3 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // r3 = M - row_base
      // slot 20
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 35 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<sub>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 8 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<r1>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // r1 = 16
      // slot 21
      vc4.qpu.ldi <splat32> {value = 16 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 33 : i32, waddr_mul = 39 : i32}

      // set flags for remaining_rows - 16
      // slot 22
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, set_flags, waddr_add = 33 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<sub>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<r3>, add_b = #vc4.qpu_mux<r1>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // if remaining_rows >= 16, use full vector depth
      // slot 23
      vc4.qpu.branch attributes {cond = #vc4.branch_cond<any_c_clear>, relative = true, use_reg = false, raddr_a = 0 : i32, immediate = 72 : i32, waddr_add = 39 : i32, waddr_mul = 39 : i32} {
        vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
        vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
        vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
      }

      // tail: ra13 = remaining_rows
      // slot 27
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 13 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<r3>, add_b = #vc4.qpu_mux<r3>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // jump to have_store_count
      // slot 28
      vc4.qpu.branch attributes {cond = #vc4.branch_cond<always>, relative = true, use_reg = false, raddr_a = 0 : i32, immediate = 48 : i32, waddr_add = 39 : i32, waddr_mul = 39 : i32} {
        vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
        vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
        vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
      }

      // label full_tile
      // full: r1 = 16
      // slot 32
      vc4.qpu.ldi <splat32> {value = 16 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 33 : i32, waddr_mul = 39 : i32}

      // ra13 = 16
      // slot 33
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 13 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<r1>, add_b = #vc4.qpu_mux<r1>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // label have_store_count
      // r2 = 0.0 accumulator
      // slot 34
      vc4.qpu.ldi <splat32> {value = 0 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 34 : i32, waddr_mul = 39 : i32}

      // ra15 = col = 0
      // slot 35
      vc4.qpu.ldi <splat32> {value = 0 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 15 : i32, waddr_mul = 39 : i32}

      // label col_check
      // r1 = N
      // slot 36
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 33 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 4 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // set flags for col - N
      // slot 37
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, set_flags, waddr_add = 33 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<sub>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 15 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<r1>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // if col >= N, store accumulated rows
      // slot 38
      vc4.qpu.branch attributes {cond = #vc4.branch_cond<any_c_clear>, relative = true, use_reg = false, raddr_a = 0 : i32, immediate = 184 : i32, waddr_add = 39 : i32, waddr_mul = 39 : i32} {
        vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
        vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
        vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
      }

      // r0 = elem_num * N
      // slot 42
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<always>, waddr_add = 39 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<mul24>, raddr_a = 38 : i32, raddr_b = 21 : i32, add_a = #vc4.qpu_mux<r0>, add_b = #vc4.qpu_mux<r1>, mul_a = #vc4.qpu_mux<a>, mul_b = #vc4.qpu_mux<b>}

      // r1 = row_base * N
      // slot 43
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<always>, waddr_add = 39 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<mul24>, raddr_a = 8 : i32, raddr_b = 21 : i32, add_a = #vc4.qpu_mux<r0>, add_b = #vc4.qpu_mux<r1>, mul_a = #vc4.qpu_mux<a>, mul_b = #vc4.qpu_mux<b>}

      // r0 = (row_base + elem_num) * N
      // slot 44
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<r0>, add_b = #vc4.qpu_mux<r1>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // r0 += col
      // slot 45
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 15 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<r0>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // r0 *= sizeof(float)
      // slot 46
      vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<shl>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, small_imm = 2 : i32, add_a = #vc4.qpu_mux<r0>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // TMU0 request A[row, col]
      // slot 47
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 56 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<r0>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // r1 = col * 4
      // slot 48
      vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 33 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<shl>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 15 : i32, small_imm = 2 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // TMU0 request x[col]
      // slot 49
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 56 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 1 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<r1>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // TMU spacer
      // slot 50
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // receive A into r4
      // slot 51
      vc4.qpu.bundle {sig = #vc4.qpu_signal<ldtmu0>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // r3 = A; receive x into r4
      // slot 52
      vc4.qpu.bundle {sig = #vc4.qpu_signal<ldtmu0>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 35 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<r4>, add_b = #vc4.qpu_mux<r4>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // r0 = A * x
      // slot 53
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<always>, waddr_add = 39 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<fmul>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<r0>, add_b = #vc4.qpu_mux<r1>, mul_a = #vc4.qpu_mux<r3>, mul_b = #vc4.qpu_mux<r4>}

      // acc += A*x
      // slot 54
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 34 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<fadd>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<r2>, add_b = #vc4.qpu_mux<r0>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // r1 = 1
      // slot 55
      vc4.qpu.ldi <splat32> {value = 1 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 33 : i32, waddr_mul = 39 : i32}

      // ++col
      // slot 56
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 15 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 15 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<r1>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // next column
      // slot 57
      vc4.qpu.branch attributes {cond = #vc4.branch_cond<always>, relative = true, use_reg = false, raddr_a = 0 : i32, immediate = -168 : i32, waddr_add = 39 : i32, waddr_mul = 39 : i32} {
        vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
        vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
        vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
      }

      // label store
      // r3 = vpm_setup(1,1,h32(0)) / 0x00101a00
      // slot 61
      vc4.qpu.ldi <splat32> {value = 1055232 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 35 : i32, waddr_mul = 39 : i32}

      // vw_setup = VPM row setup + ra12
      // slot 62
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 49 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 12 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<r3>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // stage acc vector to VPM
      // slot 63
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 48 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<r2>, add_b = #vc4.qpu_mux<r2>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // wait for VPM write
      // slot 64
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 39 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 50 : i32, add_a = #vc4.qpu_mux<r0>, add_b = #vc4.qpu_mux<r1>, mul_a = #vc4.qpu_mux<r2>, mul_b = #vc4.qpu_mux<r3>}

      // r1 = store_count
      // slot 65
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 33 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 13 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // r1 = store_count << 8
      // slot 66
      vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 33 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<shl>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, small_imm = 8 : i32, add_a = #vc4.qpu_mux<r1>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // r1 = store_count << 16
      // slot 67
      vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 33 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<shl>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, small_imm = 8 : i32, add_a = #vc4.qpu_mux<r1>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // r3 = 0x80804000 VDW base descriptor
      // slot 68
      vc4.qpu.ldi <splat32> {value = -2139078656 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 35 : i32, waddr_mul = 39 : i32}

      // r2 = VDW descriptor with depth
      // slot 69
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 34 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<r3>, add_b = #vc4.qpu_mux<r1>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // r1 = VPM row << 7
      // slot 70
      vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 33 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<shl>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 12 : i32, small_imm = 7 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // vw_setup = VDW descriptor
      // slot 71
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 49 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<r2>, add_b = #vc4.qpu_mux<r1>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // vw_addr = y + row_base*4
      // slot 72
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 50 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 11 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // wait for VDW store
      // slot 73
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 39 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 50 : i32, add_a = #vc4.qpu_mux<r0>, add_b = #vc4.qpu_mux<r1>, mul_a = #vc4.qpu_mux<r2>, mul_b = #vc4.qpu_mux<r3>}

      // row_base += num_requests * 16
      // slot 74
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 8 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 8 : i32, raddr_b = 20 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // y address += num_requests * 64
      // slot 75
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 11 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 11 : i32, raddr_b = 24 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // r1 = M
      // slot 76
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 33 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 3 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // set flags for row_base - M
      // slot 77
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, set_flags, waddr_add = 33 : i32, waddr_mul = 39 : i32, op_add = #vc4.add_opcode<sub>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 8 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<r1>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // continue while row_base < M
      // slot 78
      vc4.qpu.branch attributes {cond = #vc4.branch_cond<any_c_set>, relative = true, use_reg = false, raddr_a = 0 : i32, immediate = -472 : i32, waddr_add = 39 : i32, waddr_mul = 39 : i32} {
        vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
        vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
        vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
      }

      // label end
      // thrend
      // slot 82
      vc4.qpu.bundle {sig = #vc4.qpu_signal<thrend>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // thread-end delay slot 1
      // slot 83
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

      // thread-end delay slot 2
      // slot 84
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

    }
  }
}
